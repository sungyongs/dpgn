"""
Modules

Notations:
    N^v: The total number of nodes. i is an iteration var for node-level loop
    N^e: The total number of edges. k is an iteration var for edge-level loop.
    d_v: The dimension of node attributes
    d_e: The dimension of edge attributes
    d_g: The dimension of global attributes
    
    data.x: (N^v, d_v) tensor. data.x[i,:] is i-th node attributes.
    data.edge_index: (2, N^e) tensor (long type).
        s,r = data.edge_index[:,k] -> k-th edge connects s-node and r-node.
    data.edge_attr: (N^e, d_e) tensor. data.edge_attr[k,:] is k-th node attributes.
"""
import random

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.data import Data

from blocks import EdgeBlockInd, NodeBlockInd, GlobalBlockInd
from blocks import EdgeBlock, NodeBlock, GlobalBlock
from utils import graph_concat, copy_geometric_data, decompose_graph





class GNConv(nn.Module):
    """Graph Networks (https://arxiv.org/abs/1806.01261) module.
    (This code is mainly based on 
      https://github.com/deepmind/graph_nets/blob/master/graph_nets/modules.py)
      
    A graph network takes a graph as input and returns a graph as output.
    The input graph has edge- (E ), node- (V ), and global-level (u) attributes.
    The output graph has the same structure but the updated attributes.
    
    Notations: (h is the character for attribute(vector, feature).)
        h_i, h_j: i-th and j-th node attributes, respectively.
        h_ij: Edge attributes connecting i and j node. (If Directed, h_ij is i->j edge.)
        u: Global attributes
        AGG(...): Aggregated attributes. (It is usually aggregated edge or node attributes.)
    
    Args:
        edge_model_block: f_e(h_ij, h_i, h_j, u), per-edge computations. Use nn.Module()
        node_model_block: f_v(h_i, AGG(h_ij), AGG(h_ji), u), per-node computations. Use nn.Module()
        global_model_block: f_g(AGG(all nodes), AGG(all edges), u), global attribute computations. Use nn.Module()
        what else??:
        
    """
    
    def __init__(self,
                 edge_model_block,
                 node_model_block,
                 global_model_block,
                 use_edge_block=True,
                 use_node_block=True,
                 use_global_block=True,
                 update_graph=False):
        
        super(GNConv, self).__init__()
        
        # f_e, f_v, f_g
        self.edge_model_block = edge_model_block
        self.node_model_block = node_model_block
        self.global_model_block = global_model_block
        self._use_edge_block = use_edge_block
        self._use_node_block = use_node_block
        self._use_global_block = use_global_block
        self._update_graph = update_graph
            
        
        # initialization
#         self.reset_parameters()
        
    def reset_parameters(self):
        for m in self.edge_model_block.modules():
            if hasattr(m, 'reset_parameters'):
                m.reset_parameters()
        for m in self.node_model_block.modules():
            if hasattr(m, 'reset_parameters'):
                m.reset_parameters()
        for m in self.global_model_block.modules():
            if hasattr(m, 'reset_parameters'):
                m.reset_parameters()

        
    def forward(self, graph):
        """This is a high-level module.
        Read graph and
        1. update edge-level
        2. update node-level
        3. update global-level
        and return the updated graph
        
        Args:
            graph: torch_geometric.data.data.Data
                It has [x, edge_index, edge_attr, global_attr] as keys.
        """
            
        if self._use_edge_block:
            # Edge-level update
            graph = self.edge_model_block(graph)    # (N^e, d_e+d_v+d_v+d_g) -> (N^e, out_features)
        
        if self._use_node_block:
            # Node-level update
            graph = self.node_model_block(graph)    # (N^v, d_v+d_e+d_e+d_g) -> (N^v, out_features)
        
        if self._use_global_block:
            # Global-level update
            graph = self.global_model_block(graph)

        return graph



class PhysicsInformedGNConv(nn.Module):
    """Physics Informed GNConv model.
    Set in_features/out_features of each block for each task.
    e.g. Even if there is node_attr only (no edge_attr) (GCN task),
    still edge_attr is required to propagate information
    from neighboring nodes to the reference node.
    GN doesn't have a single path from a node to another node.
    Instead, all information flow as follow;
        
        node->edge->node->edge

    (If global_attr is used, it might be different though.)
    """
    #       Hidden(t) <-- Hidden(t+1)
    #           |             ^
    #           |  *--------* |
    #           *->|        | |
    # Input(t) --->| GNConv |-*-> Output(t+1)
    #              |        |
    #              *--------*
    # 
    # Physics rule: (e.g.,) Output(t+1) - Input(t) = coeff*[∇^2 Input(t)]
    
    def __init__(self,
                 edge_block_model,
                 node_block_model,
                 global_block_model,
                 use_edge_block=True,
                 use_node_block=True,
                 use_global_block=False):
        
        super(PhysicsInformedGNConv, self).__init__()
        
        # random coefficients
        self.a = (5*random.random()-2.5)
        self.b = (5*random.random()-2.5)

        # Core - GNConv Module
        #### Be careful to set in_features in each block.
        self.eb_module = edge_block_model
        self.nb_module = node_block_model
        self.gb_module = global_block_model

        self._gnc_module = GNConv(self.eb_module, 
                                  self.nb_module, 
                                  self.gb_module,
                                  use_edge_block=use_edge_block,
                                  use_node_block=use_node_block,
                                  use_global_block=use_global_block)
        
    def forward(self, input_graphs, laplacian, h_init, coeff=0.1, pde='diff', skip=False):
        """
        return future states with time_derivatives and spatial_derivatives
        coeff: PDE coefficient
        pde: 'diff', 'wave'
        """
        num_processing_steps = len(input_graphs)
        
        output_tensors = []
        time_derivatives = []
        spatial_derivatives = []
        
        h_prev = None
        h_curr = h_init

        for input_graph in input_graphs:
#             if skip:
#                 #### GN-skip
#                 h_curr_concat = Data(x=input_graph.x+h_curr.x,
#                                      edge_index=input_graph.edge_index,
#                                      edge_attr=input_graph.edge_attr+h_curr.edge_attr)
#                 h_curr_concat.global_attr = h_curr.global_attr
            
#             else:
            #### GN
            h_curr_concat = graph_concat(input_graph, h_curr, node_cat=True, edge_cat=True, global_cat=False)

            
            h_next = self._gnc_module(h_curr_concat)    # h_curr is NOT updated.
            
            if skip:
                _global_attr = h_next.global_attr
                h_next = Data(x=h_next.x+h_curr.x,
                              edge_index=input_graph.edge_index,
                              edge_attr=h_next.edge_attr+h_curr.edge_attr)
                h_next.global_attr = _global_attr

            if self.training:
                # time 1st derivative = H(t+1) - H(t)    Diffusion
                # time 2nd derivative = H(t+1) - 2H(t) + H(t-1)    Wave
                if h_prev and pde=="wave":
                    time_derivatives.append(h_next.x - 2*h_curr.x + h_prev.x)
                elif pde=="diff":
                    time_derivatives.append(h_next.x - h_curr.x)
                elif h_prev and pde=="random":
#                     time_derivatives.append(h_next.x + (5*random.random()-10)*h_curr.x)
                    time_derivatives.append(h_next.x + self.a*h_curr.x + self.b*h_prev.x)
                elif h_prev and pde=="both":
                    time_derivatives.append(h_next.x - 2*h_curr.x + h_prev.x + h_next.x - h_curr.x)
                else:
                    time_derivatives.append(h_next.x - h_curr.x)
                # spatial derivative = coeff*[∇^2 H(t)]    (∇^2=-Laplacian)
                spatial_derivatives.append(-coeff*laplacian.mm(h_curr.x))

            h_prev = h_curr    # H(t) -> H(t-1)
            h_curr = h_next    # H(t+1) -> H(t)

            output_tensors.append(copy_geometric_data(h_curr))

        return output_tensors, time_derivatives, spatial_derivatives