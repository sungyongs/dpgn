"""
Differentiable Physics-informed Graph Networks (DPGN)
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch_scatter import scatter_add, scatter_max, scatter_mean, scatter_min, scatter_mul
from torch_geometric.data import Data

from blocks import EdgeBlock, NodeBlock, GlobalBlock
from modules import PhysicsInformedGNConv


class Net(nn.Module):

    def __init__(self,
                 node_attr_size,
                 edge_num_embeddings,
                 out_size,
                 edge_hidden_size=64,
                 node_hidden_size=64,
                 global_hidden_size=64,
                 skip=False,
                 device='cpu'):
        super(Net, self).__init__()
        
        self.input_size = node_attr_size
        self.edge_h_dim = edge_hidden_size
        self.edge_half_h_dim = int(self.edge_h_dim)/2
        self.node_h_dim = node_hidden_size
        self.node_half_h_dim = int(self.node_h_dim)/2
        self.global_h_dim = global_hidden_size
        self.global_half_h_dim = int(self.global_h_dim)/2
        self.skip = skip
        self.device = device
        
        #### Encoder
        self.edge_embedding = nn.Sequential(nn.Embedding(edge_num_embeddings, self.edge_h_dim))
        
#         self.eb_enc_custom_func = nn.Sequential(nn.Linear(node_attr_size*2, self.edge_h_dim),
#                                                 nn.ReLU())
#         self.edge_embedding = EdgeBlock(node_attr_size*2, 
#                                         self.edge_h_dim,
#                                         use_edges=False,
#                                         use_sender_nodes=True,
#                                         use_receiver_nodes=True,
#                                         use_globals=False,
#                                         custom_func=self.eb_enc_custom_func)

        self.node_enc = nn.Sequential(nn.Linear(self.input_size, self.node_h_dim),
                                      nn.ReLU(),
                                     )
        
        #### GN only model
#         if self.skip:
#             self.eb_custom_func = nn.Sequential(nn.Linear((self.edge_h_dim+self.node_h_dim*2)+self.global_h_dim,
#                                                            self.edge_h_dim),
#                                                 nn.ReLU(),
#                                                )
#             self.nb_custom_func = nn.Sequential(nn.Linear(self.node_h_dim+self.edge_h_dim*2+self.global_h_dim,
#                                                           self.node_h_dim),
#                                                 nn.ReLU(),
#                                                )
#             self.gb_custom_func = nn.Sequential(nn.Linear(self.node_h_dim+self.edge_h_dim+self.global_h_dim,
#                                                           self.global_h_dim),
#                                                 nn.ReLU(),
#                                                )
#             self.eb_module = EdgeBlock((self.edge_h_dim+self.node_h_dim*2)+self.global_h_dim,
#                                        self.edge_h_dim,
#                                        use_edges=True, 
#                                        use_sender_nodes=True, 
#                                        use_receiver_nodes=True, 
#                                        use_globals=True,
#                                        custom_func=self.eb_custom_func)

#             self.nb_module = NodeBlock(self.node_h_dim+self.edge_h_dim*2+self.global_h_dim,
#                                        self.node_h_dim,
#                                        use_nodes=True, 
#                                        use_sent_edges=True, 
#                                        use_received_edges=True, 
#                                        use_globals=True,
#                                        sent_edges_reducer=scatter_add, 
#                                        received_edges_reducer=scatter_add,
#                                        custom_func=self.nb_custom_func)

#             self.gb_module = GlobalBlock(self.node_h_dim+self.edge_h_dim+self.global_h_dim,
#                                          self.global_h_dim,
#                                          edge_reducer=scatter_mean, 
#                                          node_reducer=scatter_mean,
#                                          custom_func=self.gb_custom_func,
#                                          device=device)
            
#         else:
        # Check the dimension. Since the latent representations are concatenated, it is doubled.
        self.eb_custom_func = nn.Sequential(nn.Linear((self.edge_h_dim+self.node_h_dim*2)*2+self.global_h_dim,
                                                       self.edge_h_dim),
                                            nn.ReLU(),
                                           )
        self.nb_custom_func = nn.Sequential(nn.Linear(self.node_h_dim*2+self.edge_h_dim*2+self.global_h_dim,
                                                      self.node_h_dim),
                                            nn.ReLU(),
                                           )
        self.gb_custom_func = nn.Sequential(nn.Linear(self.node_h_dim+self.edge_h_dim+self.global_h_dim,
                                                      self.global_h_dim),
                                            nn.ReLU(),
                                           )

        self.eb_module = EdgeBlock((self.edge_h_dim+self.node_h_dim*2)*2+self.global_h_dim,
                                   self.edge_h_dim,
                                   use_edges=True, 
                                   use_sender_nodes=True, 
                                   use_receiver_nodes=True, 
                                   use_globals=True,
                                   custom_func=self.eb_custom_func)

        self.nb_module = NodeBlock(self.node_h_dim*2+self.edge_h_dim*2+self.global_h_dim,
                                   self.node_h_dim,
                                   use_nodes=True, 
                                   use_sent_edges=True, 
                                   use_received_edges=True, 
                                   use_globals=True,
                                   sent_edges_reducer=scatter_add, 
                                   received_edges_reducer=scatter_add,
                                   custom_func=self.nb_custom_func)

        self.gb_module = GlobalBlock(self.node_h_dim+self.edge_h_dim+self.global_h_dim,
                                     self.global_h_dim,
                                     edge_reducer=scatter_mean, 
                                     node_reducer=scatter_mean,
                                     custom_func=self.gb_custom_func,
                                     device=device)
        
        self.gn = PhysicsInformedGNConv(self.eb_module, 
                                        self.nb_module, 
                                        self.gb_module, 
                                        use_edge_block=True, 
                                        use_node_block=True, 
                                        use_global_block=True)
        
        #### Decoder
        self.node_dec = nn.Sequential(nn.Linear(self.node_h_dim, self.node_h_dim),
                                      nn.ReLU(),
                                      nn.Linear(self.node_h_dim, out_size)
                                     )
        self.node_dec_for_input = nn.Sequential(nn.Linear(self.node_h_dim, self.node_h_dim),
                                                nn.ReLU(),
                                                nn.Linear(self.node_h_dim, self.input_size))    # to predict input.
        
        
    def forward(self, data, sp_L, num_processing_steps, coeff, pde='diff'):
        from utils import decompose_graph
        
        input_graphs = []
        node_attrs = []
        edge_indexs = []
        edge_attrs = []

        for step_t in range(num_processing_steps):
            node_attr, edge_index, edge_attr, global_attr = decompose_graph(data[step_t])
        
            #### Encoder
            encoded_edge = self.edge_embedding(edge_attr)    # Use embedding
            encoded_node = self.node_enc(node_attr)

            #### GN
            input_graph = Data(x=encoded_node, edge_index=edge_index, edge_attr=encoded_edge)
            if step_t == 0:
                input_graph.global_attr = global_attr
            
            input_graphs.append(input_graph)

        init_graph = input_graphs[0]
        # h_init is zero tensor
        h_init = Data(x=torch.zeros(init_graph.x.size(), dtype=torch.float32, device=self.device), 
                      edge_index=init_graph.edge_index, 
                      edge_attr=torch.zeros(init_graph.edge_attr.size(), dtype=torch.float32, device=self.device))
        h_init.global_attr = init_graph.global_attr

        output_graphs, time_derivatives, spatial_derivatives = self.gn(input_graphs, sp_L, h_init, coeff, pde, self.skip)
        
        #### Decoder
        output_nodes, pred_inputs = [], []
        for output_graph in output_graphs:
            output_nodes.append(self.node_dec(output_graph.x))
            pred_inputs.append(self.node_dec_for_input(output_graph.x))
            
#         output_nodes = [self.node_dec(output_graph.x) for output_graph in output_graphs]
#         pred_inputs = [self.node_dec_for_input(output_graph.x) for output_graph in output_graphs]
        
        return output_nodes, time_derivatives, spatial_derivatives, pred_inputs

