"""
Utilities for data preprocessing.
"""

import pandas as pd
import geopandas
from sklearn.neighbors import kneighbors_graph

import torch
import numpy as np
import networkx as nx


def pos2xycoord(pos2node):
    ret = []
    for key, val in sorted(pos2node.items(), key=lambda kv: kv[1]):
        ret.append([key[0]*1.0, key[1]*1.0])
        
    return np.array(ret)

def node2xycoord(node2feature):
    ret = []
    for key, val in sorted(node2feature.items(), key=lambda kv: kv[0]):
        ret.append([val['XLAT'], val['XLONG']])
        
    return np.array(ret)

def random_sampling_from_image(image, yrange, xrange, prob=0.1, seed=42):
    np.random.seed(seed)
    
    sampling_kernel = np.zeros(image.shape)
    
    for i in range(yrange[0], yrange[1]):
        for j in range(xrange[0], xrange[1]):
            if np.random.binomial(n=1, p=prob):
                sampling_kernel[i,j] = 1.0
            else:
                pass
        
    return np.multiply(image, sampling_kernel)

def sampling_from_image(image, filter_size):
    """
    Sample pixels from image using Conv2d operation.
    Args:
        image: 2D image.
        filter_size: int
    Output:
        sampled image. size is unchanged.
    """
    sampling_kernel = np.zeros(image.shape)
    
    for i in range(sampling_kernel.shape[0]):
        for j in range(sampling_kernel.shape[1]):
            if (i%filter_size) == (filter_size//2) and (j%filter_size) == (filter_size//2):
                sampling_kernel[i,j] = 1.0
            else:
                pass
        
    return np.multiply(image, sampling_kernel)


def collect_features(image, features, variables, vertical_flip=True, urban_only=False):
    """
    Collecting features at each pixel.
    Args:
        image: LU_INDEX map.
        features: collected features.
        vertical_flip: True if image is flipped vertically.
        urban_only: True if (LU_INDEX=31,32,33) is only considered.
    Output:
        node2feature: dictionary to retrieve features(or time series) at each node
        pos2node: dictionary to retrieve node index for each (i,j) position
    """
    node2feature = {}
    pos2node = {}
    node_ind = 0
    pq0, a2, a3, a4 = 379.90516, 17.2693882, 273.16, 35.86    # for Relative Humidity calculation
    
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):

            tmp_index = image[i,j]
            if tmp_index == 0:
                # 0 index is dummy index.
                continue
                
            if urban_only:
                if tmp_index==31 or tmp_index==32 or tmp_index==33:
                    pass
                else:
                    continue

            tmp_dict = {'LU_INDEX': tmp_index.item(), 'ij_loc': (i,j)}

            for f in features:
                if f in {'T2','ALBEDO','RAINNC','PBLH','Q2','PSFC'}:    # 3D time varying
                    if vertical_flip:
                        tmp_dict[f] = np.array(variables[f][:,-i-1,j])
                    else:
                        tmp_dict[f] = np.array(variables[f][:,i,j])
                elif f in {'U','V','SMOIS'}:        # 4D time varying
                    if vertical_flip:
                        tmp_dict[f] = np.array(variables[f][:,0,-i-1,j])
                    else:
                        tmp_dict[f] = np.array(variables[f][:,0,i,j])
                elif f in {'XLAT','XLONG','FRC_URB2D','VEGFRA'}:    # 3D time invariant
                    if vertical_flip:
                        tmp_dict[f] = variables[f][0,-i-1,j].item()    #
                    else:
                        tmp_dict[f] = variables[f][0,i,j].item()    #
                else:
                    pass    # RH2 will be calculated later

            # manually calculate relative humidity
            q2 = tmp_dict['Q2']
            t2 = tmp_dict['T2']
            psfc = tmp_dict['PSFC']
            tmp_dict['RH2'] = q2 / ( (pq0 / psfc) * np.exp(a2*(t2 - a3) / (t2 - a4)) )

            node2feature[node_ind] = tmp_dict
            pos2node[(i,j)] = node_ind
            node_ind += 1

    return node2feature, pos2node


def build_regular_adj(sampled_LU_INDEX, distance, pos2node):
    """
    Build a regular adjacency matrix from sampled_LU_INDEX
    Args:
        sampled_LU_INDEX: This image has some sampled non-zero values.
            The non-zero values are considered as nodes.
        distance: distance between two adjacent pixels(nodes).
        pos2node: 
    Output:
        adjacency matrix
    """
    num_nodes = np.nonzero(sampled_LU_INDEX)[0].shape[0]
    A = np.zeros((num_nodes, num_nodes))
    adj_dict = {}
    for i in range(sampled_LU_INDEX.shape[0]):
        for j in range(sampled_LU_INDEX.shape[1]):
            
            if (i,j) in pos2node:
                node_ind = pos2node[(i,j)]
                neighbors = []
                for pos in [(i-distance,j),(i+distance,j),(i,j+distance),(i,j-distance)]:
                    if pos in pos2node:
                        neighbors.append(pos2node[pos])
                        A[node_ind, pos2node[pos]] = 1.0
                        A[pos2node[pos], node_ind] = 1.0
                
                adj_dict[node_ind] = neighbors
                
    return A, adj_dict


def build_feature_matrix(node2feature, length, features):
    """
    Build a feature matrix for all nodes.
    Args:
        node2feature: node2feature: dictionary to retrieve features(or time series) at each node.
        length: time series length.
        features: a list of features considered.
    Output:
        X: [T,N,D] matrix.
    """
    num_nodes = len(node2feature)
    num_features = len(features)
    X = np.zeros((length, num_nodes, num_features))
    
    for key, val in node2feature.items():
        for i, f in enumerate(features):
            X[:,key,i] = val[f]
            
    return X

def add_neighbors(node2feature, adj_dict):
    """
    Add neighbors to node2feature.
    """
    for key, val in adj_dict.items():
        node2feature[key]['neighbors'] = val
        
    return node2feature

def build_edge_attr(node2feature, edge_index):
    """
    Build edge_attr from LU_INDEX of connected nodes of each edge.
    """
    r,c = edge_index
    edge_attr = []
    features = {}
    num_features = 0
    for sent_node, received_node in zip(r,c):
        
        sent_LU_INDEX = int(node2feature[sent_node.item()]['LU_INDEX'])
        received_LU_INDEX = int(node2feature[received_node.item()]['LU_INDEX'])
        
        feature = (sent_LU_INDEX, received_LU_INDEX)
        if feature in features:
            edge_attr.append(features[feature])
        else:
            features[feature] = num_features
            edge_attr.append(features[feature])
            num_features += 1
        
    return torch.tensor(edge_attr), features


def get_xy_from_ind(xy_array, ind):
    return xy_array[ind]

def get_graph(xy_array, num_neighbors=4, one_component=True):
    
    adj_dict = {}
    A = kneighbors_graph(xy_array, num_neighbors, mode='connectivity')    # anti-symmetry
    A = (A + A.transpose()) / 2    # symmetry
    A = (A>0)*1.0

    G = nx.from_numpy_matrix(A.todense())
    if one_component:
        assert nx.number_connected_components(G)==1
    for n in range(G.number_of_nodes()):
        adj_dict[n] = [nn for nn in G.neighbors(n)]

    edge_index = np.array([indices for indices in np.nonzero(A)])
    num_nodes = A.shape[0]
    num_edges = edge_index.shape[1]

    start_pts = get_xy_from_ind(xy_array, edge_index[0])
    end_pts = get_xy_from_ind(xy_array, edge_index[1])

    return A.todense(), adj_dict, start_pts, end_pts

def get_numpy_from_graph(start_pts, end_pts, CRS_code):
    from shapely.geometry import Point
    
    start_pts = pd.DataFrame(start_pts, columns=['LAT', 'LONG'])
    geometry = [Point(xy) for xy in zip(start_pts['LONG'], start_pts['LAT'])]
    start_pts = geopandas.GeoDataFrame(start_pts, geometry=geometry)
    start_pts.crs = {'init': 'epsg:4326'}
    start_pts = start_pts.to_crs({'init': CRS_code})
    start_pts = np.array(start_pts['geometry'].apply(lambda x: [x.x, x.y]).values.tolist())

    end_pts = pd.DataFrame(end_pts, columns=['LAT', 'LONG'])
    geometry = [Point(xy) for xy in zip(end_pts['LONG'], end_pts['LAT'])]
    end_pts = geopandas.GeoDataFrame(end_pts, geometry=geometry)
    end_pts.crs = {'init': 'epsg:4326'}
    end_pts = end_pts.to_crs({'init': CRS_code})
    end_pts = np.array(end_pts['geometry'].apply(lambda x: [x.x, x.y]).values.tolist())
    
    return start_pts, end_pts