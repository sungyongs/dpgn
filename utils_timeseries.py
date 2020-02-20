"""
Utilities for time series preprocessing
"""
import numpy as np
import pandas as pd


def get_timeseries_at_node(node_ind, node2feature, ts_code):
    """
    Return ts_code time series at node_ind
    output shape : (T, )
    """
    return node2feature[node_ind][ts_code]

def merge_timeseries(node_indices, node2feature, ts_code):
    """
    Return merged ts_code time series of node_indices
    Input:
        node_indices : a list of N nodes we want to consider.
        node2feature : node to features dictionary
        ts_code : a code of time series
    Output:
        ret : a merged time series (pd.DataFrame). shape=(T,N)
    """
    ret, cols = [], []
    for nid in node_indices:
        cols.append(node2feature[nid]['ij_loc'])
        ret.append(get_timeseries_at_node(nid, node2feature, ts_code))
        
    ret = pd.DataFrame(np.array(ret).transpose())    # (T,N)
    ret.columns = cols
    return ret

def add_timestamp(df, start_time="2012-06-28 21:00:00", timedelta=None):
    """
    Return a dataframe having time stamps as indices.
    Input:
        df : (T,N) dataframe. It is an output of merge_timeseries() function.
        start_time : str, or pd.datetime
        timedelta : a list of time delta. Read 'XTIME' in the dataset.
    Output:
        ret : a dataframe with time stamps as an index column
    """
    df['Time'] = pd.to_datetime(start_time)
    df['Time'] = df['Time'] + pd.to_timedelta(timedelta, unit='m')    # you may need different unit.
    
    return df.set_index('Time')

def get_vars3D(node_indices, node2feature, features, start_time, timedelta):
    """
    Get all 3D time varying different measurements (length T).
    Input:
        node_indices : a list of N nodes we want to consider.
        node2feature : node to features dictionary
        features : a list of F features we want to consider.
        start_time : str, or pd.datetime
        timedelta : a list of time delta. Read 'XTIME' in the dataset.
    Output:
        df3d : multi-index dataframe. (F*T,N) (10*384,2390)
    """
    df3d = []
    for ts_code in features:
        tmp = merge_timeseries(node_indices, node2feature, ts_code)
        tmp = add_timestamp(tmp, start_time, timedelta)
        df3d.append(tmp)
        print(ts_code, "is done")

    df3d = pd.concat(df3d, keys=features, axis=0)    # multi-index dataframe
    return df3d
    # indexing
    # df3d.loc['T2'] -> return 2D dataframe (T,N)
    # df3d.loc['T2'].loc[:, [(0,89)]] -> return T2 time series at node (0,89)     (384, 1)

def get_vars2D(node_indices, node2feature, features, pos2node):
    """
    Get all 2D time invariant different measurements.
    Input:
        node_indices : a list of N nodes we want to consider.
        node2feature : node to features dictionary
        features : a list of F features we want to consider.
        pos2node : position to node index.
    Output:
        df2d : single index dataframe. (F,N) (5,2390)
    """
    df2d = {}
    cols = []
    for nid in node_indices:
        tmp = []
        loc = node2feature[nid]['ij_loc']
        cols.append(loc)
        for ts_code in features:
            tmp.append(node2feature[pos2node[loc]][ts_code])

        df2d[loc] = tmp

    df2d = pd.DataFrame(df2d)
    df2d.columns = cols
    df2d.index = features
    return df2d

def get_scaled_df3d(df3d, feature2stats, type='minmax'):
    """
    Return scaled dataframe.
    Input:
        df3d : multi-index dataframe. (F*T,N) (10*384,2390)
        feature2stats : feature-to-df.describe(). It has (std, mean, min, max, etc.) for each measurement.
        [DEPRECATED] features : a list of features we want to consider.
        type : 'minmax', 'std'
    Output:
        scaled_df3d : scaled df3d. (F*T,N)
    """
    scaled_df3d = []
    features = df3d.index.levels[0]
    for ts_code in features:
        tmp = df3d.loc[ts_code]
        if type=='std':
            tmp = (tmp - feature2stats[ts_code].loc['mean',:]) / feature2stats[ts_code].loc['std',:]    # standard normalization
        elif type=='minmax':
            tmp = (tmp - feature2stats[ts_code].loc['min',:]) / (feature2stats[ts_code].loc['max',:] - feature2stats[ts_code].loc['min',:])    # minmax normalization
        else:
            raise Exception("type value should be std or minmax. type={}".format(type))
            
        scaled_df3d.append(tmp)
    
    scaled_df3d = pd.concat(scaled_df3d, keys=features, axis=0)    # multi-index dataframe
    return scaled_df3d

def split_df3d(df3d, ratios=[0.7, 0.0, 0.3]):
    """
    Return split dataset.
    Input:
        df3d : (F*T,N) dataframe
        ratios : a list of training/validation/test sets.
    Output:
        [df3d_tr, df3d_val, df3d_te]    [(F*T*TR_RATIO,N) (F*T*VAL_RATIO,N), (F*T*TE_RATIO,N)]
    """
    features = df3d.index.levels[0]
    T = df3d.loc[features[0]].shape[0]
    N_TR, N_VAL, N_TE = int(T*ratios[0]), int(T*ratios[1]), T-int(T*ratios[0])-int(T*ratios[1])
    split_tr, split_val, split_te = [], [], []
    for ts_code in features:
        tmp = df3d.loc[ts_code]
        _tr, _val, _te = tmp.iloc[:N_TR], tmp.iloc[N_TR:N_TR+N_VAL], tmp.iloc[N_TR+N_VAL:N_TR+N_VAL+N_TE]
        
        split_tr.append(_tr)
        split_val.append(_val)
        split_te.append(_te)
    
    split_df3d_tr = pd.concat(split_tr, keys=features, axis=0)
    split_df3d_val = pd.concat(split_val, keys=features, axis=0)
    split_df3d_te = pd.concat(split_te, keys=features, axis=0)

    return (split_df3d_tr, split_df3d_val, split_df3d_te)

def get_3d_array(df3d):
    """
    Return numpy 3D array
    Input:
        df3d : (F*T,N) dataframe
    Output:
        ret : (N,F,T) numpy array
    """
    features = df3d.index.levels[0]
    ret = []
    for ts_code in features:
        tmp = df3d.loc[ts_code].values.transpose()    # 2D ndarray (N,T)
        ret.append(tmp)
    
    ret = np.array(ret)    # 3D ndarray (F,N,T)
    ret = np.swapaxes(ret, 0, 1)    # 3D ndarray (N,F,T)
    return ret

"""
Preprocess for a seq2seq module input/output
Description:
    Given array_tr/array_te (N,D,T), there are N different nodes (objects or entities).
    Each node has "a" D dimensional length T time-series. (D,T)
    To train a seq2seq-like model (LSTM, GRU, RNN), we need to segment length T to multiple chunks.
    (D,T) -> Input (M,D,Tin) and Output (M,Dout,Tout)    # Note that the target dimension might be different with the input dim. (Dout < D)
    M = T - (Tin + Tout) + 1    # M : the number of chunks, input/output pairs
    Thus,
    Input:
        array : (N,D,T) 3D array
    Output:
        array_input : (N,M,D,Tin) 4D array
        array_output : (N,M,Dout,Tout) 4D array
"""

def get_chunk_IO(array3d, t_in, d_out):
    """
    Return input and output chunks.
    Input:
        array3d : (N,D,T) 3D array. e.g., array_tr, array_te
        t_in : the length of input seq
        [DEPRECATED] t_out : the length of output seq. t_out = t_in
        d_out : indicies of output features. e.g., [0,1] for T2 and ALBEDO
    Output:
        chunk_in : (N,M,D,Tin) 4D array
        chunk_out : (N,M,Dout,Tout) 4D array
    """
#     dT = t_in + t_out
    d_all = np.arange(array3d.shape[1])
    d_in = np.setdiff1d(d_all, np.array(d_out))
    dT = t_in + 1
    T = array3d.shape[2]
    M = T - dT + 1
    chunk = []
    for mind in range(M):
        chunk.append(array3d[:,:,range(mind, mind+dT)])
    chunk = np.array(chunk)
    chunk = np.swapaxes(chunk,0,1)
    print("chunk is created. shape (N,M,D,T_IN+T_OUT)={}".format(chunk.shape))
#     print("chunk is split into input chunks and output chunks.")
    chunk_in = chunk[:,:,d_in,:t_in] # chunk[:,:,:,:t_in]
    chunk_out = chunk[:,:,d_out,1:]    # onestep prediction
#     chunk_out = chunk[:,:,d_ind,t_in:]
    print("chunk_in shape (N,M,D,T_IN)={}".format(chunk_in.shape))
    print("chunk_out shape (N,M,D_OUT,T_OUT)={}".format(chunk_out.shape))    # Only consider T2
    
    return (chunk, chunk_in, chunk_out)