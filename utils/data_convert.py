import numpy as np
import os
import os.path as osp
import pdb
import networkx as nx
from utils.common_tools import mkdirs
import tqdm
import random

def z_score(data):
    return (data - np.mean(data)) / np.std(data)

def generate_dataset(data, idx, x_len=12, y_len=12):
    res = data[idx]
    node_size = data.shape[1]
    t = len(idx)-1
    idic = 0
    x_index, y_index = [], []
    
    for i in tqdm.tqdm(range(t,0,-1)):
        if i-x_len-y_len>=0:
            x_index.extend(list(range(i-x_len-y_len, i-y_len)))
            y_index.extend(list(range(i-y_len, i)))

    x_index = np.asarray(x_index)
    y_index = np.asarray(y_index)
    x = res[x_index].reshape((-1, x_len, node_size))
    y = res[y_index].reshape((-1, y_len, node_size))
 
    return x, y

def generate_samples(days, savepath, data, graph, train_rate=0.6, val_rate=0.2, test_rate=0.2, val_test_mix=False):
    edge_index = np.array(list(graph.edges)).T
    del graph
    data = data[0:days*288, :]
    t, n = data.shape[0], data.shape[1]
    
    train_idx = [i for i in range(int(t*train_rate))]
    val_idx = [i for i in range(int(t*train_rate), int(t*(train_rate+val_rate)))]
    test_idx = [i for i in range(int(t*(train_rate+val_rate)), t)]
    
    train_x, train_y = generate_dataset(data, train_idx)
    val_x, val_y = generate_dataset(data, val_idx)
    test_x, test_y = generate_dataset(data, test_idx)
    if val_test_mix:
        val_test_x = np.concatenate((val_x, test_x), 0)
        val_test_y = np.concatenate((val_y, test_y), 0)
        val_test_idx = np.arange(val_x.shape[0]+test_x.shape[0])
        np.random.shuffle(val_test_idx)
        val_x, val_y = val_test_x[val_test_idx[:int(t*val_rate)]], val_test_y[val_test_idx[:int(t*val_rate)]]
        test_x, test_y = val_test_x[val_test_idx[int(t*val_rate):]], val_test_y[val_test_idx[int(t*val_rate):]]

    train_x = z_score(train_x)
    val_x = z_score(val_x)
    test_x = z_score(test_x)
    np.savez(savepath, train_x=train_x, train_y=train_y, val_x=val_x, val_y=val_y, test_x=test_x, test_y=test_y, edge_index=edge_index)
    data = {"train_x":train_x, "train_y":train_y, "val_x":val_x, "val_y":val_y, "test_x":test_x, "test_y":test_y, "edge_index":edge_index}
    return data


# generate_samples('../data/tmp', data)