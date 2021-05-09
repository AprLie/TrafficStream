import sys
sys.path.append('src/')
import numpy as np
from scipy.stats import entropy as kldiv
from datetime import datetime
from torch_geometric.utils import to_dense_batch 
from src.trafficDataset import continue_learning_Dataset
from torch_geometric.data import Data, Batch, DataLoader
import torch
from scipy.spatial import distance
import os.path as osp
# scipy.stats.entropy(x, y) 


def get_feature(data, graph, args, model, adj):
    node_size = data.shape[1]
    data = np.reshape(data[-288*7-1:-1,:], (-1, args.x_len, node_size))
    dataloader = DataLoader(continue_learning_Dataset(data), batch_size=data.shape[0], shuffle=False, pin_memory=True, num_workers=3)
    # feature shape [T', feature_dim, N]
    for data in dataloader:
        data = data.to(args.device, non_blocking=True)
        feature, _ = to_dense_batch(model.feature(data, adj), batch=data.batch)
        node_size = feature.size()[1]
        # print("before permute:", feature.size())
        feature = feature.permute(1,0,2)

        # [N, T', feature_dim]
        return feature.cpu().detach().numpy()


def get_adj(year, args):
    adj = np.load(osp.join(args.graph_path, str(year)+"_adj.npz"))["x"]
    adj = adj / (np.sum(adj, 1, keepdims=True) + 1e-6)
    return torch.from_numpy(adj).to(torch.float).to(args.device)
    

def score_func(pre_data, cur_data, args):
    # shape: [T, N]
    node_size = pre_data.shape[1]
    score = []
    for node in range(node_size):
        max_val = max(max(pre_data[:,node]), max(cur_data[:,node]))
        min_val = min(min(pre_data[:,node]), min(cur_data[:,node]))
        pre_prob, _ = np.histogram(pre_data[:,node], bins=10, range=(min_val, max_val))
        pre_prob = pre_prob *1.0 / sum(pre_prob)
        cur_prob, _ = np.histogram(cur_data[:,node], bins=10, range=(min_val, max_val))
        cur_prob = cur_prob * 1.0 /sum(cur_prob)
        score.append(kldiv(pre_prob, cur_prob))
    # return staiton_id of topk max score, station with larger KL score needs more training
    return np.argpartition(np.asarray(score), -args.topk)[-args.topk:]


def influence_node_selection(model, args, pre_data, cur_data, pre_graph, cur_graph):
    # detect_strategy: "original": hist of original series; "feature": hist of feature at each dimension
    # pre_data/cur_data: data of seven day, shape [T, N] T=288*7
    # assert pre_data.shape[0] == 288*7
    # assert cur_data.shape[0] == 288*7
    if args.detect_strategy == 'original':
        pre_data = pre_data[-288*7-1:-1,:]
        cur_data = cur_data[-288*7-1:-1,:]
        node_size = pre_data.shape[1]
        score = []
        for node in range(node_size):
            max_val = max(max(pre_data[:,node]), max(cur_data[:,node]))
            min_val = min(min(pre_data[:,node]), min(cur_data[:,node]))
            pre_prob, _ = np.histogram(pre_data[:,node], bins=10, range=(min_val, max_val))
            pre_prob = pre_prob *1.0 / sum(pre_prob)
            cur_prob, _ = np.histogram(cur_data[:,node], bins=10, range=(min_val, max_val))
            cur_prob = cur_prob * 1.0 /sum(cur_prob)
            score.append(kldiv(pre_prob, cur_prob))
        # return staiton_id of topk max score, station with larger KL score needs more training
        return np.argpartition(np.asarray(score), -args.topk)[-args.topk:]
    elif args.detect_strategy == 'feature':
        model.eval()
        pre_adj = get_adj(args.year-1, args)
        cur_adj = get_adj(args.year, args)
        
        pre_data = get_feature(pre_data, pre_graph, args, model, pre_adj)
        cur_data = get_feature(cur_data, cur_graph, args, model, cur_adj)
        score = []
        # print(pre_data)
        # print(pre_data.shape)
        # print(cur_data.shape)
        for i in range(pre_data.shape[0]):
            score_ = 0.0
            for j in range(pre_data.shape[2]):
                # if max(pre_data[i,:,j]) - min(pre_data[i,:,j]) == 0 and max(cur_data[i,:,j]) - min(cur_data[i,:,j]) == 0: continue
                pre_data[i,:,j] = (pre_data[i,:,j] - min(pre_data[i,:,j]))/(max(pre_data[i,:,j]) - min(pre_data[i,:,j]))
                cur_data[i,:,j] = (cur_data[i,:,j] - min(cur_data[i,:,j]))/(max(cur_data[i,:,j]) - min(cur_data[i,:,j]))
                
                pre_prob, _ = np.histogram(pre_data[i,:,j], bins=10, range=(0, 1))
                pre_prob = pre_prob *1.0 / sum(pre_prob)
                cur_prob, _ = np.histogram(cur_data[i,:,j], bins=10, range=(0, 1))
                cur_prob = cur_prob * 1.0 /sum(cur_prob)
                score_ += distance.jensenshannon(pre_prob, cur_prob)
            score.append(score_)
        # print(sorted(score))
        return np.argpartition(np.asarray(score), -args.topk)[-args.topk:]
    else: args.logger.info("node selection mode illegal!")

