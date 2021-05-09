import numpy as np
import torch 
from torch_geometric.data import Data, Dataset

class TrafficDataset(Dataset):
    def __init__(self, inputs, split, x='', y='', edge_index='', mode='default'):
        if mode == 'default':
            self.x = inputs[split+'_x'] # [T, Len, N]
            self.y = inputs[split+'_y'] # [T, Len, N]
            # self.edge_index = inputs['edge_index'] # [2, N]
        else:
            self.x = x
            self.y = y
            # self.edge_index = edge_index.numpy()
        # print(self.x.shape, self.y.shape, self.edge_index.shape)
    
    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, index):
        x = torch.Tensor(self.x[index].T)
        y = torch.Tensor(self.y[index].T)
        # edge_index = torch.Tensor(self.edge_index).to(torch.long)
        return Data(x=x, y=y)#, edge_index=edge_index)   
    
class continue_learning_Dataset(Dataset):
    def __init__(self, inputs):#, graph):
        self.x = inputs # [T, Len, N]
        # self.edge_index = graph # [N, N]
        # print(self.x.shape, self.y.shape, self.edge_index.shape)
    
    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, index):
        x = torch.Tensor(self.x[index].T)
        # edge_index = torch.Tensor(self.edge_index).to(torch.long)
        return Data(x=x)#, edge_index=edge_index)