import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

class DataSet():
    def __init__(self, data, labels, feature_scales={}, N=25000, length_lim=90, sparse_rate=1):

#         if len(data) > N:
#             sub_inds = np.random.choice(len(data), N, replace=False)
#         else:
#             sub_inds = np.arange(len(data))
        
        self.data = data
        self.data = torch.tensor(self.data).to(torch.float32)
        self.labels = torch.tensor(labels)
        self.sparse_rate = sparse_rate
        self.feature_scales = feature_scales
        
        for f in range(data.shape[2]):
            self.data[:, :, f] = (self.data[:, :, f].T - self.data[:, :, f].mean(axis=1)).T 
            
        for f in range(data.shape[2]):
            if f not in feature_scales:
                std = self.data[:, :, f].ravel().std()
                feature_scales[f] = std
            self.data[:, :, f] = self.data[:, :, f]/feature_scales[f]
            
        self.data = self.data.transpose(1, 2)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        return self.data[i], self.labels[i]