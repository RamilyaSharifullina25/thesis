import torch
import torch.nn as nn
from torch.utils.data import Dataset

class Main_dataset(Dataset):
    def __init__(self, data, labels, feature_scales={}, N=25000, length_lim=90, sparse_rate=1):
        
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
    

class Verification_dataset(Dataset):
    def __init__(self, needed, feature_scales_simil={}, feature_scales_diff={}):
        
        self.needed = needed
        self.feature_scales_simil = feature_scales_simil
        self.feature_scales_diff = feature_scales_diff
        
    def __getitem__(self, index):

        time1 = self.needed[index][0]
        time2 = self.needed[index][1]
        relation = self.needed[index][2]
        
        self.time1 = torch.tensor(time1).to(torch.float32)
        self.time2 = torch.tensor(time2).to(torch.float32)
        self.relation = torch.tensor(relation).to(torch.float32)
        
        # time1
        for f in range(self.time1.shape[1]):
            self.time1[:, f] = self.time1[:, f] - self.time1[:, f].mean()
            
        for f in range(self.time1.shape[1]):
            if f not in self.feature_scales_simil:
                std = self.time1[:, f].std()
                self.feature_scales_simil[f] = std
            self.time1[:, f] = self.time1[:, f] / self.feature_scales_simil[f]
            
        # time2
        for f in range(self.time2.shape[1]):
            self.time2[:, f] = self.time2[:, f] - self.time2[:, f].mean()
            
        for f in range(self.time2.shape[1]):
            if f not in self.feature_scales_diff:
                std = self.time2[:, f].std()
                self.feature_scales_diff[f] = std
            self.time2[:, f] = self.time2[:, f] / self.feature_scales_diff[f]
            
        self.time1 = self.time1.transpose(0, 1)
        self.time2 = self.time2.transpose(0, 1)

        return self.time1, self.time2, self.relation

    def __len__(self):
        return len(self.needed)
    