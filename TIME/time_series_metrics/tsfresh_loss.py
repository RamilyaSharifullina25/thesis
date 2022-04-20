import torch
import torch.nn as nn

class Mean_similarity_loss(nn.Module):
    def __init__(self):
        super(Mean_similarity_loss, self).__init__()
        
    def forward(self, x, y):
        return (1 - abs(x - y) / (abs(x) + abs(y))).mean()

class Root_mean_square_similarity_loss(nn.Module):
    def __init__(self):
        super(Root_mean_square_similarity_loss, self).__init__()
        
    def forward(self, x, y):
        return ((1 - abs(x - y) / (abs(x) + abs(y))) ** 2).mean() ** 0.5

class Coisine_between_angles_loss(nn.Module):
    def __init__(self, x, y):
        super(Coisine_between_angles_loss, self).__init__()
        
    def forward(self, x, y):
        return (x * y).sum() / ((x ** 2).sum() ** 0.5 * (y ** 2).sum() ** 0.5)

class Pearson_corr_funct_loss(nn.Module):
    def __init__(self, x, y):
        super(Pearson_corr_funct_loss, self).__init__()
    
    def forward(self, x, y):
        return ((x - x.mean()) * (y - y.mean())).sum() / (((x - x.mean()) ** 2.0).sum() ** 0.5 * ((y - y.mean()) ** 2.0).sum() ** 0.5)

class Eucledian_distance_loss(nn.Module):
    def __init__(self, x, y):
        super(Eucledian_distance_loss, self).__init__()
        
    def forward(self, x, y):
        return (((x - y) ** 2).sum()) ** 0.5

# class Minkowski_distance_loss(x, y, p):
#     def __init__(self, x, y):
#         super(Minkowski_distance_loss, self).__init__()
        
#     def forward(x, y, p):
#         return (((x - y) ** p).sum()) ** (1 / p)