import torch
import torch.nn
import torch.nn.functional as F
from torch.autograd import Variable
import math

batch_size = 64
device = torch.device('cuda:1')

def gaussian(window_size, sigma):
    gauss = torch.Tensor([math.exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])
    return gauss/gauss.sum()

def create_window(window_size, channel=5):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(0).unsqueeze(0)
    window = Variable(_1D_window.expand(batch_size, channel,window_size).contiguous())
    return window

def ssim_1d_conv(time_1, time_2, window_size=11, window=None, channel = 5, val_range = None, size_average = True, full=False):
    
    time_1 = torch.from_numpy(time_1)
    time_2 = torch.from_numpy(time_2)
    if val_range is None:
        if torch.max(time_1) > 128:
            max_val = 255
        else:
            max_val = 1

        if torch.min(time_1) < -0.5:
            min_val = -1
        else:
            min_val = 0
        L = max_val - min_val
    else:
        L = val_range
        
    padd = 0
    window = create_window(window_size = window_size, channel = channel).to(device)
    
    time_1 = time_1.float()
    time_2 = time_2.float()
    
    mu1 = F.conv1d(time_1.float(), window, padding=padd)
    mu2 = F.conv1d(time_2.float(), window, padding=padd)
    
    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2
    
    sigma1_sq = F.conv1d(time_1 * time_1, window, padding=padd) - mu1_sq
    sigma2_sq = F.conv1d(time_2 * time_2, window, padding=padd) - mu2_sq
    sigma12 = F.conv1d(time_1 * time_2, window, padding=padd) - mu1_mu2
    
    C1 = (0.01 * L) ** 2
    C2 = (0.03 * L) ** 2
    
    v1 = 2.0 * sigma12 + C2
    v2 = sigma1_sq + sigma2_sq + C2
    cs = v1 / v2
    
    ssim_map = ((2 * mu1_mu2 + C1) * v1) / ((mu1_sq + mu2_sq + C1) * v2)

    if size_average:
        cs = cs.mean()
        ret = ssim_map.mean()
    else:
        cs = cs.mean(1).mean(1).mean(1)
        ret = ssim_map.mean(1).mean(1).mean(1)

    if full:
        return ret, cs
    return ret

class SSIM_1d_conv(torch.nn.Module):
    def __init__(self, window_size=11, size_average=True, val_range=None):
        super(SSIM_1d_conv, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.val_range = val_range

        self.channel = 5
        self.window = create_window(window_size)

    def forward(self, time_1, time_2):
#         (_, channel, _) = time_1.size()
#         channel = time_1.size(1)
        channel = 5
        if channel == self.channel and self.window.dtype == time_1.dtype:
            window = self.window.to(device)
        else:
            window = create_window(self.window_size, channel)
            
            self.window = window
            self.channel = channel


        return ssim_1d_conv(time_1, time_2, window=window, window_size=self.window_size, size_average=self.size_average)
    
   