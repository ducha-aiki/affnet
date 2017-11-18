from __future__ import division, print_function
import os
import errno
import numpy as np
import sys
from copy import deepcopy
import math
import torch
import torch.nn.init
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.autograd import Variable
from Utils import L2Norm, generate_2dgrid
from Utils import str2bool
from LAF import denormalizeLAFs, LAFs2ell, abc2A, extract_patches,normalizeLAFs,  get_rotation_matrix
from LAF import get_LAFs_scales, get_normalized_affine_shape
from LAF import rectifyAffineTransformationUpIsUp

class OriNetFast(nn.Module):
    def __init__(self, PS = 16):
        super(OriNetFast, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1, bias = False),
            nn.BatchNorm2d(16, affine=False),
            nn.ReLU(),
            nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1, bias = False),
            nn.BatchNorm2d(16, affine=False),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1, bias = False),
            nn.BatchNorm2d(32, affine=False),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1, bias = False),
            nn.BatchNorm2d(32, affine=False),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1, bias = False),
            nn.BatchNorm2d(64, affine=False),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias = False),
            nn.BatchNorm2d(64, affine=False),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Conv2d(64, 2, kernel_size=int(PS/4), stride=1,padding=1, bias = True),
            nn.Tanh(),
            nn.AdaptiveAvgPool2d(1)
        )
        self.PS = PS
        self.features.apply(self.weights_init)
        self.halfPS = int(PS/4)
        return
    def input_norm(self,x):
        flat = x.view(x.size(0), -1)
        mp = torch.mean(flat, dim=1)
        sp = torch.std(flat, dim=1) + 1e-7
        return (x - mp.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).expand_as(x)) / sp.unsqueeze(-1).unsqueeze(-1).unsqueeze(1).expand_as(x)
    def weights_init(self,m):
        if isinstance(m, nn.Conv2d):
            nn.init.orthogonal(m.weight.data, gain=0.9)
            try:
                nn.init.constant(m.bias.data, 0.01)
            except:
                pass
        return
    def forward(self, input, return_rot_matrix = True):
        xy = self.features(self.input_norm(input)).view(-1,2) 
        angle = torch.atan2(xy[:,0] + 1e-8, xy[:,1]+1e-8);
        if return_rot_matrix:
            return get_rotation_matrix(angle)
        return angle

class GHH(nn.Module):
    def __init__(self, n_in, n_out, s = 4, m = 4):
        super(GHH, self).__init__()
        self.n_out = n_out
        self.s = s
        self.m = m
        self.conv = nn.Linear(n_in, n_out * s * m)
        d = torch.arange(0, s)
        self.deltas = -1.0 * (d % 2 != 0).float()  + 1.0 * (d % 2 == 0).float()
        self.deltas = Variable(self.deltas)
        return
    def forward(self,x):
        x_feats = self.conv(x.view(x.size(0),-1)).view(x.size(0), self.n_out, self.s, self.m);
        max_feats = x_feats.max(dim = 3)[0];
        if x.is_cuda:
            self.deltas = self.deltas.cuda()
        else:
            self.deltas = self.deltas.cpu()
        out =  (max_feats * self.deltas.view(1,1,-1).expand_as(max_feats)).sum(dim = 2)
        return out

class YiNet(nn.Module):
    def __init__(self, PS = 28):
        super(YiNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 10, kernel_size=5, padding=0, bias = True),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding = 1),
            nn.Conv2d(10, 20, kernel_size=5, stride=1, padding=0, bias = True),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=4, stride=2, padding = 2),
            nn.Conv2d(20, 50, kernel_size=3, stride=1, padding=0, bias = True),
            nn.ReLU(),
            nn.AdaptiveMaxPool2d(1),
            GHH(50, 100),
            GHH(100, 2)
        )
        self.input_mean = 0.427117081207483
        self.input_std = 0.21888339179665006;
        self.PS = PS
        return
    def import_weights(self, dir_name):
        self.features[0].weight.data = torch.from_numpy(np.load(os.path.join(dir_name, 'layer0_W.npy'))).float()
        self.features[0].bias.data = torch.from_numpy(np.load(os.path.join(dir_name, 'layer0_b.npy'))).float().view(-1)
        self.features[3].weight.data = torch.from_numpy(np.load(os.path.join(dir_name, 'layer1_W.npy'))).float()
        self.features[3].bias.data = torch.from_numpy(np.load(os.path.join(dir_name, 'layer1_b.npy'))).float().view(-1)
        self.features[6].weight.data = torch.from_numpy(np.load(os.path.join(dir_name, 'layer2_W.npy'))).float()
        self.features[6].bias.data = torch.from_numpy(np.load(os.path.join(dir_name, 'layer2_b.npy'))).float().view(-1)
        self.features[9].conv.weight.data = torch.from_numpy(np.load(os.path.join(dir_name, 'layer3_W.npy'))).float().view(50, 1600).contiguous().t().contiguous()#.view(1600, 50, 1, 1).contiguous()
        self.features[9].conv.bias.data = torch.from_numpy(np.load(os.path.join(dir_name, 'layer3_b.npy'))).float().view(1600)
        self.features[10].conv.weight.data = torch.from_numpy(np.load(os.path.join(dir_name, 'layer4_W.npy'))).float().view(100, 32).contiguous().t().contiguous()#.view(32, 100, 1, 1).contiguous()
        self.features[10].conv.bias.data = torch.from_numpy(np.load(os.path.join(dir_name, 'layer4_b.npy'))).float().view(32)
        self.input_mean = float(np.load(os.path.join(dir_name, 'input_mean.npy')))
        self.input_std = float(np.load(os.path.join(dir_name, 'input_std.npy')))
        return
    def input_norm1(self,x):
        return (x - self.input_mean) / self.input_std
    def input_norm(self,x):
        flat = x.view(x.size(0), -1)
        mp = torch.mean(flat, dim=1)
        sp = torch.std(flat, dim=1) + 1e-7
        return (x - mp.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).expand_as(x)) / sp.unsqueeze(-1).unsqueeze(-1).unsqueeze(1).expand_as(x)
    def forward(self, input, return_rot_matrix = False):
        xy = self.features(self.input_norm(input))
        angle = torch.atan2(xy[:,0] + 1e-8, xy[:,1]+1e-8);
        if return_rot_matrix:
            return get_rotation_matrix(-angle)
        return angle
    
class AffNetFast(nn.Module):
    def __init__(self, PS = 32):
        super(AffNetFast, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1, bias = False),
            nn.BatchNorm2d(16, affine=False),
            nn.ReLU(),
            nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1, bias = False),
            nn.BatchNorm2d(16, affine=False),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1, bias = False),
            nn.BatchNorm2d(32, affine=False),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1, bias = False),
            nn.BatchNorm2d(32, affine=False),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1, bias = False),
            nn.BatchNorm2d(64, affine=False),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias = False),
            nn.BatchNorm2d(64, affine=False),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Conv2d(64, 3, kernel_size=8, stride=1, padding=0, bias = True),
            nn.Tanh(),
            nn.AdaptiveAvgPool2d(1)
        )
        self.PS = PS
        self.features.apply(self.weights_init)
        self.halfPS = int(PS/2)
        return
    def input_norm(self,x):
        flat = x.view(x.size(0), -1)
        mp = torch.mean(flat, dim=1)
        sp = torch.std(flat, dim=1) + 1e-7
        return (x - mp.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).expand_as(x)) / sp.unsqueeze(-1).unsqueeze(-1).unsqueeze(1).expand_as(x)
    def weights_init(self,m):
        if isinstance(m, nn.Conv2d):
            nn.init.orthogonal(m.weight.data, gain=0.8)
            try:
                nn.init.constant(m.bias.data, 0.01)
            except:
                pass
        return
    def forward(self, input, return_A_matrix = False):
        xy = self.features(self.input_norm(input)).view(-1,3)
        a1 = torch.cat([1.0 + xy[:,0].contiguous().view(-1,1,1), 0 * xy[:,0].contiguous().view(-1,1,1)], dim = 2).contiguous()
        a2 = torch.cat([xy[:,1].contiguous().view(-1,1,1), 1.0 + xy[:,2].contiguous().view(-1,1,1)], dim = 2).contiguous()
        return rectifyAffineTransformationUpIsUp(torch.cat([a1,a2], dim = 1).contiguous())

