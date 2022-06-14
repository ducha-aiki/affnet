from __future__ import division, print_function
import kornia.feature as KF
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
from LAF import rectifyAffineTransformationUpIsUp,rectifyAffineTransformationUpIsUpFullyConv

from kornia.filters.sobel import SpatialGradient
from LAF import *
import kornia
class LocalNorm2d(nn.Module):
    def __init__(self, kernel_size = 33):
        super(LocalNorm2d, self).__init__()
        self.ks = kernel_size
        self.pool = nn.AvgPool2d(kernel_size = self.ks, stride = 1,  padding = 0)
        self.eps = 1e-10
        return
    def forward(self,x):
        pd = int(self.ks/2)
        mean = self.pool(F.pad(x, (pd,pd,pd,pd), 'reflect'))
        return torch.clamp((x - mean) / (torch.sqrt(torch.abs(self.pool(F.pad(x*x,  (pd,pd,pd,pd), 'reflect')) - mean*mean )) + self.eps), min = -6.0, max = 6.0)

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
        self.features[9].conv.weight.data = torch.from_numpy(np.load(os.path.join(dir_name, 'layer3_W.npy'))).float().view(50, 1600).contiguous().t().contiguous()
        self.features[9].conv.bias.data = torch.from_numpy(np.load(os.path.join(dir_name, 'layer3_b.npy'))).float().view(1600)
        self.features[10].conv.weight.data = torch.from_numpy(np.load(os.path.join(dir_name, 'layer4_W.npy'))).float().view(100, 32).contiguous().t().contiguous()
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
class AffNetFast4(nn.Module):
    def __init__(self, PS = 32):
        super(AffNetFast4, self).__init__()
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
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Conv2d(64, 4, kernel_size=8, stride=1, padding=0, bias = True),
            nn.AdaptiveAvgPool2d(1)
        )
        self.PS = PS
        self.features.apply(self.weights_init)
        self.halfPS = int(PS/2)
        return
    def input_norm(self,x):
        flat = x.view(x.size(0), -1)
        mp = torch.mean(flat, dim=1).detach()
        sp = torch.std(flat, dim=1).detach() + 1e-7
        return (x - mp.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).expand_as(x)) / sp.unsqueeze(-1).unsqueeze(-1).unsqueeze(1).expand_as(x)
    def weights_init(self,m):
        if isinstance(m, nn.Conv2d):
            nn.init.orthogonal(m.weight.data, gain=0.8)
            try:
                if m.weight.data.shape[-1] == 8: #last layer:
                    nn.init.orthogonal(m.weight.data, gain=1.0)
                    print ('last layer init bias')
                    m.bias.data = torch.FloatTensor([1,0,0,1])
                else:
                    nn.init.constant(m.bias.data, 0.01)
            except:
                pass
        return
    def forward(self, input, return_A_matrix = False):
        xy = self.features(self.input_norm(input)).view(-1,2,2).contiguous()
        return rectifyAffineTransformationUpIsUp(xy).contiguous()

    
class AffNetFastSIFT(nn.Module):
    def __init__(self, PS = 32):
        super(AffNetFastSIFT, self).__init__()
        self.SIFT = nn.Sequential(KF.DenseSIFTDescriptor(8, 2, 4, True, stride=2, padding=0),
                                  nn.BatchNorm2d(32, affine=False))

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
            nn.ReLU())
        self.features2 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1, bias = False),
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
        self.features2.apply(self.weights_init)
        self.halfPS = int(PS/2)
        return
    def input_norm(self,x):
        flat = x.view(x.size(0), -1)
        mp = torch.mean(flat, dim=1).detach()
        sp = torch.std(flat, dim=1).detach() + 1e-7
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
        ni = self.input_norm(input)
        f1 = self.features(ni)
        sift = self.SIFT(ni)
        xy = self.features2(torch.cat([f1, sift], dim=1)).view(-1,3)
        a1 = torch.cat([1.0 + xy[:,0].contiguous().view(-1,1,1), 0 * xy[:,0].contiguous().view(-1,1,1)], dim = 2).contiguous()
        a2 = torch.cat([xy[:,1].contiguous().view(-1,1,1), 1.0 + xy[:,2].contiguous().view(-1,1,1)], dim = 2).contiguous()
        return rectifyAffineTransformationUpIsUp(torch.cat([a1,a2], dim = 1).contiguous())

class AffTFeat(nn.Module):
    def __init__(self, PS = 32):
        super(AffTFeat, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=7),
            nn.Tanh(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=6),
            nn.Tanh(),
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
        mp = torch.mean(flat, dim=1).detach()
        sp = torch.std(flat, dim=1).detach() + 1e-7
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
        mp = torch.mean(flat, dim=1).detach()
        sp = torch.std(flat, dim=1).detach() + 1e-7
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

class AffNetFast52RotUp(nn.Module):
    def __init__(self, PS = 32):
        super(AffNetFast52RotUp, self).__init__()
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
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Conv2d(64, 5, kernel_size=8, stride=1, padding=0, bias = True),
            nn.AdaptiveAvgPool2d(1)
        )
        self.PS = PS
        self.features.apply(self.weights_init)
        self.halfPS = int(PS/2)
        return
    def input_norm(self,x):
        flat = x.view(x.size(0), -1)
        mp = torch.mean(flat, dim=1).detach()
        sp = torch.std(flat, dim=1).detach() + 1e-7
        return (x - mp.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).expand_as(x)) / sp.unsqueeze(-1).unsqueeze(-1).unsqueeze(1).expand_as(x)
    def weights_init(self,m):
        if isinstance(m, nn.Conv2d):
            nn.init.orthogonal(m.weight.data, gain=0.8)
            try:
                if m.weight.data.shape[-1] == 8: #last layer:
                    nn.init.orthogonal(m.weight.data, gain=1.0)
                    print ('last layer init bias')
                    m.bias.data = torch.FloatTensor([1,0, 1, 0, 1])
                else:
                    nn.init.constant(m.bias.data, 0.01)
            except:
                pass
        return
    def forward(self, input, return_A_matrix = False):
        x  = self.features(self.input_norm(input)).view(-1,5)
        angle = torch.atan2(x[:,3], x[:,4]+1e-8);
        rot = get_rotation_matrix(angle)
        return torch.bmm(rot, rectifyAffineTransformationUpIsUp(torch.cat([torch.cat([x[:,0:1].view(-1,1,1), x[:,1:2].view(x.size(0),1,1).contiguous()], dim = 2), x[:,1:3].view(-1,1,2).contiguous()], dim = 1)).contiguous())

class AffNetFast52Rot(nn.Module):
    def __init__(self, PS = 32):
        super(AffNetFast52Rot, self).__init__()
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
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Conv2d(64, 5, kernel_size=8, stride=1, padding=0, bias = True),
            nn.AdaptiveAvgPool2d(1),
            nn.Tanh()
        )
        self.PS = PS
        self.features.apply(self.weights_init)
        self.halfPS = int(PS/2)
        return
    def input_norm(self,x):
        flat = x.view(x.size(0), -1)
        mp = torch.mean(flat, dim=1).detach()
        sp = torch.std(flat, dim=1).detach() + 1e-7
        return (x - mp.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).expand_as(x)) / sp.unsqueeze(-1).unsqueeze(-1).unsqueeze(1).expand_as(x)
    def weights_init(self,m):
        if isinstance(m, nn.Conv2d):
            nn.init.orthogonal(m.weight.data, gain=0.8)
            try:
                if m.weight.data.shape[-1] == 8: #last layer:
                    nn.init.orthogonal(m.weight.data, gain=1.0)
                    print ('last layer init bias')
                    m.bias.data = torch.FloatTensor([0.8,0, 0.8, 0, 1])
                else:
                    nn.init.constant(m.bias.data, 0.01)
            except:
                pass
        return
    def forward(self, input, return_A_matrix = False):
        x  = self.features(self.input_norm(input)).view(-1,5)
        angle = torch.atan2(x[:,3], x[:,4]+1e-8);
        rot = get_rotation_matrix(angle)
        return torch.bmm(rot, torch.cat([torch.cat([x[:,0:1].view(-1,1,1), x[:,1:2].view(x.size(0),1,1).contiguous()], dim = 2), x[:,1:3].view(-1,1,2).contiguous()], dim = 1))

class AffNetFast5Rot(nn.Module):
    def __init__(self, PS = 32):
        super(AffNetFast5Rot, self).__init__()
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
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Conv2d(64, 5, kernel_size=8, stride=1, padding=0, bias = True),
            nn.AdaptiveAvgPool2d(1)
        )
        self.PS = PS
        self.features.apply(self.weights_init)
        self.halfPS = int(PS/2)
        return
    def input_norm(self,x):
        flat = x.view(x.size(0), -1)
        mp = torch.mean(flat, dim=1).detach()
        sp = torch.std(flat, dim=1).detach() + 1e-7
        return (x - mp.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).expand_as(x)) / sp.unsqueeze(-1).unsqueeze(-1).unsqueeze(1).expand_as(x)
    def weights_init(self,m):
        if isinstance(m, nn.Conv2d):
            nn.init.orthogonal(m.weight.data, gain=0.8)
            try:
                if m.weight.data.shape[-1] == 8: #last layer:
                    nn.init.orthogonal(m.weight.data, gain=1.0)
                    print ('last layer init bias')
                    m.bias.data = torch.FloatTensor([1,0, 1, 0, 1])
                else:
                    nn.init.constant(m.bias.data, 0.01)
            except:
                pass
        return
    def forward(self, input, return_A_matrix = False):
        x  = self.features(self.input_norm(input)).view(-1,5)
        rot = get_rotation_matrix(torch.atan2(x[:,3], x[:,4]+1e-8))
        if input.is_cuda:
            return torch.bmm(rot, torch.cat([torch.cat([x[:,0:1].view(-1,1,1), torch.zeros(x.size(0),1,1).cuda()], dim = 2), x[:,1:3].view(-1,1,2).contiguous()], dim = 1))
        else:
            return torch.bmm(rot, torch.cat([torch.cat([x[:,0:1].view(-1,1,1), torch.zeros(x.size(0),1,1)], dim = 2), x[:,1:3].view(-1,1,2).contiguous()], dim = 1))

class AffNetFast4Rot(nn.Module):
    def __init__(self, PS = 32):
        super(AffNetFast4Rot, self).__init__()
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
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Conv2d(64, 4, kernel_size=8, stride=1, padding=0, bias = True),
            nn.AdaptiveAvgPool2d(1),
            nn.Tanh()
        )
        self.PS = PS
        self.features.apply(self.weights_init)
        self.halfPS = int(PS/2)
        return
    def input_norm(self,x):
        flat = x.view(x.size(0), -1)
        mp = torch.mean(flat, dim=1).detach()
        sp = torch.std(flat, dim=1).detach() + 1e-7
        return (x - mp.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).expand_as(x)) / sp.unsqueeze(-1).unsqueeze(-1).unsqueeze(1).expand_as(x)
    def weights_init(self,m):
        if isinstance(m, nn.Conv2d):
            nn.init.orthogonal(m.weight.data, gain=0.8)
            try:
                if m.weight.data.shape[-1] == 8: #last layer:
                    nn.init.orthogonal(m.weight.data, gain=1.0)
                    print ('last layer init bias')
                    m.bias.data = torch.FloatTensor([0.8,0,0,0.8])
                else:
                    nn.init.constant(m.bias.data, 0.01)
            except:
                pass
        return
    def forward(self, input, return_A_matrix = False):
        return self.features(self.input_norm(input)).view(-1,2,2).contiguous()

class AffNetFast4RotNosc(nn.Module):
    def __init__(self, PS = 32):
        super(AffNetFast4RotNosc, self).__init__()
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
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Conv2d(64, 4, kernel_size=8, stride=1, padding=0, bias = True),
            nn.AdaptiveAvgPool2d(1)
        )
        self.PS = PS
        self.features.apply(self.weights_init)
        self.halfPS = int(PS/2)
        return
    def input_norm(self,x):
        flat = x.view(x.size(0), -1)
        mp = torch.mean(flat, dim=1).detach()
        sp = torch.std(flat, dim=1).detach() + 1e-7
        return (x - mp.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).expand_as(x)) / sp.unsqueeze(-1).unsqueeze(-1).unsqueeze(1).expand_as(x)
    def weights_init(self,m):
        if isinstance(m, nn.Conv2d):
            nn.init.orthogonal(m.weight.data, gain=0.8)
            try:
                if m.weight.data.shape[-1] == 8: #last layer:
                    nn.init.orthogonal(m.weight.data, gain=1.0)
                    print ('last layer init bias')
                    m.bias.data = torch.FloatTensor([1,0,0,1])
                else:
                    nn.init.constant(m.bias.data, 0.01)
            except:
                pass
        return
    def forward(self, input, return_A_matrix = False):
        A = self.features(self.input_norm(input)).view(-1,2,2).contiguous()
        scale =  torch.sqrt(torch.abs(A[:,0,0]*A[:,1,1] - A[:,1,0]*A[:,0,1] + 1e-10))
        return A / (scale.view(-1,1,1).repeat(1,2,2) + 1e-8)

class AffNetFastScale(nn.Module):
    def __init__(self, PS = 32):
        super(AffNetFastScale, self).__init__()
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
            nn.Conv2d(64, 4, kernel_size=8, stride=1, padding=0, bias = True),
            nn.Tanh(),
            nn.AdaptiveAvgPool2d(1)
        )
        self.PS = PS
        self.features.apply(self.weights_init)
        self.halfPS = int(PS/2)
        return
    def input_norm(self,x):
        flat = x.view(x.size(0), -1)
        mp = torch.mean(flat, dim=1).detach()
        sp = torch.std(flat, dim=1).detach() + 1e-7
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
        xy = self.features(self.input_norm(input)).view(-1,4)
        a1 = torch.cat([1.0 + xy[:,0].contiguous().view(-1,1,1), 0 * xy[:,0].contiguous().view(-1,1,1)], dim = 2).contiguous()
        a2 = torch.cat([xy[:,1].contiguous().view(-1,1,1), 1.0 + xy[:,2].contiguous().view(-1,1,1)], dim = 2).contiguous()
        scale = torch.exp(xy[:,3].contiguous().view(-1,1,1).repeat(1,2,2))
        return scale * rectifyAffineTransformationUpIsUp(torch.cat([a1,a2], dim = 1).contiguous())

class AffNetFast2Par(nn.Module):
    def __init__(self, PS = 32):
        super(AffNetFast2Par, self).__init__()
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
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Conv2d(64, 3, kernel_size=8, stride=1, padding=0, bias = True),
            nn.AdaptiveAvgPool2d(1)
        )
        self.PS = PS
        self.features.apply(self.weights_init)
        self.halfPS = int(PS/2)
        return
    def input_norm(self,x):
        flat = x.view(x.size(0), -1)
        mp = torch.mean(flat, dim=1).detach()
        sp = torch.std(flat, dim=1).detach() + 1e-7
        return (x - mp.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).expand_as(x)) / sp.unsqueeze(-1).unsqueeze(-1).unsqueeze(1).expand_as(x)
    def weights_init(self,m):
        if isinstance(m, nn.Conv2d):
            nn.init.orthogonal(m.weight.data, gain=0.8)
            try:
                if m.weight.data.shape[-1] == 8: #last layer:
                    nn.init.orthogonal(m.weight.data, gain=1.0)
                    print ('last layer init bias')
                    m.bias.data = torch.FloatTensor([0, 0, 1 ])
                else:
                    nn.init.constant(m.bias.data, 0.01)
            except:
                pass
        return
    def forward(self, input, return_A_matrix = False):
        x  = self.features(self.input_norm(input)).view(-1,3)
        angle = torch.atan2(x[:,1], x[:,2]+1e-8);
        rot = get_rotation_matrix(angle)
        tilt = torch.exp(1.8 * F.tanh(x[:,0]))
        tilt_matrix = torch.eye(2).unsqueeze(0).repeat(input.size(0),1,1)
        if x.is_cuda:
            tilt_matrix = tilt_matrix.cuda()
        tilt_matrix[:,0,0] = torch.sqrt(tilt)
        tilt_matrix[:,1,1] = 1.0 / torch.sqrt(tilt)
        return rectifyAffineTransformationUpIsUp(torch.bmm(rot, tilt_matrix)).contiguous()

class AffNetFastFullConv(nn.Module):
    def __init__(self, PS = 32, stride = 2):
        super(AffNetFastFullConv, self).__init__()
        self.lrn = LocalNorm2d(33)
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1, bias = False),
            nn.BatchNorm2d(16, affine=False),
            nn.ReLU(),
            nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1, bias = False),
            nn.BatchNorm2d(16, affine=False),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=stride, padding=1, bias = False),
            nn.BatchNorm2d(32, affine=False),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1, bias = False),
            nn.BatchNorm2d(32, affine=False),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=stride, padding=1, bias = False),
            nn.BatchNorm2d(64, affine=False),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias = False),
            nn.BatchNorm2d(64, affine=False),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Conv2d(64, 3, kernel_size=8, stride=1, padding = 0, bias = True),
        )
        self.stride = stride
        self.PS = PS
        self.features.apply(self.weights_init)
        self.halfPS = int(PS/2)
        return
    def weights_init(self,m):
        if isinstance(m, nn.Conv2d):
            nn.init.orthogonal(m.weight.data, gain=0.8)
            try:
                nn.init.constant(m.bias.data, 0.01)
            except:
                pass
        return
    def forward(self, input, return_A_matrix = False):
        norm_inp  = self.lrn(input)
        ff = self.features(F.pad(norm_inp, (14,14,14,14), 'reflect'))
        xy = F.tanh(F.upsample(ff, (input.size(2), input.size(3)),mode='bilinear'))
        a0bc = torch.cat([1.0 + xy[:,0:1,:,:].contiguous(), 0*xy[:,1:2,:,:].contiguous(),
                          xy[:,1:2,:,:].contiguous(),  1.0 + xy[:,2:,:,:].contiguous()], dim = 1).contiguous()
        return rectifyAffineTransformationUpIsUpFullyConv(a0bc).contiguous()
    
class AffNetFast52RotL(nn.Module):
    def __init__(self, PS = 32):
        super(AffNetFast52RotL, self).__init__()
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
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Conv2d(64, 5, kernel_size=8, stride=1, padding=0, bias = True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.PS = PS
        self.features.apply(self.weights_init)
        self.halfPS = int(PS/2)
        return
    def input_norm(self,x):
        flat = x.view(x.size(0), -1)
        mp = torch.mean(flat, dim=1).detach()
        sp = torch.std(flat, dim=1).detach() + 1e-7
        return (x - mp.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).expand_as(x)) / sp.unsqueeze(-1).unsqueeze(-1).unsqueeze(1).expand_as(x)
    def weights_init(self,m):
        if isinstance(m, nn.Conv2d):
            nn.init.orthogonal(m.weight.data, gain=0.8)
            try:
                if m.weight.data.shape[-1] == 8: #last layer:
                    nn.init.orthogonal(m.weight.data, gain=1.0)
                    print ('last layer init bias')
                    m.bias.data = torch.FloatTensor([0.8,0, 0.8, 0, 1])
                else:
                    nn.init.constant(m.bias.data, 0.01)
            except:
                pass
        return
    def forward(self, input, return_A_matrix = False):
        x  = self.features(self.input_norm(input)).view(-1,5)
        angle = torch.atan2(x[:,3], x[:,4]+1e-8);
        rot = get_rotation_matrix(angle)
        return torch.bmm(rot, torch.cat([torch.cat([x[:,0:1].view(-1,1,1), x[:,1:2].view(x.size(0),1,1).contiguous()], dim = 2), x[:,1:3].view(-1,1,2).contiguous()], dim = 1))

class AffNetFastBias(nn.Module):
    def __init__(self, PS = 32):
        super(AffNetFastBias, self).__init__()
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
        mp = torch.mean(flat, dim=1).detach()
        sp = torch.std(flat, dim=1).detach() + 1e-7
        return (x - mp.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).expand_as(x)) / sp.unsqueeze(-1).unsqueeze(-1).unsqueeze(1).expand_as(x)
    def weights_init(self,m):
        if isinstance(m, nn.Conv2d):
            nn.init.orthogonal(m.weight.data, gain=0.8)
            try:
                if m.weight.data.shape[-1] == 8: #last layer:
                    nn.init.orthogonal(m.weight.data, gain=1.0)
                    print ('last layer init bias')
                    m.bias.data = torch.FloatTensor([0.8, 0, 0.8 ])
                else:
                    nn.init.constant(m.bias.data, 0.01)
            except:
                pass
        return
    def forward(self, input, return_A_matrix = False):
        xy = self.features(self.input_norm(input)).view(-1,3)
        a1 = torch.cat([xy[:,0].contiguous().view(-1,1,1), 0 * xy[:,0].contiguous().view(-1,1,1)], dim = 2).contiguous()
        a2 = torch.cat([xy[:,1].contiguous().view(-1,1,1), xy[:,2].contiguous().view(-1,1,1)], dim = 2).contiguous()
        return rectifyAffineTransformationUpIsUp(torch.cat([a1,a2], dim = 1).contiguous())


class AffNetWithHandCrafted(nn.Module):
    def __init__(self, PS = 32):
        super(AffNetWithHandCrafted, self).__init__()
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
        self.gradient = SpatialGradient('sobel', 1)
        self.gradient2 = SpatialGradient('sobel', 2)
        self.mix = nn.Sequential(
                nn.BatchNorm2d(6, affine=False),
                nn.Conv2d(6, 12, kernel_size=1, stride=1, padding=0, bias = False),
                nn.BatchNorm2d(12, affine=False),
                nn.Tanh(),
                nn.Conv2d(12, 3, kernel_size=1, stride=1, padding=0, bias = True),
                nn.Tanh())
        self.PS = PS
        self.features.apply(self.weights_init)
        self.halfPS = int(PS/2)
        self.register_buffer('circular_shape', torch.tensor([1.0, 0.0, 1.0]))
        return
    def input_norm(self,x):
        flat = x.view(x.size(0), -1)
        mp = torch.mean(flat, dim=1).detach()
        sp = torch.std(flat, dim=1).detach() + 1e-7
        return (x - mp.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).expand_as(x)) / sp.unsqueeze(-1).unsqueeze(-1).unsqueeze(1).expand_as(x)
    def weights_init(self,m):
        if isinstance(m, nn.Conv2d):
            nn.init.orthogonal(m.weight.data, gain=0.8)
            try:
                nn.init.constant(m.bias.data, 0.01)
            except:
                pass
        return
    def BaumbergAffine(self, input):
        
        grads: torch.Tensor = self.gradient(input)
        # unpack the edges
        gx: torch.Tensor = grads[:, :, 0]
        gy: torch.Tensor = grads[:, :, 1]
        # abc == 1st axis, mixture, 2nd axis. Ellipse_shape is a 2nd moment matrix.
        ellipse_shape = torch.cat(
            [
                gx.pow(2).mean(dim=2).mean(dim=2, keepdim=True),
                (gx * gy).mean(dim=2).mean(dim=2, keepdim=True),
                gy.pow(2).mean(dim=2).mean(dim=2, keepdim=True),
            ],
            dim=2,
        )

        # Now lets detect degenerate cases: when 2 or 3 elements are close to zero (e.g. if patch is completely black
        bad_mask = ((ellipse_shape < 1e-6).float().sum(dim=2, keepdim=True) >= 2).to(ellipse_shape.dtype)
        # We will replace degenerate shape with circular shapes.
        circular_shape = self.circular_shape.to(ellipse_shape.device).to(ellipse_shape.dtype).view(1, 1, 3)
        ellipse_shape = ellipse_shape * (1.0 - bad_mask) + circular_shape * bad_mask
        # normalizatio'
        ellipse_shape = ellipse_shape / ellipse_shape.max(dim=2, keepdim=True)[0]
        return ellipse_shape

    def Baumberg2ndgrad(self, input):
        grads: torch.Tensor = self.gradient2(input) 
        # unpack the edges
        gxx: torch.Tensor = grads[:, :, 0]
        gxy: torch.Tensor = grads[:, :, 1]
        gyy: torch.Tensor = grads[:, :, 2]
        # abc == 1st axis, mixture, 2nd axis. Ellipse_shape is a 2nd moment matrix.
        ellipse_shape = torch.cat(
            [
                gxx.mean(dim=2).mean(dim=2, keepdim=True),
                gxy.mean(dim=2).mean(dim=2, keepdim=True),
                gyy.mean(dim=2).mean(dim=2, keepdim=True),
            ],
            dim=2,
        )

        # Now lets detect degenerate cases: when 2 or 3 elements are close to zero (e.g. if patch is completely black
        bad_mask = ((ellipse_shape.abs() < 1e-6).float().sum(dim=2, keepdim=True) >= 2).to(ellipse_shape.dtype)
        # We will replace degenerate shape with circular shapes.
        circular_shape = self.circular_shape.to(ellipse_shape.device).to(ellipse_shape.dtype).view(1, 1, 3)
        ellipse_shape = ellipse_shape * (1.0 - bad_mask) + circular_shape * bad_mask
        # normalizatio'
        ellipse_shape = ellipse_shape / ellipse_shape.abs().max(dim=2, keepdim=True)[0]
        return ellipse_shape
    def forward(self, input, return_A_matrix = False):
        xy = self.features(self.input_norm(input)).view(-1,3)
        baum = self.BaumbergAffine(input)
        #hes = self.Baumberg2ndgrad(input)
        out_cat = torch.cat([xy.reshape(-1,3), baum.reshape(-1,3)], dim=1).view(-1, 6, 1, 1)
        xy_again = self.mix(out_cat).view(-1,3)
        
        new_a1 = torch.cat([1.0 + xy_again[:,0].contiguous().view(-1,1,1), 0 * xy_again[:,0].contiguous().view(-1,1,1)], dim = 2).contiguous()
        new_a2 = torch.cat([xy_again[:,1].contiguous().view(-1,1,1), 1.0 +xy_again[:,2].contiguous().view(-1,1,1)], dim = 2).contiguous()
        new_out =  rectifyAffineTransformationUpIsUp(torch.cat([new_a1, new_a2], dim = 1).contiguous())
        
        return new_out
