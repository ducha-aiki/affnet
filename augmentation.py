import numpy as np
from PIL import Image

import sys
from copy import deepcopy
import argparse
import math
import torch.utils.data as data
import torch
import torch.nn.init
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import torchvision.transforms as transforms
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
from LAF import get_rotation_matrix,get_normalized_affine_shape 

def get_random_rotation_LAFs(patches, angle_mag = math.pi):
    rot_LAFs = Variable(torch.FloatTensor([[0.5, 0, 0.5],[0, 0.5, 0.5]]).unsqueeze(0).repeat(patches.size(0),1,1));
    phi  = (Variable(2.0 * torch.rand(patches.size(0)) - 1.0) ).view(-1,1,1)
    if patches.is_cuda:
        rot_LAFs = rot_LAFs.cuda()
        phi = phi.cuda()
    rotmat = get_rotation_matrix(angle_mag * phi)
    inv_rotmat = get_rotation_matrix(-angle_mag * phi)
    rot_LAFs[:,0:2,0:2]  = torch.bmm(rotmat, rot_LAFs[:,0:2,0:2]);
    return rot_LAFs, inv_rotmat

def get_random_shifts_LAFs(patches, w_mag, h_mag = 3):
    shift_w =  (torch.IntTensor(patches.size(0)).random_(2*w_mag) - w_mag / 2).float() / 2.0
    shift_h =  (torch.IntTensor(patches.size(0)).random_(2*w_mag) - w_mag / 2).float() / 2.0
    if patches.is_cuda:
        shift_h = shift_h.cuda()
        shift_w = shift_w.cuda()
    shift_h = Variable(shift_h)
    shift_w = Variable(shift_w)
    return shift_w, shift_h

def get_random_norm_affine_LAFs(patches, max_tilt = 1.0):
    assert max_tilt > 0
    aff_LAFs = Variable(torch.FloatTensor([[0.5, 0, 0.5],[0, 0.5, 0.5]]).unsqueeze(0).repeat(patches.size(0),1,1));
    tilt = Variable( 1/max_tilt + (max_tilt - 1./max_tilt)* torch.rand(patches.size(0), 1, 1));
    phi  = math.pi * (Variable(2.0 * torch.rand(patches.size(0)) - 1.0) ).view(-1,1,1)
    if patches.is_cuda:
        tilt = tilt.cuda()
        phi = phi.cuda()
        aff_LAFs = aff_LAFs.cuda()
    TA = get_normalized_affine_shape(tilt, phi)
    #inv_TA = Variable(torch.zeros(patches.size(0),2,2));
    #if patches.is_cuda:
    #    inv_TA = inv_TA.cuda()
    #for i in range(len(inv_TA)):
    #    inv_TA[i,:,:] = TA[i,:,:].inverse();
    aff_LAFs[:,0:2,0:2]  = torch.bmm(TA, aff_LAFs[:,0:2,0:2]);
    return aff_LAFs, None#inv_TA;

