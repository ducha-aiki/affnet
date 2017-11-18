#!/usr/bin/python2 -utt
# -*- coding: utf-8 -*-
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
import time
import os
import cv2
import math
import numpy as np
from architectures import AffNetFast 
PS = 32
USE_CUDA = False


model = AffNetFast(PS = PS)
weightd_fname = '../../pretrained/AffNet.pth'

checkpoint = torch.load(weightd_fname)
model.load_state_dict(checkpoint['state_dict'])

model.eval()
if USE_CUDA:
    model.cuda()

try:
    input_img_fname = sys.argv[1]
    output_fname = sys.argv[2]
except:
    print "Wrong input format. Try ./detect_affine_shape.py imgs/ref.png out.txt"
    sys.exit(1)

image = cv2.imread(input_img_fname,0)
h,w = image.shape

n_patches =  h/w

descriptors_for_net = np.zeros((n_patches, 4))

patches = np.ndarray((n_patches, 1, PS, PS), dtype=np.float32)
for i in range(n_patches):
    patch =  image[i*(w): (i+1)*(w), 0:w]
    patches[i,0,:,:] = cv2.resize(patch,(PS,PS)) / 255.
bs = 128;
outs = []
n_batches = n_patches / bs + 1
t = time.time()
for batch_idx in range(n_batches):
    if batch_idx == n_batches - 1:
        if (batch_idx + 1) * bs > n_patches:
            end = n_patches
        else:
            end = (batch_idx + 1) * bs
    else:
        end = (batch_idx + 1) * bs
    if batch_idx * bs >= end:
        continue
    data_a = patches[batch_idx * bs: end, :, :, :].astype(np.float32)
    data_a = torch.from_numpy(data_a)
    if USE_CUDA:
        data_a = data_a.cuda()
    data_a = Variable(data_a, volatile=True)
    # compute output
    out_a = model(data_a)
    descriptors_for_net[batch_idx * bs: end,:] = out_a.data.cpu().numpy().reshape(-1, 4)
et  = time.time() - t
np.savetxt(output_fname,  descriptors_for_net, delimiter=' ', fmt='%10.5f')    
