#!/usr/bin/python2 -utt
# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import numpy as np
import sys
import os
import time

from PIL import Image
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
import torch.optim as optim
from tqdm import tqdm
import math
import torch.nn.functional as F

from copy import deepcopy
from HardNet import HardNet
from OnePassSIR import OnePassSIR
from LAF import denormalizeLAFs, LAFs2ell,LAFs2ellT, abc2A
from Utils import line_prepender
from architectures import AffNetFastFullAff, OriNetFast
from time import time
USE_CUDA = True

try:
    input_img_fname = sys.argv[1]
    output_fname = sys.argv[2]
    th = 28.41#float(sys.argv[3])
except:
    print "Wrong input format. Try python hesaffnet.py imgs/cat.png cat.txt 5.3333"
    sys.exit(1)

def get_geometry_and_descriptors(img, det,desc):
    with torch.no_grad():
        tt = time()
        LAFs, resp = det(img)
        print('det time = ', time() - tt)
        tt = time()
        patches = det.extract_patches_from_pyr(LAFs, PS = 32)
        print('extract time = ', time() - tt)
        tt = time()
        descriptors = desc(patches)
        print('desc time = ', time() - tt)
    return LAFs, descriptors
def load_grayscale_var(fname):
    img = Image.open(fname).convert('RGB')
    img = np.mean(np.array(img), axis = 2)
    var_image = torch.autograd.Variable(torch.from_numpy(img.astype(np.float32)), volatile = True)
    var_image_reshape = var_image.view(1, 1, var_image.size(0),var_image.size(1))
    if USE_CUDA:
        var_image_reshape = var_image_reshape.cuda()
    return var_image_reshape


img = load_grayscale_var(input_img_fname)
AffNetPix = AffNetFastFullAff(PS = 32)
weightd_fname = '/home/old-ufo/dev/affnet-priv/pretrained/AffNet.pth'
checkpoint = torch.load(weightd_fname)
AffNetPix.load_state_dict(checkpoint['state_dict'])
AffNetPix.eval()
ONet = OriNetFast(PS =32)
o_fname = '/home/old-ufo/dev/affnet-priv/examples/hesaffnet/OriNet.pth'
checkpoint = torch.load(o_fname)
ONet.load_state_dict(checkpoint['state_dict'])
ONet.eval()
descriptor = HardNet()
model_weights = '../../HardNet++.pth'
hncheckpoint = torch.load(model_weights)
descriptor.load_state_dict(hncheckpoint['state_dict'])
descriptor.eval()

HA = OnePassSIR( mrSize = 5.192, num_features = -1, th = th, border = 15, num_Baum_iters = 1, AffNet = AffNetPix, OriNet = ONet)
import scipy.io as sio
if USE_CUDA:
    HA = HA.cuda()
    descriptor = descriptor.cuda()
with torch.no_grad():
    t = time()
    LAFs,descs = get_geometry_and_descriptors(img, HA,descriptor)
    #lt = time()
    #ells = LAFs2ellT(LAFs.cpu()).cpu().numpy()
    #print ('LAFs2ell time', time() - lt)
#print ('Total time', time() - t)
#sio.savemat('descs.mat', {'descs': descs.cpu().numpy()})
#sio.savemat('geoms.mat', {'LAFs': LAFs.cpu().numpy()})
np.save('lafs1.npy',LAFs.cpu().numpy())
#np.savetxt(output_fname, ells, delimiter=' ', fmt='%10.10f')
#line_prepender(output_fname, str(len(ells)))
#line_prepender(output_fname, '1.0')
#torch.save(descs, 'dd.t7')
