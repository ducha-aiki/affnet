#from __future__ import division, print_function
import os
import errno
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
import torchvision.datasets as dset

import torchvision.transforms as transforms
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
from tqdm import tqdm
import random
import cv2
import copy
from Utils import L2Norm, cv2_scale
#from Utils import np_reshape64 as np_reshape
np_reshape = lambda x: np.reshape(x, (64, 64, 1))
from Utils import str2bool
from dataset import HPatchesDM,TripletPhotoTour, TotalDatasetsLoader
cv2_scale40 = lambda x: cv2.resize(x, dsize=(40, 40),
                                 interpolation=cv2.INTER_LINEAR)
from augmentation import get_random_norm_affine_LAFs,get_random_rotation_LAFs, get_random_shifts_LAFs
from LAF import denormalizeLAFs, LAFs2ell, abc2A, extract_patches,normalizeLAFs
from pytorch_sift import SIFTNet
from HardNet import HardNet
from Losses import loss_HardNet
PS = 32
# Training settings
parser = argparse.ArgumentParser(description='PyTorch AffNet')

parser.add_argument('--dataroot', type=str,
                    default='datasets/',
                    help='path to dataset')
parser.add_argument('--log-dir', default='./logs',
                    help='folder to output model checkpoints')
parser.add_argument('--num-workers', default= 8,
                    help='Number of workers to be created')
parser.add_argument('--pin-memory',type=bool, default= True,
                    help='')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--epochs', type=int, default=10, metavar='E',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--batch-size', type=int, default=128, metavar='BS',
                    help='input batch size for training (default: 128)')
parser.add_argument('--test-batch-size', type=int, default=1024, metavar='BST',
                    help='input batch size for testing (default: 1000)')
parser.add_argument('--n-pairs', type=int, default=500000, metavar='N',
                    help='how many pairs will generate from the dataset')
parser.add_argument('--n-test-pairs', type=int, default=50000, metavar='N',
                    help='how many pairs will generate from the test dataset')
parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                    help='learning rate (default: 0.01)')
parser.add_argument('--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
# Device options
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--gpu-id', default='0', type=str,
                    help='id(s) for CUDA_VISIBLE_DEVICES')
parser.add_argument('--expname', default='', type=str,
                    help='experiment name')
parser.add_argument('--seed', type=int, default=0, metavar='S',
                    help='random seed (default: 0)')
parser.add_argument('--log-interval', type=int, default=10, metavar='LI',
                    help='how many batches to wait before logging training status')
parser.add_argument('--descriptor', type=str,
                    default='pixels',
                    help='what is minimized. Variants: pixels, SIFT, HardNet')
parser.add_argument('--merge', type=str,
                    default='sum',
                    help='Combination of geom loss and descriptor loss: mul, sum')
parser.add_argument('--geom-loss-coef', type=float,
                    default=1.0,
                    help='coef of geom loss (linear if sum, power if mul) (defualt 1.0')
parser.add_argument('--descr-loss-coef', type=float,
                    default=0.0,
                    help='coef of descr loss (linear if sum, power if mul (default 0)')


args = parser.parse_args()


# set the device to use by setting CUDA_VISIBLE_DEVICES env variable in
# order to prevent any memory allocation on unused GPUs
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id

args.cuda = not args.no_cuda and torch.cuda.is_available()
if args.cuda:
    cudnn.benchmark = True
    torch.cuda.manual_seed_all(args.seed)

# create loggin directory
if not os.path.exists(args.log_dir):
    os.makedirs(args.log_dir)

# set random seeds
torch.manual_seed(args.seed)
np.random.seed(args.seed)

if args.descriptor == 'SIFT':
    descriptor = SIFTNet(patch_size=PS)
    if not args.no_cuda:
        descriptor = descriptor.cuda()
elif args.descriptor == 'HardNet':
    descriptor = HardNet()
    if not args.no_cuda:
        descriptor = descriptor.cuda()
    model_weights = 'HardNet++.pth'
    hncheckpoint = torch.load(model_weights)
    descriptor.load_state_dict(hncheckpoint['state_dict'])
    descriptor.train()
else:
    descriptor = lambda x: x.view(x.size(0),-1)

suffix= args.expname + '_6Brown_' + args.merge + '_' + args.descriptor + '_' + str(args.lr) + '_' + str(args.n_pairs) 
##########################################3
def create_loaders():

    kwargs = {'num_workers': args.num_workers, 'pin_memory': args.pin_memory} if args.cuda else {}
    transform = transforms.Compose([
            transforms.Lambda(np_reshape),
            transforms.ToTensor()
            ])

    train_loader = torch.utils.data.DataLoader(
            TotalDatasetsLoader(datasets_path = args.dataroot, train=True,
                             n_triplets = args.n_pairs,
                             fliprot=True,
                             batch_size=args.batch_size,
                             download=True,
                             transform=transform),
                             batch_size=args.batch_size,
                             shuffle=False, **kwargs)

    test_loader = torch.utils.data.DataLoader(
            HPatchesDM('dataset/HP_HessianPatches/','', train=False,
                             n_pairs = args.n_test_pairs,
                             batch_size=args.test_batch_size,
                             download=True,
                             transform=transforms.Compose([])),
                             batch_size=args.test_batch_size,
                             shuffle=False, **kwargs)
    return train_loader, test_loader

def extract_and_crop_patches_by_predicted_transform(patches, trans, crop_size = 32):
    assert patches.size(0) == trans.size(0)
    st = int((patches.size(2) - crop_size) / 2)
    fin = st + crop_size
    rot_LAFs = Variable(torch.FloatTensor([[0.5, 0, 0.5],[0, 0.5, 0.5]]).unsqueeze(0).repeat(patches.size(0),1,1));
    if patches.is_cuda:
        rot_LAFs = rot_LAFs.cuda()
        trans = trans.cuda()
    rot_LAFs1  = torch.cat([torch.bmm(trans, rot_LAFs[:,0:2,0:2]), rot_LAFs[:,0:2,2:]], dim = 2);
    return extract_patches(patches,  rot_LAFs1, PS = patches.size(2))[:,:, st:fin, st:fin].contiguous()
    

def train(train_loader, model, optimizer, epoch):
    # switch to train mode
    model.train()
    pbar = tqdm(enumerate(train_loader))
    for batch_idx, data in pbar:
        data_a, data_p = data
        if args.cuda:
            data_a, data_p  = data_a.float().cuda(), data_p.float().cuda()
            data_a, data_p = Variable(data_a), Variable(data_p)
        st = int((data_p.size(2) - model.PS)/2)
        fin = st + model.PS
        #
        #
        max_tilt = 3.0
        if epoch > 1:
            max_tilt = 4.0
        if epoch > 3:
            max_tilt = 4.5
        if epoch > 5:
            max_tilt = 4.8
        rot_LAFs_a, inv_rotmat_a = get_random_rotation_LAFs(data_a, math.pi)
        aff_LAFs_a, inv_TA_a = get_random_norm_affine_LAFs(data_a, max_tilt);
        aff_LAFs_a[:,0:2,0:2] = torch.bmm(rot_LAFs_a[:,0:2,0:2],aff_LAFs_a[:,0:2,0:2])
        data_a_aff = extract_patches(data_a,  aff_LAFs_a, PS = data_a.size(2))
        data_a_aff_crop = data_a_aff[:,:, st:fin, st:fin].contiguous()
        aff_LAFs_p, inv_TA_p = get_random_norm_affine_LAFs(data_p, max_tilt);
        aff_LAFs_p[:,0:2,0:2] = torch.bmm(rot_LAFs_a[:,0:2,0:2],aff_LAFs_p[:,0:2,0:2])
        data_p_aff = extract_patches(data_p,  aff_LAFs_p, PS = data_p.size(2))
        data_p_aff_crop = data_p_aff[:,:, st:fin, st:fin].contiguous()
        out_a_aff, out_p_aff = model(data_a_aff_crop,True), model(data_p_aff_crop,True)
        out_p_aff_back = torch.bmm(inv_TA_p, out_p_aff)
        out_a_aff_back = torch.bmm(inv_TA_a, out_a_aff)
        ######Apply rot and get sifts
        out_patches_a_crop = extract_and_crop_patches_by_predicted_transform(data_a_aff, out_a_aff, crop_size = model.PS)
        out_patches_p_crop = extract_and_crop_patches_by_predicted_transform(data_p_aff, out_p_aff, crop_size = model.PS)
        desc_a = descriptor(out_patches_a_crop)
        desc_p = descriptor(out_patches_p_crop)
        descr_dist =  torch.sqrt(((desc_a - desc_p)**2).view(data_a.size(0),-1).sum(dim=1) + 1e-6)
        descr_loss = loss_HardNet(desc_a,desc_p, anchor_swap = True); 
        geom_dist = torch.sqrt(((out_a_aff_back - out_p_aff_back)**2 ).view(-1,4).mean(dim=1) + 1e-8)
        if args.merge == 'sum':
            loss = descr_loss
        elif args.merge == 'mul':
            loss = descr_loss
        else:
            print ('Unknown merge option')
            sys.exit(0)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        adjust_learning_rate(optimizer)
        if batch_idx % 2 == 0:
            pbar.set_description(
                'Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.4f}, {},{:.4f}'.format(
                    epoch, batch_idx * len(data_a), len(train_loader.dataset),
                           100. * batch_idx / len(train_loader),
                    loss.data[0], geom_dist.mean().data[0], descr_dist.mean().data[0]))
    torch.save({'epoch': epoch + 1, 'state_dict': model.state_dict()},
               '{}/checkpoint_{}.pth'.format(LOG_DIR,epoch))

def test(test_loader, model, epoch):
    # switch to evaluate mode
    model.eval()
    geom_distances, desc_distances = [], []
    pbar = tqdm(enumerate(test_loader))
    for batch_idx, data in pbar:
        data_a, data_p = data
        if args.cuda:
            data_a, data_p = data_a.float().cuda(), data_p.float().cuda()
        data_a, data_p = Variable(data_a, volatile=True), Variable(data_p, volatile=True)
        st = int((data_p.size(2) - model.PS)/2)
        fin = st + model.PS
        aff_LAFs_a, inv_TA_a = get_random_norm_affine_LAFs(data_a, 3.0);
        shift_w_a, shift_h_a = get_random_shifts_LAFs(data_a, 3, 3)
        aff_LAFs_a[:,0,2] = aff_LAFs_a[:,0,2] + shift_w_a / float(data_a.size(3))
        aff_LAFs_a[:,1,2] = aff_LAFs_a[:,1,2] + shift_h_a / float(data_a.size(2))
        data_a_aff = extract_patches(data_a,  aff_LAFs_a, PS = data_a.size(2))
        data_a_aff_crop = data_a_aff[:,:, st:fin, st:fin].contiguous()
        aff_LAFs_p, inv_TA_p = get_random_norm_affine_LAFs(data_p, 3.0);
        shift_w_p, shift_h_p = get_random_shifts_LAFs(data_p, 3, 3)
        aff_LAFs_p[:,0,2] = aff_LAFs_p[:,0,2] + shift_w_p / float(data_a.size(3))
        aff_LAFs_p[:,1,2] = aff_LAFs_p[:,1,2] + shift_h_p / float(data_a.size(2))
        data_p_aff = extract_patches(data_p,  aff_LAFs_p, PS = data_p.size(2))
        data_p_aff_crop = data_p_aff[:,:, st:fin, st:fin].contiguous()
        out_a_aff, out_p_aff = model(data_a_aff_crop,True), model(data_p_aff_crop,True)
        out_p_aff_back = torch.bmm(inv_TA_p, out_p_aff)
        out_a_aff_back = torch.bmm(inv_TA_a, out_a_aff)
        ######Apply rot and get sifts
        out_patches_a_crop = extract_and_crop_patches_by_predicted_transform(data_a_aff, out_a_aff, crop_size = model.PS)
        out_patches_p_crop = extract_and_crop_patches_by_predicted_transform(data_p_aff, out_p_aff, crop_size = model.PS)
        desc_a = descriptor(out_patches_a_crop)
        desc_p = descriptor(out_patches_p_crop)
        descr_dist =  torch.sqrt(((desc_a - desc_p)**2).view(data_a.size(0),-1).sum(dim=1) + 1e-6) / float(desc_a.size(1))
        geom_dist = torch.sqrt(((out_a_aff_back - out_p_aff_back)**2 ).view(-1,4).mean(dim=1) + 1e-8)
        geom_distances.append(geom_dist.mean().data.cpu().numpy().reshape(-1,1))
        desc_distances.append(descr_dist.mean().data.cpu().numpy().reshape(-1,1))
        if batch_idx % args.log_interval == 0:
            pbar.set_description(' Test Epoch: {} [{}/{} ({:.0f}%)]'.format(
                epoch, batch_idx * len(data_a), len(test_loader.dataset),
                       100. * batch_idx / len(test_loader)))

    geom_distances = np.vstack(geom_distances).reshape(-1,1)
    desc_distances = np.vstack(desc_distances).reshape(-1,1)
    print('\33[91mTest set: Geom MSE: {:.8f}\n\33[0m'.format(geom_distances.mean()))
    print('\33[91mTest set: Desc dist: {:.8f}\n\33[0m'.format(desc_distances.mean()))
    return

def adjust_learning_rate(optimizer):
    """Updates the learning rate given the learning rate decay.
    The routine has been implemented according to the original Lua SGD optimizer
    """
    for group in optimizer.param_groups:
        if 'step' not in group:
            group['step'] = 0.
        else:
            group['step'] += 1.
        group['lr'] = args.lr * (
        1.0 - float(group['step']) * float(args.batch_size) / (args.n_pairs * float(args.epochs)))
    return

def create_optimizer(model, new_lr):
    optimizer = optim.SGD(model.parameters(), lr=new_lr,
                          momentum=0.9, dampening=0.9,
                          weight_decay=args.wd)
    return optimizer

def main(train_loader, test_loader, model):
    # print the experiment configuration
    print('\nparsed options:\n{}\n'.format(vars(args)))
    if args.cuda:
        model.cuda()
    optimizer1 = create_optimizer(model, args.lr)
    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print('=> loading checkpoint {}'.format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            checkpoint = torch.load(args.resume)
            model.load_state_dict(checkpoint['state_dict'])
        else:
            print('=> no checkpoint found at {}'.format(args.resume))
    start = args.start_epoch
    end = start + args.epochs
    for epoch in range(start, end):
        # iterate over test loaders and test results
        train(train_loader, model, optimizer1, epoch)
        #test(test_loader, model, epoch)
    return 0
if __name__ == '__main__':
    LOG_DIR = args.log_dir
    LOG_DIR = os.path.join(args.log_dir,suffix)
    if not os.path.isdir(LOG_DIR):
        os.makedirs(LOG_DIR)
    from architectures import AffNetFast
    model = AffNetFast(PS=PS)
    train_loader, test_loader = create_loaders()
    main(train_loader, test_loader, model)
