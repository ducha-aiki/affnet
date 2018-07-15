#from __future__ import division, print_function
import matplotlib
matplotlib.use('Agg')
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
import gc
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
from HardNet import HardNet, L2Norm
from Losses import loss_HardNetDetach, loss_HardNet
from SparseImgRepresenter import ScaleSpaceAffinePatchExtractor
from LAF import denormalizeLAFs, LAFs2ell, abc2A,visualize_LAFs
import seaborn as sns
from Losses import distance_matrix_vector
from ReprojectionStuff import get_GT_correspondence_indexes

PS = 32
tilt_schedule = {'0': 3.0, '1': 4.0, '3': 4.5, '5': 4.8, '6': 5.2, '8':  5.8 }

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
parser.add_argument('--lr', type=float, default=0.005, metavar='LR',
                    help='learning rate (default: 0.005)')
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
                    help='which descriptor distance is minimized. Variants: pixels, SIFT, HardNet')
parser.add_argument('--loss', type=str,
                    default='HardNet',
                    help='Variants: HardNet, HardNetDetach, PosDist, Geom')
parser.add_argument('--arch', type=str,
                    default='AffNetFast',
                    help='Variants: AffNetFast, AffNetSlow, AffNetFast4, AffNetFast4Rot')


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
random.seed(args.seed)
class HardTFeatNet(nn.Module):
    """TFeat model definition
    """

    def __init__(self, sm):
        super(HardTFeatNet, self).__init__()

        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=7),
            nn.Tanh(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=6),
            nn.Tanh()
        )
        self.classifier = nn.Sequential(
            nn.Dropout(0.1),
            nn.Conv2d(64, 128, kernel_size=8),
            nn.Tanh())
        self.SIFT = sm
    def input_norm(self,x):
        flat = x.view(x.size(0), -1)
        mp = torch.mean(flat, dim=1)
        sp = torch.std(flat, dim=1) + 1e-7
        return (x - mp.detach().unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).expand_as(x)) / sp.detach().unsqueeze(-1).unsqueeze(-1).unsqueeze(1).expand_as(x)
    
    def forward(self, input):
        x_features = self.features(self.input_norm(input))
        x = x_features.view(x_features.size(0), -1)
        x = self.classifier(x_features)
        return x.view(x.size(0), -1)

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
elif args.descriptor == 'TFeat':
    descriptor = HardTFeatNet(sm=SIFTNet(patch_size = 32))
    if not args.no_cuda:
        descriptor = descriptor.cuda()
    model_weights = 'HardTFeat.pth'
    hncheckpoint = torch.load(model_weights)
    descriptor.load_state_dict(hncheckpoint['state_dict'])
    descriptor.train()
else:
    descriptor = lambda x: L2Norm()(x.view(x.size(0),-1) - x.view(x.size(0),-1).mean(dim=1, keepdim=True).expand(x.size(0),x.size(1)*x.size(2)*x.size(3)).detach())

suffix = args.expname +'_OriNet_6Brown_' +  args.descriptor + '_' + str(args.lr) + '_' + str(args.n_pairs) + "_" + str(args.loss) 
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

    #test_loader = torch.utils.data.DataLoader(
    #        HPatchesDM('dataset/HP_HessianPatches/','', train=False,
    #                         n_pairs = args.n_test_pairs,
    #                         batch_size=args.test_batch_size,
    #                         download=True,
    #                         transform=transforms.Compose([])),
    #                         batch_size=args.test_batch_size,
    #                         shuffle=False, **kwargs)
    return train_loader, None

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
    
def extract_random_LAF(data, max_rot = math.pi, max_tilt = 1.0, crop_size = 32):
    st = int((data.size(2) - crop_size)/2)
    fin = st + crop_size
    if type(max_rot) is float:
        rot_LAFs, inv_rotmat = get_random_rotation_LAFs(data, max_rot)
    else:
        rot_LAFs = max_rot
        inv_rotmat = None
    aff_LAFs, inv_TA = get_random_norm_affine_LAFs(data, max_tilt);
    aff_LAFs[:,0:2,0:2] = torch.bmm(rot_LAFs[:,0:2,0:2],aff_LAFs[:,0:2,0:2])
    data_aff = extract_patches(data,  aff_LAFs, PS = data.size(2))
    data_affcrop = data_aff[:,:, st:fin, st:fin].contiguous()
    return data_affcrop, data_aff, rot_LAFs,inv_rotmat,inv_TA 
def train(train_loader, model, optimizer, epoch):
    # switch to train mode
    model.train()
    pbar = tqdm(enumerate(train_loader))
    for batch_idx, data in pbar:
        data_a, data_p = data
        if args.cuda:
            data_a, data_p  = data_a.float().cuda(), data_p.float().cuda()
            data_a, data_p = Variable(data_a), Variable(data_p)
        rot_LAFs, inv_rotmat = get_random_rotation_LAFs(data_a, math.pi)
        scale = Variable( 0.9 + 0.3* torch.rand(data_a.size(0), 1, 1));
        if args.cuda:
            scale = scale.cuda()
        rot_LAFs[:,0:2,0:2] = rot_LAFs[:,0:2,0:2] * scale.expand(data_a.size(0),2,2)
        shift_w, shift_h = get_random_shifts_LAFs(data_a, 2, 2)
        rot_LAFs[:,0,2] = rot_LAFs[:,0,2] + shift_w / float(data_a.size(3))
        rot_LAFs[:,1,2] = rot_LAFs[:,1,2] + shift_h / float(data_a.size(2))
        data_a_rot = extract_patches(data_a,  rot_LAFs, PS = data_a.size(2))
        st = int((data_p.size(2) - model.PS)/2)
        fin = st + model.PS

        data_p_crop = data_p[:,:, st:fin, st:fin].contiguous()
        data_a_rot_crop = data_a_rot[:,:, st:fin, st:fin].contiguous()
        out_a_rot, out_p, out_a = model(data_a_rot_crop,True), model(data_p_crop,True), model(data_a[:,:, st:fin, st:fin].contiguous(), True)
        out_p_rotatad = torch.bmm(inv_rotmat, out_p)

        ######Apply rot and get sifts
        out_patches_a_crop = extract_and_crop_patches_by_predicted_transform(data_a_rot, out_a_rot, crop_size = model.PS)
        out_patches_p_crop = extract_and_crop_patches_by_predicted_transform(data_p, out_p, crop_size = model.PS)

        desc_a = descriptor(out_patches_a_crop)
        desc_p = descriptor(out_patches_p_crop)
        descr_dist =  torch.sqrt(((desc_a - desc_p)**2).view(data_a.size(0),-1).sum(dim=1) + 1e-6).mean()
        geom_dist = torch.sqrt(((out_a_rot - out_p_rotatad)**2 ).view(-1,4).sum(dim=1)[0] + 1e-8).mean()
        if args.loss == 'HardNet':
            loss = loss_HardNet(desc_a,desc_p); 
        elif args.loss == 'HardNetDetach':
            loss = loss_HardNetDetach(desc_a,desc_p); 
        elif args.loss == 'Geom':
            loss = geom_dist; 
        elif args.loss == 'PosDist':
            loss = descr_dist; 
        else:
            print('Unknown loss function')
            sys.exit(1)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        adjust_learning_rate(optimizer)
        if batch_idx % args.log_interval == 0:
            pbar.set_description(
                'Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.4f}, {:.4f},{:.4f}'.format(
                    epoch, batch_idx * len(data_a), len(train_loader.dataset),
                           100. * batch_idx / len(train_loader),
                    float(loss.detach().cpu().numpy()), float(geom_dist.detach().cpu().numpy()), float(descr_dist.detach().cpu().numpy())))
    torch.save({'epoch': epoch + 1, 'state_dict': model.state_dict()},
               '{}/checkpoint_{}.pth'.format(LOG_DIR,epoch))
def load_grayscale_var(fname):
    img = Image.open(fname).convert('RGB')
    img = np.mean(np.array(img), axis = 2)
    var_image = torch.autograd.Variable(torch.from_numpy(img.astype(np.float32)), volatile = True)
    var_image_reshape = var_image.view(1, 1, var_image.size(0),var_image.size(1))
    if args.cuda:
        var_image_reshape = var_image_reshape.cuda()
    return var_image_reshape
def get_geometry_and_descriptors(img, det, desc, do_ori = True):
    with torch.no_grad():
        LAFs, resp = det(img,do_ori = do_ori)
        patches = det.extract_patches_from_pyr(LAFs, PS = 32)
        descriptors = desc(patches)
    return LAFs, descriptors

def test(model,epoch):
    torch.cuda.empty_cache()
    # switch to evaluate mode
    model.eval()
    from architectures import AffNetFast
    affnet = AffNetFast()
    model_weights = 'pretrained/AffNet.pth'
    hncheckpoint = torch.load(model_weights)
    affnet.load_state_dict(hncheckpoint['state_dict'])
    affnet.eval()
    detector = ScaleSpaceAffinePatchExtractor( mrSize = 5.192, num_features = 3000,
                                          border = 5, num_Baum_iters = 1, 
                                          AffNet = affnet, OriNet = model)
    descriptor = HardNet()
    model_weights = 'HardNet++.pth'
    hncheckpoint = torch.load(model_weights)
    descriptor.load_state_dict(hncheckpoint['state_dict'])
    descriptor.eval()
    if args.cuda:
        detector = detector.cuda()
        descriptor = descriptor.cuda()
    input_img_fname1 = 'test-graf/img1.png'#sys.argv[1]
    input_img_fname2 = 'test-graf/img6.png'#sys.argv[1]
    H_fname = 'test-graf/H1to6p'#sys.argv[1]
    output_img_fname = 'graf_match.png'#sys.argv[3]
    img1 = load_grayscale_var(input_img_fname1)
    img2 = load_grayscale_var(input_img_fname2)
    H = np.loadtxt(H_fname)    
    H1to2 = Variable(torch.from_numpy(H).float())
    SNN_threshold = 0.8
    with torch.no_grad():
        LAFs1, descriptors1 = get_geometry_and_descriptors(img1, detector, descriptor)
        torch.cuda.empty_cache()
        LAFs2, descriptors2 = get_geometry_and_descriptors(img2, detector, descriptor)
        visualize_LAFs(img1.detach().cpu().numpy().squeeze(), LAFs1.detach().cpu().numpy().squeeze(), 'b', show = False, save_to = LOG_DIR + "/detections1_" + str(epoch) + '.png')
        visualize_LAFs(img2.detach().cpu().numpy().squeeze(), LAFs2.detach().cpu().numpy().squeeze(), 'g', show = False, save_to = LOG_DIR + "/detection2_" + str(epoch) + '.png')
        dist_matrix = distance_matrix_vector(descriptors1, descriptors2)
        min_dist, idxs_in_2 = torch.min(dist_matrix,1)
        dist_matrix[:,idxs_in_2] = 100000;# mask out nearest neighbour to find second nearest
        min_2nd_dist, idxs_2nd_in_2 = torch.min(dist_matrix,1)
        mask = (min_dist / (min_2nd_dist + 1e-8)) <= SNN_threshold
        tent_matches_in_1 = indxs_in1 = torch.autograd.Variable(torch.arange(0, idxs_in_2.size(0)), requires_grad = False).cuda()[mask]
        tent_matches_in_2 = idxs_in_2[mask]
        tent_matches_in_1 = tent_matches_in_1.long()
        tent_matches_in_2 = tent_matches_in_2.long()
        LAF1s_tent = LAFs1[tent_matches_in_1,:,:]
        LAF2s_tent = LAFs2[tent_matches_in_2,:,:]
        min_dist, plain_indxs_in1, idxs_in_2 = get_GT_correspondence_indexes(LAF1s_tent, LAF2s_tent,H1to2.cuda(), dist_threshold = 6) 
        plain_indxs_in1 = plain_indxs_in1.long()
        inl_ratio = float(plain_indxs_in1.size(0)) / float(tent_matches_in_1.size(0))
        print 'Test epoch', str(epoch) 
        print 'Test on graf1-6,', tent_matches_in_1.size(0), 'tentatives', plain_indxs_in1.size(0), 'true matches', str(inl_ratio)[:5], ' inl.ratio'
        visualize_LAFs(img1.detach().cpu().numpy().squeeze(), LAF1s_tent[plain_indxs_in1.long(),:,:].detach().cpu().numpy().squeeze(), 'g', show = False, save_to = LOG_DIR + "/inliers1_" + str(epoch) + '.png')
        visualize_LAFs(img2.detach().cpu().numpy().squeeze(), LAF2s_tent[idxs_in_2.long(),:,:].detach().cpu().numpy().squeeze(), 'g', show = False, save_to = LOG_DIR + "/inliers2_" + str(epoch) + '.png')
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
    test(model, -1)
    for epoch in range(start, end):
        # iterate over test loaders and test results
        train(train_loader, model, optimizer1, epoch)
        test(model, epoch)
    return 0
if __name__ == '__main__':
    LOG_DIR = args.log_dir
    LOG_DIR = os.path.join(args.log_dir,suffix)
    if not os.path.isdir(LOG_DIR):
        os.makedirs(LOG_DIR)
    from architectures import OriNetFast
    model = OriNetFast(PS=32)
    train_loader, test_loader = create_loaders()
    main(train_loader, test_loader, model)

