import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import math
import numpy as np
from Utils import  GaussianBlur, CircularGaussKernel
from LAF import abc2A,rectifyAffineTransformationUpIsUp, sc_y_x2LAFs,sc_y_x_and_A2LAFs
from Utils import generate_2dgrid, generate_2dgrid, generate_3dgrid
from Utils import zero_response_at_border


class ScalePyramid(nn.Module):
    def __init__(self, nLevels = 3, init_sigma = 1.6, border = 5):
        super(ScalePyramid,self).__init__()
        self.nLevels = nLevels;
        self.init_sigma = init_sigma
        self.sigmaStep =  2 ** (1. / float(self.nLevels))
        #print 'step',self.sigmaStep
        self.b = border
        self.minSize = 2 * self.b + 2 + 1;
        return
    def forward(self,x):
        pixelDistance = 1.0;
        curSigma = 0.5
        if self.init_sigma > curSigma:
            sigma = np.sqrt(self.init_sigma**2 - curSigma**2)
            curSigma = self.init_sigma
            curr = GaussianBlur(sigma = sigma)(x)
        else:
            curr = x
        sigmas = [[curSigma]]
        pixel_dists = [[1.0]]
        pyr = [[curr]]
        j = 0
        while True:
            curr = pyr[-1][0]
            for i in range(1, self.nLevels + 2):
                sigma = curSigma * np.sqrt(self.sigmaStep*self.sigmaStep - 1.0 )
                #print 'blur sigma', sigma
                curr = GaussianBlur(sigma = sigma)(curr)
                curSigma *= self.sigmaStep
                pyr[j].append(curr)
                sigmas[j].append(curSigma)
                pixel_dists[j].append(pixelDistance)
                if i == self.nLevels:
                    nextOctaveFirstLevel = F.avg_pool2d(curr, kernel_size = 1, stride = 2, padding = 0) 
            pixelDistance = pixelDistance * 2.0
            curSigma = self.init_sigma
            if (nextOctaveFirstLevel[0,0,:,:].size(0)  <= self.minSize) or (nextOctaveFirstLevel[0,0,:,:].size(1) <= self.minSize):
                break
            pyr.append([nextOctaveFirstLevel])
            sigmas.append([curSigma])
            pixel_dists.append([pixelDistance])
            j+=1
        return pyr, sigmas, pixel_dists

class HessianResp(nn.Module):
    def __init__(self):
        super(HessianResp, self).__init__()
        
        self.gx =  nn.Conv2d(1, 1, kernel_size=(1,3), bias = False)
        self.gx.weight.data = torch.from_numpy(np.array([[[[0.5, 0, -0.5]]]], dtype=np.float32))

        self.gy =  nn.Conv2d(1, 1, kernel_size=(3,1), bias = False)
        self.gy.weight.data = torch.from_numpy(np.array([[[[0.5], [0], [-0.5]]]], dtype=np.float32))

        self.gxx =  nn.Conv2d(1, 1, kernel_size=(1,3),bias = False)
        self.gxx.weight.data = torch.from_numpy(np.array([[[[1.0, -2.0, 1.0]]]], dtype=np.float32))
        
        self.gyy =  nn.Conv2d(1, 1, kernel_size=(3,1), bias = False)
        self.gyy.weight.data = torch.from_numpy(np.array([[[[1.0], [-2.0], [1.0]]]], dtype=np.float32))
        return
    def forward(self, x, scale):
        gxx = self.gxx(F.pad(x, (1,1,0, 0), 'replicate'))
        gyy = self.gyy(F.pad(x, (0,0, 1,1), 'replicate'))
        gxy = self.gy(F.pad(self.gx(F.pad(x, (1,1,0, 0), 'replicate')), (0,0, 1,1), 'replicate'))
        return torch.abs(gxx * gyy - gxy * gxy) * (scale**4)


class AffineShapeEstimator(nn.Module):
    def __init__(self, threshold = 0.001, patch_size = 19):
        super(AffineShapeEstimator, self).__init__()
        self.threshold = threshold;
        self.PS = patch_size
        self.gx =  nn.Conv2d(1, 1, kernel_size=(1,3), bias = False)
        self.gx.weight.data = torch.from_numpy(np.array([[[[-1, 0, 1]]]], dtype=np.float32))
        self.gy =  nn.Conv2d(1, 1, kernel_size=(3,1), bias = False)
        self.gy.weight.data = torch.from_numpy(np.array([[[[-1], [0], [1]]]], dtype=np.float32))
        self.gk = torch.from_numpy(CircularGaussKernel(kernlen = self.PS, sigma = (self.PS / 2) /3.0).astype(np.float32))
        self.gk = Variable(self.gk, requires_grad=False)
        return
    def invSqrt(self,a,b,c):
        eps = 1e-12
        mask = (b != 0).float()
        r1 = mask * (c - a) / (2. * b + eps)
        t1 = torch.sign(r1) / (torch.abs(r1) + torch.sqrt(1. + r1*r1));
        r = 1.0 / torch.sqrt( 1. + t1*t1)
        t = t1*r;
        r = r * mask + 1.0 * (1.0 - mask);
        t = t * mask;
        
        x = 1. / torch.sqrt( r*r*a - 2.0*r*t*b + t*t*c)
        z = 1. / torch.sqrt( t*t*a + 2.0*r*t*b + r*r*c)
        
        d = torch.sqrt( x * z)
        
        x = x / d
        z = z / d
        
        l1 = torch.max(x,z)
        l2 = torch.min(x,z)
        
        new_a = r*r*x + t*t*z
        new_b = -r*t*x + t*r*z
        new_c = t*t*x + r*r *z

        return new_a, new_b, new_c, l1, l2
    def forward(self,x):
        if x.is_cuda:
            self.gk = self.gk.cuda()
        else:
            self.gk = self.gk.cpu()
        gx = self.gx(F.pad(x, (1, 1, 0, 0), 'replicate'))
        gy = self.gy(F.pad(x, (0, 0, 1, 1), 'replicate'))
        a1 = (gx * gx * self.gk.unsqueeze(0).unsqueeze(0).expand_as(gx)).view(x.size(0),-1).mean(dim=1)
        b1 = (gx * gy * self.gk.unsqueeze(0).unsqueeze(0).expand_as(gx)).view(x.size(0),-1).mean(dim=1)
        c1 = (gy * gy * self.gk.unsqueeze(0).unsqueeze(0).expand_as(gx)).view(x.size(0),-1).mean(dim=1)
        a, b, c, l1, l2 = self.invSqrt(a1,b1,c1)
        rat1 = l1/l2
        mask = (torch.abs(rat1) <= 6.).float().view(-1);
        return rectifyAffineTransformationUpIsUp(abc2A(a,b,c))#, mask
class OrientationDetector(nn.Module):
    def __init__(self,
                mrSize = 3.0, patch_size = None):
        super(OrientationDetector, self).__init__()
        if patch_size is None:
            patch_size = 32;
        self.PS = patch_size;
        self.bin_weight_kernel_size, self.bin_weight_stride = self.get_bin_weight_kernel_size_and_stride(self.PS, 1)
        self.mrSize = mrSize;
        self.num_ang_bins = 36
        self.gx =  nn.Conv2d(1, 1, kernel_size=(1,3),  bias = False)
        self.gx.weight.data = torch.from_numpy(np.array([[[[0.5, 0, -0.5]]]], dtype=np.float32))

        self.gy =  nn.Conv2d(1, 1, kernel_size=(3,1), bias = False)
        self.gy.weight.data = torch.from_numpy(np.array([[[[0.5], [0], [-0.5]]]], dtype=np.float32))

        self.angular_smooth =  nn.Conv1d(1, 1, kernel_size=3, padding = 1, bias = False)
        self.angular_smooth.weight.data = torch.from_numpy(np.array([[[0.33, 0.34, 0.33]]], dtype=np.float32))

        self.gk = 10. * torch.from_numpy(CircularGaussKernel(kernlen=self.PS).astype(np.float32))
        self.gk = Variable(self.gk, requires_grad=False)
        return
    def get_bin_weight_kernel_size_and_stride(self, patch_size, num_spatial_bins):
        bin_weight_stride = int(round(2.0 * np.floor(patch_size / 2) / float(num_spatial_bins + 1)))
        bin_weight_kernel_size = int(2 * bin_weight_stride - 1);
        return bin_weight_kernel_size, bin_weight_stride
    def get_rotation_matrix(self, angle_in_radians):
        angle_in_radians = angle_in_radians.view(-1, 1, 1);
        sin_a = torch.sin(angle_in_radians)
        cos_a = torch.cos(angle_in_radians)
        A1_x = torch.cat([cos_a, sin_a], dim = 2)
        A2_x = torch.cat([-sin_a, cos_a], dim = 2)
        transform = torch.cat([A1_x,A2_x], dim = 1)
        return transform

    def forward(self, x, return_rot_matrix = False):
        gx = self.gx(F.pad(x, (1,1,0, 0), 'replicate'))
        gy = self.gy(F.pad(x, (0,0, 1,1), 'replicate'))
        mag = torch.sqrt(gx * gx + gy * gy + 1e-10)
        if x.is_cuda:
            self.gk = self.gk.cuda()
        mag = mag * self.gk.unsqueeze(0).unsqueeze(0).expand_as(mag)
        ori = torch.atan2(gy,gx)
        o_big = float(self.num_ang_bins) *(ori + 1.0 * math.pi )/ (2.0 * math.pi)
        bo0_big =  torch.floor(o_big)
        wo1_big = o_big - bo0_big
        bo0_big =  bo0_big %  self.num_ang_bins
        bo1_big = (bo0_big + 1) % self.num_ang_bins
        wo0_big = (1.0 - wo1_big) * mag
        wo1_big = wo1_big * mag
        ang_bins = []
        for i in range(0, self.num_ang_bins):
            ang_bins.append(F.adaptive_avg_pool2d((bo0_big == i).float() * wo0_big, (1,1)))
        ang_bins = torch.cat(ang_bins,1).view(-1,1,self.num_ang_bins)
        ang_bins = self.angular_smooth(ang_bins)
        values, indices = ang_bins.view(-1,self.num_ang_bins).max(1)
        angle =  -((2. * float(np.pi) * indices.float() / float(self.num_ang_bins)) - float(math.pi))
        if return_rot_matrix:
            return self.get_rotation_matrix(angle)
        return angle
    
class NMS2d(nn.Module):
    def __init__(self, kernel_size = 3, threshold = 0):
        super(NMS2d, self).__init__()
        self.MP = nn.MaxPool2d(kernel_size, stride=1, return_indices=False, padding = kernel_size/2)
        self.eps = 1e-5
        self.th = threshold
        return
    def forward(self, x):
        #local_maxima = self.MP(x)
        if self.th > self.eps:
            return  x * (x > self.th).float() * ((x + self.eps - self.MP(x)) > 0).float()
        else:
            return ((x - self.MP(x) + self.eps) > 0).float() * x

class NMS3d(nn.Module):
    def __init__(self, kernel_size = 3, threshold = 0):
        super(NMS3d, self).__init__()
        self.MP = nn.MaxPool3d(kernel_size, stride=1, return_indices=False, padding = (0, kernel_size//2, kernel_size//2))
        self.eps = 1e-5
        self.th = threshold
        return
    def forward(self, x):
        #local_maxima = self.MP(x)
        if self.th > self.eps:
            return  x * (x > self.th).float() * ((x + self.eps - self.MP(x)) > 0).float()
        else:
            return ((x - self.MP(x) + self.eps) > 0).float() * x
        
class NMS3dAndComposeA(nn.Module):
    def __init__(self, w = 0, h = 0, kernel_size = 3, threshold = 0, scales = None, border = 3, mrSize = 1.0):
        super(NMS3dAndComposeA, self).__init__()
        self.eps = 1e-7
        self.ks = 3
        self.th = threshold
        self.cube_idxs = []
        self.border = border
        self.mrSize = mrSize
        self.beta = 1.0
        self.grid_ones = Variable(torch.ones(3,3,3,3), requires_grad=False)
        self.NMS3d = NMS3d(kernel_size, threshold)
        if (w > 0) and (h > 0):
            self.spatial_grid = generate_2dgrid(h, w, False).view(1, h, w,2).permute(3,1, 2, 0)
            self.spatial_grid = Variable(self.spatial_grid)
        else:
            self.spatial_grid = None
        return
    def forward(self, low, cur, high, num_features = 0, octaveMap = None, scales = None):
        assert low.size() == cur.size() == high.size()
        #Filter responce map
        self.is_cuda = low.is_cuda;
        resp3d = torch.cat([low,cur,high], dim = 1)
        
        mrSize_border = int(self.mrSize);
        if octaveMap is not None:
            nmsed_resp = zero_response_at_border(self.NMS3d(resp3d.unsqueeze(1)).squeeze(1)[:,1:2,:,:], mrSize_border) * (1. - octaveMap.float())
        else:
            nmsed_resp = zero_response_at_border(self.NMS3d(resp3d.unsqueeze(1)).squeeze(1)[:,1:2,:,:], mrSize_border)
        
        num_of_nonzero_responces = (nmsed_resp > 0).float().sum().item()#data[0]
        if (num_of_nonzero_responces <= 1):
            return None,None,None
        if octaveMap is not None:
            octaveMap = (octaveMap.float() + nmsed_resp.float()).byte()
        
        nmsed_resp = nmsed_resp.view(-1)
        if (num_features > 0) and (num_features < num_of_nonzero_responces):
            nmsed_resp, idxs = torch.topk(nmsed_resp, k = num_features, dim = 0);
        else:
            idxs = nmsed_resp.data.nonzero().squeeze()
            nmsed_resp = nmsed_resp[idxs]
        #Get point coordinates grid
        
        if type(scales) is not list:
            self.grid = generate_3dgrid(3,self.ks,self.ks)
        else:
            self.grid = generate_3dgrid(scales,self.ks,self.ks)
        self.grid = Variable(self.grid.t().contiguous().view(3,3,3,3), requires_grad=False)
        if self.spatial_grid is None:
            self.spatial_grid = generate_2dgrid(low.size(2), low.size(3), False).view(1, low.size(2), low.size(3),2).permute(3,1, 2, 0)
            self.spatial_grid = Variable(self.spatial_grid)
        if self.is_cuda:
            self.spatial_grid = self.spatial_grid.cuda()
            self.grid_ones = self.grid_ones.cuda()
            self.grid = self.grid.cuda()
        #residual_to_patch_center
        sc_y_x = F.conv2d(resp3d, self.grid,
                                padding = 1) / (F.conv2d(resp3d, self.grid_ones, padding = 1) + 1e-8)
        
        ##maxima coords
        sc_y_x[0,1:,:,:] = sc_y_x[0,1:,:,:] + self.spatial_grid[:,:,:,0]
        sc_y_x = sc_y_x.view(3,-1).t()
        sc_y_x = sc_y_x[idxs,:]
        
        min_size = float(min((cur.size(2)), cur.size(3)))
        sc_y_x[:,0] = sc_y_x[:,0] / min_size
        sc_y_x[:,1] = sc_y_x[:,1] / float(cur.size(2))
        sc_y_x[:,2] = sc_y_x[:,2] / float(cur.size(3))
        return nmsed_resp, sc_y_x2LAFs(sc_y_x), octaveMap
class NMS3dAndComposeAAff(nn.Module):
    def __init__(self, w = 0, h = 0, kernel_size = 3, threshold = 0, scales = None, border = 3, mrSize = 1.0):
        super(NMS3dAndComposeAAff, self).__init__()
        self.eps = 1e-7
        self.ks = 3
        self.th = threshold
        self.cube_idxs = []
        self.border = border
        self.mrSize = mrSize
        self.beta = 1.0
        self.grid_ones = Variable(torch.ones(3,3,3,3), requires_grad=False)
        self.NMS3d = NMS3d(kernel_size, threshold)
        if (w > 0) and (h > 0):
            self.spatial_grid = generate_2dgrid(h, w, False).view(1, h, w,2).permute(3,1, 2, 0)
            self.spatial_grid = Variable(self.spatial_grid)
        else:
            self.spatial_grid = None
        return
    def forward(self, low, cur, high, num_features = 0, octaveMap = None, scales = None, aff_resp = None):
        assert low.size() == cur.size() == high.size()
        #Filter responce map
        self.is_cuda = low.is_cuda;
        resp3d = torch.cat([low,cur,high], dim = 1)
        
        mrSize_border = int(self.mrSize);
        if octaveMap is not None:
            nmsed_resp = zero_response_at_border(self.NMS3d(resp3d.unsqueeze(1)).squeeze(1)[:,1:2,:,:], mrSize_border) * (1. - octaveMap.float())
        else:
            nmsed_resp = zero_response_at_border(self.NMS3d(resp3d.unsqueeze(1)).squeeze(1)[:,1:2,:,:], mrSize_border)
        
        num_of_nonzero_responces = (nmsed_resp > 0).float().sum().item()#data[0]
        if (num_of_nonzero_responces <= 1):
            return None,None,None
        if octaveMap is not None:
            octaveMap = (octaveMap.float() + nmsed_resp.float()).byte()
        
        nmsed_resp = nmsed_resp.view(-1)
        if (num_features > 0) and (num_features < num_of_nonzero_responces):
            nmsed_resp, idxs = torch.topk(nmsed_resp, k = num_features, dim = 0);
        else:
            idxs = nmsed_resp.data.nonzero().squeeze()
            nmsed_resp = nmsed_resp[idxs]
        #Get point coordinates grid
        if type(scales) is not list:
            self.grid = generate_3dgrid(3,self.ks,self.ks)
        else:
            self.grid = generate_3dgrid(scales,self.ks,self.ks)
        self.grid = Variable(self.grid.t().contiguous().view(3,3,3,3), requires_grad=False)
        if self.spatial_grid is None:
            self.spatial_grid = generate_2dgrid(low.size(2), low.size(3), False).view(1, low.size(2), low.size(3),2).permute(3,1, 2, 0)
            self.spatial_grid = Variable(self.spatial_grid)
        if self.is_cuda:
            self.spatial_grid = self.spatial_grid.cuda()
            self.grid_ones = self.grid_ones.cuda()
            self.grid = self.grid.cuda()
        
        #residual_to_patch_center
        sc_y_x = F.conv2d(resp3d, self.grid,
                                padding = 1) / (F.conv2d(resp3d, self.grid_ones, padding = 1) + 1e-8)
        
        ##maxima coords
        sc_y_x[0,1:,:,:] = sc_y_x[0,1:,:,:] + self.spatial_grid[:,:,:,0]
        sc_y_x = sc_y_x.view(3,-1).t()
        sc_y_x = sc_y_x[idxs,:]
        if aff_resp is not None:
            A_matrices = aff_resp.view(4,-1).t()[idxs,:]        
        min_size = float(min((cur.size(2)), cur.size(3)))
        
        sc_y_x[:,0] = sc_y_x[:,0] / min_size
        sc_y_x[:,1] = sc_y_x[:,1] / float(cur.size(2))
        sc_y_x[:,2] = sc_y_x[:,2] / float(cur.size(3))
        return nmsed_resp, sc_y_x_and_A2LAFs(sc_y_x,A_matrices), octaveMap
