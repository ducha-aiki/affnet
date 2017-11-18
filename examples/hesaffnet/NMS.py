import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable
from Utils import CircularGaussKernel, generate_2dgrid, generate_2dgrid, generate_3dgrid, zero_response_at_border
from LAF import sc_y_x2LAFs
 
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
        self.MP = nn.MaxPool3d(kernel_size, stride=1, return_indices=False, padding = (0, kernel_size/2, kernel_size/2))
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
    def __init__(self,kernel_size = 3, threshold = 0, scales = None, border = 3, mrSize = 1.0):
        super(NMS3dAndComposeA, self).__init__()
        self.eps = 1e-7
        self.ks = 3
        if type(scales) is not list:
            self.grid = generate_3dgrid(3,self.ks,self.ks)
        else:
            self.grid = generate_3dgrid(scales,self.ks,self.ks)
        self.grid = Variable(self.grid.t().contiguous().view(3,3,3,3), requires_grad=False)
        self.th = threshold
        self.cube_idxs = []
        self.border = border
        self.mrSize = mrSize
        self.beta = 1.0
        self.grid_ones = Variable(torch.ones(3,3,3,3), requires_grad=False)
        self.NMS3d = NMS3d(kernel_size, threshold)
        return
    def forward(self, low, cur, high, octaveMap = None, num_features = 0):
        assert low.size() == cur.size() == high.size()
        
        #Filter responce map
        self.is_cuda = low.is_cuda;
        resp3d = torch.cat([low,cur,high], dim = 1)
        
        mrSize_border = int(self.mrSize);
        if octaveMap is not None:
            nmsed_resp = zero_response_at_border(self.NMS3d(resp3d.unsqueeze(1)).squeeze(1)[:,1:2,:,:], mrSize_border) * (1. - octaveMap.float())
        else:
            nmsed_resp = zero_response_at_border(self.NMS3d(resp3d.unsqueeze(1)).squeeze(1)[:,1:2,:,:], mrSize_border)
        
        num_of_nonzero_responces = (nmsed_resp > 0).sum().data[0]
        if (num_of_nonzero_responces == 0):
            return None,None,None
        if octaveMap is not None:
            octaveMap = (octaveMap.float() + nmsed_resp.float()).byte()
        
        nmsed_resp = nmsed_resp.view(-1)
        if (num_features > 0) and (num_features < num_of_nonzero_responces):
            nmsed_resp, idxs = torch.topk(nmsed_resp, k = num_features);
        else:
            idxs = nmsed_resp.data.nonzero().squeeze()
            nmsed_resp = nmsed_resp[idxs]
        
        #Get point coordinates
        
        spatial_grid = Variable(generate_2dgrid(low.size(2), low.size(3), False)).view(1,low.size(2), low.size(3),2)
        spatial_grid = spatial_grid.permute(3,1, 2, 0)
        if self.is_cuda:
            spatial_grid = spatial_grid.cuda()
            self.grid = self.grid.cuda()
            self.grid_ones = self.grid_ones.cuda()
        #residual_to_patch_center
        sc_y_x = F.conv2d(resp3d, self.grid,
                                padding = 1) / (F.conv2d(resp3d, self.grid_ones, padding = 1) + 1e-8)
        
        ##maxima coords
        sc_y_x[0,1:,:,:] = sc_y_x[0,1:,:,:] + spatial_grid[:,:,:,0]
        sc_y_x = sc_y_x.view(3,-1).t()
        sc_y_x = sc_y_x[idxs,:]
        
        min_size = float(min((cur.size(2)), cur.size(3)))
        sc_y_x[:,0] = sc_y_x[:,0] / min_size
        sc_y_x[:,1] = sc_y_x[:,1] / float(cur.size(2))
        sc_y_x[:,2] = sc_y_x[:,2] / float(cur.size(3))
        
        return nmsed_resp, sc_y_x2LAFs(sc_y_x), octaveMap