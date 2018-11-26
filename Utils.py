import torch
import torch.nn.init
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import cv2
import numpy as np

# resize image to size 32x32
cv2_scale = lambda x: cv2.resize(x, dsize=(32, 32),
                                 interpolation=cv2.INTER_LINEAR)
# reshape image
np_reshape32 = lambda x: np.reshape(x, (32, 32, 1))
np_reshape64 = lambda x: np.reshape(x, (64, 64, 1))

def zeros_like(x):
    assert x.__class__.__name__.find('Variable') != -1 or x.__class__.__name__.find('Tensor') != -1, "Object is neither a Tensor nor a Variable"
    y = torch.zeros(x.size())
    if x.is_cuda:
       y = y.cuda()
    if x.__class__.__name__ == 'Variable':
        return torch.autograd.Variable(y, requires_grad=x.requires_grad)
    elif x.__class__.__name__.find('Tensor') != -1:
        return torch.zeros(y)

def ones_like(x):
    assert x.__class__.__name__.find('Variable') != -1 or x.__class__.__name__.find('Tensor') != -1, "Object is neither a Tensor nor a Variable"
    y = torch.ones(x.size())
    if x.is_cuda:
       y = y.cuda()
    if x.__class__.__name__ == 'Variable':
        return torch.autograd.Variable(y, requires_grad=x.requires_grad)
    elif x.__class__.__name__.find('Tensor') != -1:
        return torch.ones(y)
    

def batched_forward(model, data, batch_size, **kwargs):
    n_patches = len(data)
    if n_patches > batch_size:
        bs = batch_size
        n_batches = int(n_patches / bs + 1)
        for batch_idx in range(n_batches):
            st = batch_idx * bs
            if batch_idx == n_batches - 1:
                if (batch_idx + 1) * bs > n_patches:
                    end = n_patches
                else:
                    end = (batch_idx + 1) * bs
            else:
                end = (batch_idx + 1) * bs
            if st >= end:
                continue
            if batch_idx == 0:
                first_batch_out = model(data[st:end], kwargs)
                out_size = torch.Size([n_patches] + list(first_batch_out.size()[1:]))
                #out_size[0] = n_patches
                out = torch.zeros(out_size);
                if data.is_cuda:
                    out = out.cuda()
                out = Variable(out)
                out[st:end] = first_batch_out
            else:
                out[st:end,:,:] = model(data[st:end], kwargs)
        return out
    else:
        return model(data, kwargs)

class L2Norm(nn.Module):
    def __init__(self):
        super(L2Norm,self).__init__()
        self.eps = 1e-10
    def forward(self, x):
        norm = torch.sqrt(torch.sum(x * x, dim = 1) + self.eps)
        x= x / norm.unsqueeze(-1).expand_as(x)
        return x

class L1Norm(nn.Module):
    def __init__(self):
        super(L1Norm,self).__init__()
        self.eps = 1e-10
    def forward(self, x):
        norm = torch.sum(torch.abs(x), dim = 1) + self.eps
        x= x / norm.expand_as(x)
        return x

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False

def CircularGaussKernel(kernlen=None, circ_zeros = False, sigma = None, norm = True):
    assert ((kernlen is not None) or sigma is not None)
    if kernlen is None:
        kernlen = int(2.0 * 3.0 * sigma + 1.0)
        if (kernlen % 2 == 0):
            kernlen = kernlen + 1;
        halfSize = kernlen / 2;
    halfSize = kernlen / 2;
    r2 = float(halfSize*halfSize)
    if sigma is None:
        sigma2 = 0.9 * r2;
        sigma = np.sqrt(sigma2)
    else:
        sigma2 = 2.0 * sigma * sigma    
    x = np.linspace(-halfSize,halfSize,kernlen)
    xv, yv = np.meshgrid(x, x, sparse=False, indexing='xy')
    distsq = (xv)**2 + (yv)**2
    kernel = np.exp(-( distsq/ (sigma2)))
    if circ_zeros:
        kernel *= (distsq <= r2).astype(np.float32)
    if norm:
        kernel /= np.sum(kernel)
    return kernel

def generate_2dgrid(h,w, centered = True):
    if centered:
        x = torch.linspace(-w/2+1, w/2, w)
        y = torch.linspace(-h/2+1, h/2, h)
    else:
        x = torch.linspace(0, w-1, w)
        y = torch.linspace(0, h-1, h)
    grid2d = torch.stack([y.repeat(w,1).t().contiguous().view(-1), x.repeat(h)],1)
    return grid2d

def generate_3dgrid(d, h, w, centered = True):
    if type(d) is not list:
        if centered:
            z = torch.linspace(-d/2+1, d/2, d)
        else:
            z = torch.linspace(0, d-1, d)
        dl = d
    else:
        z = torch.FloatTensor(d)
        dl = len(d)
    grid2d = generate_2dgrid(h,w, centered = centered)
    grid3d = torch.cat([z.repeat(w*h,1).t().contiguous().view(-1,1), grid2d.repeat(dl,1)],dim = 1)
    return grid3d

def zero_response_at_border(x, b):
    if (b < x.size(3)) and (b < x.size(2)):
        x[:, :,  0:b, :] =  0
        x[:, :,  x.size(2) - b: , :] =  0
        x[:, :, :,  0:b] =  0
        x[:, :, :,   x.size(3) - b: ] =  0
    else:
        return x * 0
    return x

class GaussianBlur(nn.Module):
    def __init__(self, sigma=1.6):
        super(GaussianBlur, self).__init__()
        weight = self.calculate_weights(sigma)
        self.register_buffer('buf', weight)
        return
    def calculate_weights(self,  sigma):
        kernel = CircularGaussKernel(sigma = sigma, circ_zeros = False)
        h,w = kernel.shape
        halfSize = float(h) / 2.;
        self.pad = int(np.floor(halfSize))
        return torch.from_numpy(kernel.astype(np.float32)).view(1,1,h,w);
    def forward(self, x):
        w = Variable(self.buf)
        if x.is_cuda:
            w = w.cuda()
        return F.conv2d(F.pad(x, (self.pad,self.pad,self.pad,self.pad), 'replicate'), w, padding = 0)

def batch_eig2x2(A):
    trace = A[:,0,0] + A[:,1,1]
    delta1 = (trace*trace - 4 * ( A[:,0,0]*  A[:,1,1] -  A[:,1,0]* A[:,0,1]))
    mask = delta1 > 0
    delta = torch.sqrt(torch.abs(delta1))
    l1 = mask.float() * (trace + delta) / 2.0 +  1000.  * (1.0 - mask.float())
    l2 = mask.float() * (trace - delta) / 2.0 +  0.0001  * (1.0 - mask.float())
    return l1,l2

def line_prepender(filename, line):
    with open(filename, 'r+') as f:
        content = f.read()
        f.seek(0, 0)
        f.write(line.rstrip('\r\n') + '\n' + content)
    return
