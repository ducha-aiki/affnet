import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy
from scipy.spatial.distance import cdist
from numpy.linalg import inv    
from scipy.linalg import schur, sqrtm
import torch
from  torch.autograd import Variable

##########numpy
def invSqrt(a,b,c):
    eps = 1e-12 
    mask = (b !=  0)
    r1 = mask * (c - a) / (2. * b + eps)
    t1 = np.sign(r1) / (np.abs(r1) + np.sqrt(1. + r1*r1));
    r = 1.0 / np.sqrt( 1. + t1*t1)
    t = t1*r;
    
    r = r * mask + 1.0 * (1.0 - mask);
    t = t * mask;
    
    x = 1. / np.sqrt( r*r*a - 2*r*t*b + t*t*c)
    z = 1. / np.sqrt( t*t*a + 2*r*t*b + r*r*c)
    
    d = np.sqrt( x * z)
    
    x = x / d
    z = z / d
       
    new_a = r*r*x + t*t*z
    new_b = -r*t*x + t*r*z
    new_c = t*t*x + r*r *z

    return new_a, new_b, new_c

def Ell2LAF(ell):
    A23 = np.zeros((2,3))
    A23[0,2] = ell[0]
    A23[1,2] = ell[1]
    a = ell[2]
    b = ell[3]
    c = ell[4]
    sc = np.sqrt(np.sqrt(a*c - b*b))
    ia,ib,ic = invSqrt(a,b,c)  #because sqrtm returns ::-1, ::-1 matrix, don`t know why 
    A = np.array([[ia, ib], [ib, ic]]) / sc
    sc = np.sqrt(A[0,0] * A[1,1] - A[1,0] * A[0,1])
    A23[0:2,0:2] = rectifyAffineTransformationUpIsUp(A / sc) * sc
    return A23

def rectifyAffineTransformationUpIsUp_np(A):
    det = np.sqrt(np.abs(A[0,0]*A[1,1] - A[1,0]*A[0,1] + 1e-10))
    b2a2 = np.sqrt(A[0,1] * A[0,1] + A[0,0] * A[0,0])
    A_new = np.zeros((2,2))
    A_new[0,0] = b2a2 / det
    A_new[0,1] = 0
    A_new[1,0] = (A[1,1]*A[0,1]+A[1,0]*A[0,0])/(b2a2*det)
    A_new[1,1] = det / b2a2
    return A_new

def ells2LAFs(ells):
    LAFs = np.zeros((len(ells), 2,3))
    for i in range(len(ells)):
        LAFs[i,:,:] = Ell2LAF(ells[i,:])
    return LAFs

def LAF2pts(LAF, n_pts = 50):
    a = np.linspace(0, 2*np.pi, n_pts);
    x = [0]
    x.extend(list(np.sin(a)))
    x = np.array(x).reshape(1,-1)
    y = [0]
    y.extend(list(np.cos(a)))
    y = np.array(y).reshape(1,-1)
    HLAF = np.concatenate([LAF, np.array([0,0,1]).reshape(1,3)])
    H_pts =np.concatenate([x,y,np.ones(x.shape)])
    H_pts_out = np.transpose(np.matmul(HLAF, H_pts))
    H_pts_out[:,0] = H_pts_out[:,0] / H_pts_out[:, 2]
    H_pts_out[:,1] = H_pts_out[:,1] / H_pts_out[:, 2]
    return H_pts_out[:,0:2]


def convertLAFs_to_A23format(LAFs):
    sh = LAFs.shape
    if (len(sh) == 3) and (sh[1]  == 2) and (sh[2] == 3): # n x 2 x 3 classical [A, (x;y)] matrix
        work_LAFs = deepcopy(LAFs)
    elif (len(sh) == 2) and (sh[1]  == 7): #flat format, x y scale a11 a12 a21 a22
        work_LAFs = np.zeros((sh[0], 2,3))
        work_LAFs[:,0,2] = LAFs[:,0]
        work_LAFs[:,1,2] = LAFs[:,1]
        work_LAFs[:,0,0] = LAFs[:,2] * LAFs[:,3] 
        work_LAFs[:,0,1] = LAFs[:,2] * LAFs[:,4]
        work_LAFs[:,1,0] = LAFs[:,2] * LAFs[:,5]
        work_LAFs[:,1,1] = LAFs[:,2] * LAFs[:,6]
    elif (len(sh) == 2) and (sh[1]  == 6): #flat format, x y s*a11 s*a12 s*a21 s*a22
        work_LAFs = np.zeros((sh[0], 2,3))
        work_LAFs[:,0,2] = LAFs[:,0]
        work_LAFs[:,1,2] = LAFs[:,1]
        work_LAFs[:,0,0] = LAFs[:,2] 
        work_LAFs[:,0,1] = LAFs[:,3]
        work_LAFs[:,1,0] = LAFs[:,4]
        work_LAFs[:,1,1] = LAFs[:,5]
    else:
        print 'Unknown LAF format'
        return None
    return work_LAFs

def LAFs2ell(in_LAFs):
    LAFs = convertLAFs_to_A23format(in_LAFs)
    ellipses = np.zeros((len(LAFs),5))
    for i in range(len(LAFs)):
        LAF = deepcopy(LAFs[i,:,:])
        scale = np.sqrt(LAF[0,0]*LAF[1,1]  - LAF[0,1]*LAF[1, 0] + 1e-10)
        u, W, v = np.linalg.svd(LAF[0:2,0:2] / scale, full_matrices=True)
        W[0] = 1. / (W[0]*W[0]*scale*scale)
        W[1] = 1. / (W[1]*W[1]*scale*scale)
        A =  np.matmul(np.matmul(u, np.diag(W)), u.transpose())
        ellipses[i,0] = LAF[0,2]
        ellipses[i,1] = LAF[1,2]
        ellipses[i,2] = A[0,0]
        ellipses[i,3] = A[0,1]
        ellipses[i,4] = A[1,1]
    return ellipses

def visualize_LAFs(img, LAFs):
    work_LAFs = convertLAFs_to_A23format(LAFs)
    plt.figure()
    plt.imshow(255 - img)
    for i in range(len(work_LAFs)):
        ell = LAF2pts(work_LAFs[i,:,:])
        plt.plot( ell[:,0], ell[:,1], 'r')
    plt.show()
    return 

####pytorch

def get_normalized_affine_shape(tilt, angle_in_radians):
    assert tilt.size(0) == angle_in_radians.size(0)
    num = tilt.size(0)
    tilt_A = Variable(torch.eye(2).view(1,2,2).repeat(num,1,1))
    if tilt.is_cuda:
        tilt_A = tilt_A.cuda()
    tilt_A[:,0,0] = tilt;
    rotmat = get_rotation_matrix(angle_in_radians)
    out_A = rectifyAffineTransformationUpIsUp(torch.bmm(rotmat, torch.bmm(tilt_A, rotmat)))
    #re_scale = (1.0/torch.sqrt((out_A **2).sum(dim=1).max(dim=1)[0])) #It is heuristic to for keeping scale change small
    #re_scale = (0.5 + 0.5/torch.sqrt((out_A **2).sum(dim=1).max(dim=1)[0])) #It is heuristic to for keeping scale change small
    return out_A# * re_scale.view(-1,1,1).expand(num,2,2)

def get_rotation_matrix(angle_in_radians):
    angle_in_radians = angle_in_radians.view(-1, 1, 1);
    sin_a = torch.sin(angle_in_radians)
    cos_a = torch.cos(angle_in_radians)
    A1_x = torch.cat([cos_a, sin_a], dim = 2)
    A2_x = torch.cat([-sin_a, cos_a], dim = 2)
    transform = torch.cat([A1_x,A2_x], dim = 1)
    return transform
    
def rectifyAffineTransformationUpIsUp(A):
    det = torch.sqrt(torch.abs(A[:,0,0]*A[:,1,1] - A[:,1,0]*A[:,0,1] + 1e-10))
    b2a2 = torch.sqrt(A[:,0,1] * A[:,0,1] + A[:,0,0] * A[:,0,0])
    A1_ell = torch.cat([(b2a2 / det).contiguous().view(-1,1,1), 0 * det.view(-1,1,1)], dim = 2)
    A2_ell = torch.cat([((A[:,1,1]*A[:,0,1]+A[:,1,0]*A[:,0,0])/(b2a2*det)).contiguous().view(-1,1,1),
                        (det / b2a2).contiguous().view(-1,1,1)], dim = 2)
    return torch.cat([A1_ell, A2_ell], dim = 1)



def abc2A(a,b,c, normalize = False):
    A1_ell = torch.cat([a.view(-1,1,1), b.view(-1,1,1)], dim = 2)
    A2_ell = torch.cat([b.view(-1,1,1), c.view(-1,1,1)], dim = 2)
    return torch.cat([A1_ell, A2_ell], dim = 1)



def angles2A(angles):
    cos_a = torch.cos(angles).view(-1, 1, 1)
    sin_a = torch.sin(angles).view(-1, 1, 1)
    A1_ang = torch.cat([cos_a, sin_a], dim = 2)
    A2_ang = torch.cat([-sin_a, cos_a], dim = 2)
    return  torch.cat([A1_ang, A2_ang], dim = 1)

def generate_patch_grid_from_normalized_LAFs(LAFs, w, h, PS):
    num_lafs = LAFs.size(0)
    min_size = min(h,w)
    coef = torch.ones(1,2,3) * min_size
    coef[0,0,2] = w
    coef[0,1,2] = h
    if LAFs.is_cuda:
        coef = coef.cuda()
    grid = torch.nn.functional.affine_grid(LAFs * Variable(coef.expand(num_lafs,2,3)), torch.Size((num_lafs,1,PS,PS)))
    grid[:,:,:,0] = 2.0 * grid[:,:,:,0] / float(w)  - 1.0
    grid[:,:,:,1] = 2.0 * grid[:,:,:,1] / float(h)  - 1.0     
    return grid
    
def extract_patches(img, LAFs, PS = 32):
    w = img.size(3)
    h = img.size(2)
    ch = img.size(1)
    grid = generate_patch_grid_from_normalized_LAFs(LAFs, float(w),float(h), PS)
    return torch.nn.functional.grid_sample(img.expand(grid.size(0), ch, h, w),  grid)  

def get_pyramid_inverted_index_for_LAFs(LAFs, PS, sigmas):
    return

def extract_patches_from_pyramid_with_inv_index(scale_pyramid, pyr_inv_idxs, LAFs, PS = 19):
    patches = torch.zeros(LAFs.size(0),scale_pyramid[0][0].size(1), PS, PS)
    if LAFs.is_cuda:
        patches = patches.cuda()
    patches = Variable(patches)
    if pyr_inv_idxs is not None:
        for i in range(len(scale_pyramid)):
            for j in range(len(scale_pyramid[i])):
                cur_lvl_idxs = pyr_inv_idxs[i][j]
                if cur_lvl_idxs is None:
                    continue
                cur_lvl_idxs = cur_lvl_idxs.view(-1)
                #print i,j,cur_lvl_idxs.shape
                patches[cur_lvl_idxs,:,:,:] = extract_patches(scale_pyramid[i][j], LAFs[cur_lvl_idxs, :,:], PS )
    return patches

def get_inverted_pyr_index(scale_pyr, pyr_idxs, level_idxs):
    pyr_inv_idxs = []
    ### Precompute octave inverted indexes
    for i in range(len(scale_pyr)):
        pyr_inv_idxs.append([])
        cur_idxs = pyr_idxs == i #torch.nonzero((pyr_idxs == i).data)
        for j in range(0, len(scale_pyr[i])):
            cur_lvl_idxs = torch.nonzero(((level_idxs == j) * cur_idxs).data)
            if len(cur_lvl_idxs.size()) == 0:
                pyr_inv_idxs[i].append(None)
            else:
                pyr_inv_idxs[i].append(cur_lvl_idxs.squeeze())
    return pyr_inv_idxs


def denormalizeLAFs(LAFs, w, h):
    w = float(w)
    h = float(h)
    num_lafs = LAFs.size(0)
    min_size = min(h,w)
    coef = torch.ones(1,2,3).float()  * min_size
    coef[0,0,2] = w
    coef[0,1,2] = h
    if LAFs.is_cuda:
        coef = coef.cuda()
    return Variable(coef.expand(num_lafs,2,3)) * LAFs

def normalizeLAFs(LAFs, w, h):
    w = float(w)
    h = float(h)
    num_lafs = LAFs.size(0)
    min_size = min(h,w)
    coef = torch.ones(1,2,3).float()  / min_size
    coef[0,0,2] = 1.0 / w
    coef[0,1,2] = 1.0 / h
    if LAFs.is_cuda:
        coef = coef.cuda()
    return Variable(coef.expand(num_lafs,2,3)) * LAFs

def sc_y_x2LAFs(sc_y_x):
    base_LAF = torch.eye(2).float().unsqueeze(0).expand(sc_y_x.size(0),2,2)
    if sc_y_x.is_cuda:
        base_LAF = base_LAF.cuda()
    base_A = Variable(base_LAF, requires_grad=False)
    A = sc_y_x[:,:1].unsqueeze(1).expand_as(base_A) * base_A
    LAFs  = torch.cat([A,
                       torch.cat([sc_y_x[:,2:].unsqueeze(-1),
                                    sc_y_x[:,1:2].unsqueeze(-1)], dim=1)], dim = 2)
        
    return LAFs
def get_LAFs_scales(LAFs):
    return torch.sqrt(torch.abs(LAFs[:,0,0] *LAFs[:,1,1] - LAFs[:,0,1] * LAFs[:,1,0]) + 1e-12)

def get_pyramid_and_level_index_for_LAFs(dLAFs,  sigmas, pix_dists, PS):
    scales = get_LAFs_scales(dLAFs);
    needed_sigmas = scales / PS;
    sigmas_full_list = []
    level_idxs_full = []
    oct_idxs_full = []
    for oct_idx in range(len(sigmas)):
        sigmas_full_list = sigmas_full_list + list(np.array(sigmas[oct_idx])*np.array(pix_dists[oct_idx]))
        oct_idxs_full = oct_idxs_full + [oct_idx]*len(sigmas[oct_idx])
        level_idxs_full = level_idxs_full + range(0,len(sigmas[oct_idx]))
    oct_idxs_full = torch.LongTensor(oct_idxs_full)
    level_idxs_full = torch.LongTensor(level_idxs_full)
    
    closest_imgs = cdist(np.array(sigmas_full_list).reshape(-1,1), needed_sigmas.data.cpu().numpy().reshape(-1,1)).argmin(axis = 0)
    closest_imgs = torch.from_numpy(closest_imgs)
    if dLAFs.is_cuda:
        closest_imgs = closest_imgs.cuda()
        oct_idxs_full = oct_idxs_full.cuda()
        level_idxs_full = level_idxs_full.cuda()
    return  Variable(oct_idxs_full[closest_imgs]), Variable(level_idxs_full[closest_imgs])
