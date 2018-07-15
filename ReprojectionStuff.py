import torch
from torch.autograd import Variable
from torch.autograd import Variable as V
import numpy as np
from LAF import rectifyAffineTransformationUpIsUp, LAFs_to_H_frames
from Utils import zeros_like


def linH(H, x, y):
    assert x.size(0) == y.size(0)
    A = torch.zeros(x.size(0),2,2)
    if x.is_cuda:
        A = A.cuda()
    den = x * H[2,0] + y * H[2,1] + H[2,2]
    num1_densq = (x*H[0,0] + y*H[0,1] + H[0,2]) / (den*den)
    num2_densq = (x*H[1,0] + y*H[1,1] + H[1,2]) / (den*den)
    A[:,0,0] = H[0,0]/den - num1_densq * H[2,0]
    A[:,0,1] = H[0,1]/den - num1_densq * H[2,1]
    A[:,1,0] = H[1,0]/den - num2_densq * H[2,0]
    A[:,1,1] = H[1,1]/den - num2_densq * H[2,1]
    return A

def reprojectLAFs(LAFs1, H1to2, return_LHFs = False):
    LHF1 = LAFs_to_H_frames(LAFs1)
    #LHF1_in_2 = torch.zeros(LHF1.size(0), ,3,3)
    #if LHF1.is_cuda:
    #    LHF1_in_2 = LHF1_in_2.cuda()
    #LHF1_in_2 = Variable(LHF1_in_2)
    #LHF1_in_2[:,:,2] = torch.bmm(H1to2.expand(LHF1.size(0),3,3), LHF1[:,:,2:])
    #LHF1_in_2[:,:,2] = LHF1_in_2[:,:,2] / LHF1_in_2[:,2:,2].expand(LHF1_in_2.size(0), 3)
    #As  = linH(H1to2, LAFs1[:,0,2], LAFs1[:,1,2])
    #LHF1_in_2[:,0:2,0:2] = torch.bmm(As, LHF1[:,0:2,0:2])
    xy1 = torch.bmm(H1to2.expand(LHF1.size(0),3,3), LHF1[:,:,2:])
    xy1 = xy1 / xy1[:,2:,:].expand(xy1.size(0), 3, 1)
    As  = linH(H1to2, LAFs1[:,0,2], LAFs1[:,1,2])
    AF = torch.bmm(As, LHF1[:,0:2,0:2])
    
    if return_LHFs:
        return LAFs_to_H_frames(torch.cat([AF, xy1[:,:2,:]], dim = 2))
    return torch.cat([AF, xy1[:,:2,:]], dim = 2)

def Px2GridA(w, h):
    A = torch.eye(3)
    A[0,0] = 2.0  / float(w)
    A[1,1] = 2.0  / float(h)
    A[0,2] = -1
    A[1,2] = -1
    return A
def Grid2PxA(w, h):
    A = torch.eye(3)
    A[0,0] = float(w) / 2.0
    A[0,2] = float(w) / 2.0
    A[1,1] = float(h) / 2.0
    A[1,2] = float(h) / 2.0
    return A

def affineAug(img, max_add = 0.5):
    img_s = img.squeeze()
    h,w = img_s.size()
    ### Generate A
    A = torch.eye(3)
    rand_add = max_add *(torch.rand(3,3) - 0.5) * 2.0
    ##No perspective change
    rand_add[2,0:2] = 0
    rand_add[2,2] = 0;
    A  = A + rand_add
    denormA = Grid2PxA(w,h)
    normA = Px2GridA(w, h)
    if img.is_cuda:
        A = A.cuda()
        denormA = denormA.cuda()
        normA = normA.cuda()
    grid = torch.nn.functional.affine_grid(A[0:2,:].unsqueeze(0), torch.Size((1,1,h,w)))
    H_Orig2New = torch.mm(torch.mm(denormA, torch.inverse(A)), normA)
    new_img = torch.nn.functional.grid_sample(img_s.float().unsqueeze(0).unsqueeze(0),  grid)  
    return new_img, H_Orig2New, 

def distance_matrix_vector(anchor, positive):
    """Given batch of anchor descriptors and positive descriptors calculate distance matrix"""

    d1_sq = torch.sum(anchor * anchor, dim=1)
    d2_sq = torch.sum(positive * positive, dim=1)
    eps = 1e-12
    return torch.sqrt(torch.abs((d1_sq.expand(positive.size(0), anchor.size(0)) +
                       torch.t(d2_sq.expand(anchor.size(0), positive.size(0)))
                      - 2.0 * torch.bmm(positive.unsqueeze(0), torch.t(anchor).unsqueeze(0)).squeeze(0))+eps))

def ratio_matrix_vector(a, p):
    eps = 1e-12
    return a.expand(p.size(0), a.size(0)) / (torch.t(p.expand(a.size(0), p.size(0))) + eps)

   
def inverseLHFs(LHFs):
    LHF1_inv =torch.zeros(LHFs.size())
    if LHFs.is_cuda:
        LHF1_inv = LHF1_inv.cuda()
    for i in range(LHF1_inv.size(0)):
        LHF1_inv[i,:,:] = LHFs[i,:,:].inverse()
    return LHF1_inv

    
def reproject_to_canonical_Frob_batched(LHF1_inv, LHF2, batch_size = 2, skip_center = False):
    out = torch.zeros((LHF1_inv.size(0), LHF2.size(0)))
    eye1 = torch.eye(3)
    if LHF1_inv.is_cuda:
        out = out.cuda()
        eye1 = eye1.cuda()
    len1 = LHF1_inv.size(0)
    len2 = LHF2.size(0)
    n_batches = int(np.floor(len1 / batch_size) + 1);
    for b_idx in range(n_batches):
        #print b_idx
        start = b_idx * batch_size;
        fin = min((b_idx+1) * batch_size, len1)
        current_bs = fin - start
        if current_bs == 0:
            break
        should_be_eyes = torch.bmm(LHF1_inv[start:fin, :, :].unsqueeze(0).expand(len2,current_bs, 3, 3).contiguous().view(-1,3,3),
                                   LHF2.unsqueeze(1).expand(len2,current_bs, 3,3).contiguous().view(-1,3,3))
        if skip_center:
            out[start:fin, :] = torch.sum(((should_be_eyes - eye1.unsqueeze(0).expand_as(should_be_eyes))**2)[:,:2,:2] , dim=1).sum(dim = 1).view(current_bs, len2)
        else:
            out[start:fin, :] = torch.sum((should_be_eyes - eye1.unsqueeze(0).expand_as(should_be_eyes))**2 , dim=1).sum(dim = 1).view(current_bs, len2)
    return out

def get_GT_correspondence_indexes(LAFs1, LAFs2, H1to2, dist_threshold = 4):    
    LHF2_in_1_pre = reprojectLAFs(LAFs2, torch.inverse(H1to2), True)
    just_centers1 = LAFs1[:,:,2];
    just_centers2_repr_to_1 = LHF2_in_1_pre[:,0:2,2];
    
    dist  = distance_matrix_vector(just_centers2_repr_to_1, just_centers1)
    min_dist, idxs_in_2 = torch.min(dist,1)
    plain_indxs_in1 = torch.arange(0, idxs_in_2.size(0))
    if LAFs1.is_cuda:
        plain_indxs_in1 = plain_indxs_in1.cuda()
    mask =  min_dist <= dist_threshold
    return min_dist[mask], plain_indxs_in1[mask], idxs_in_2[mask]

def get_GT_correspondence_indexes_Fro(LAFs1,LAFs2, H1to2, dist_threshold = 4,
                                      skip_center_in_Fro = False):
    LHF2_in_1_pre = reprojectLAFs(LAFs2, torch.inverse(H1to2), True)
    LHF1_inv = inverseLHFs(LAFs_to_H_frames(LAFs1))
    frob_norm_dist = reproject_to_canonical_Frob_batched(LHF1_inv, LHF2_in_1_pre, batch_size = 2, skip_center = skip_center_in_Fro)
    min_dist, idxs_in_2 = torch.min(frob_norm_dist,1)
    plain_indxs_in1 = torch.arange(0, idxs_in_2.size(0))
    if LAFs1.is_cuda:
        plain_indxs_in1 = plain_indxs_in1.cuda()
    #print min_dist.min(), min_dist.max(), min_dist.mean()
    mask =  min_dist <= dist_threshold
    return min_dist[mask], plain_indxs_in1[mask], idxs_in_2[mask]

def get_GT_correspondence_indexes_Fro_and_center(LAFs1,LAFs2, H1to2, 
                                                 dist_threshold = 4, 
                                                 center_dist_th = 2.0,
                                                 scale_diff_coef = 0.3,
                                                 skip_center_in_Fro = False,
                                                 do_up_is_up = False,
                                                 return_LAF2_in_1 = False,
                                                 inv_to_eye = True):
    LHF2_in_1_pre = reprojectLAFs(LAFs2, torch.inverse(H1to2), True)
    if do_up_is_up:
        sc2 = torch.sqrt(torch.abs(LHF2_in_1_pre[:,0,0] * LHF2_in_1_pre[:,1,1] - LHF2_in_1_pre[:,1,0] * LHF2_in_1_pre[:,0,1])).unsqueeze(-1).unsqueeze(-1).expand(LHF2_in_1_pre.size(0), 2,2)
        LHF2_in_1 = torch.zeros(LHF2_in_1_pre.size())
        if LHF2_in_1_pre.is_cuda:
            LHF2_in_1 = LHF2_in_1.cuda()
        LHF2_in_1[:, :2,:2] = rectifyAffineTransformationUpIsUp(LHF2_in_1_pre[:, :2,:2]/sc2) * sc2
        LHF2_in_1[:,:, 2] = LHF2_in_1_pre[:,:,2]
        sc1 = torch.sqrt(torch.abs(LAFs1[:,0,0] * LAFs1[:,1,1] - LAFs1[:,1,0] * LAFs1[:,0,1])).unsqueeze(-1).unsqueeze(-1).expand(LAFs1.size(0), 2,2)
        LHF1 = LAFs_to_H_frames(torch.cat([rectifyAffineTransformationUpIsUp(LAFs1[:, :2,:2]/sc1) * sc1, LAFs1[:,:,2:]], dim = 2 ))
    else:
        LHF2_in_1 = LHF2_in_1_pre
        LHF1 = LAFs_to_H_frames(LAFs1)
    if inv_to_eye:
        LHF1_inv = inverseLHFs(LHF1)
        frob_norm_dist = reproject_to_canonical_Frob_batched(LHF1_inv, LHF2_in_1, batch_size = 2, skip_center = skip_center_in_Fro)
    else:
        if not skip_center_in_Fro:
            frob_norm_dist = distance_matrix_vector(LHF2_in_1.view(LHF2_in_1.size(0), -1), LHF1.view(LHF1.size(0),-1))
        else:
            frob_norm_dist = distance_matrix_vector(LHF2_in_1[:,0:2, 0:2].contiguous().view(LHF2_in_1.size(0), -1), LHF1[:,0:2,0:2].contiguous().view(LHF1.size(0),-1))
    #### Center replated
    just_centers1 = LAFs1[:,:,2];
    just_centers2_repr_to_1 = LHF2_in_1[:,0:2,2];
    if scale_diff_coef > 0:
        scales1 = torch.sqrt(torch.abs(LAFs1[:,0,0] * LAFs1[:,1,1] - LAFs1[:,1,0] * LAFs1[:,0,1]))
        scales2 = torch.sqrt(torch.abs(LHF2_in_1[:,0,0] * LHF2_in_1[:,1,1] - LHF2_in_1[:,1,0] * LHF2_in_1[:,0,1]))
        scale_matrix = ratio_matrix_vector(scales2, scales1)
        scale_dist_mask = (torch.abs(1.0 - scale_matrix) <= scale_diff_coef) 
    center_dist_mask  = distance_matrix_vector(just_centers2_repr_to_1, just_centers1) >= center_dist_th
    frob_norm_dist_masked = (1.0 - scale_dist_mask.float() + center_dist_mask.float()) * 1000. + frob_norm_dist;
    
    min_dist, idxs_in_2 = torch.min(frob_norm_dist_masked,1)
    plain_indxs_in1 = torch.arange(0, idxs_in_2.size(0))
    if LAFs1.is_cuda:
        plain_indxs_in1 = plain_indxs_in1.cuda()
    #min_dist, idxs_in_2 = torch.min(dist,1)
    #print min_dist.min(), min_dist.max(), min_dist.mean()
    mask =  (min_dist <= dist_threshold )
    
    if return_LAF2_in_1:
        return min_dist[mask], plain_indxs_in1[mask], idxs_in_2[mask], LHF2_in_1[:,0:2,:]
    else:
        return min_dist[mask], plain_indxs_in1[mask], idxs_in_2[mask]
def get_closest_correspondences_idxs(LHF1, LHF2_in_1, xy_th, scale_log):
    xy1 = LHF1[:,0:2,2];
    xy2in1 = LHF2_in_1[:,0:2,2];
    center_dist_matrix =  distance_matrix_vector(xy2in1, xy1)
    scales1 = torch.sqrt(torch.abs(LHF1[:,0,0] * LHF1[:,1,1] - LHF1[:,1,0] * LHF1[:,0,1]));
    scales2 = torch.sqrt(torch.abs(LHF2_in_1[:,0,0] * LHF2_in_1[:,1,1] - LHF2_in_1[:,1,0] * LHF2_in_1[:,0,1]));
    scale_matrix = torch.abs(torch.log(ratio_matrix_vector(scales2, scales1)))
    mask_matrix = 1000.0*(scale_matrix  > scale_log).float() * (center_dist_matrix > xy_th).float() + center_dist_matrix + scale_matrix

    d2_to_1, nn_idxs_in_2 = torch.min(mask_matrix,1)
    d1_to_2, nn_idxs_in_1 = torch.min(mask_matrix,0)

    flat_idxs_1 = torch.arange(0, nn_idxs_in_2.size(0));
    if LHF1.is_cuda:
        flat_idxs_1 = flat_idxs_1.cuda()
    mask = d2_to_1 <= 100.0;

    final_mask = (flat_idxs_1 == nn_idxs_in_1[nn_idxs_in_2].float()).float() * mask.float()
    idxs_in1 = flat_idxs_1[final_mask.long()].nonzero().squeeze()
    idxs_in_2_final = nn_idxs_in_2[idxs_in1];
    #torch.arange(0, nn_idxs_in_2.size(0))#[mask2.data]
    return idxs_in1, idxs_in_2_final
def get_LHFScale(LHF):
    return torch.sqrt(torch.abs(LHF[:,0,0] * LHF[:,1,1] - LHF[:,1,0] * LHF[:,0,1]));
def LAFMagic(LAFs1, LAFs2, H1to2, xy_th  = 5.0, scale_log = 0.4, t = 1.0, sc = 1.0, aff = 1.0):
    LHF2_in_1 = reprojectLAFs(LAFs2, torch.inverse(H1to2), True)
    LHF1 = LAFs_to_H_frames(LAFs1)
    idxs_in1, idxs_in_2 = get_closest_correspondences_idxs(LHF1, LHF2_in_1, xy_th, scale_log)
    if len(idxs_in1) == 0:
        print('Warning, no correspondences found')
        return None
    LHF1_good = LHF1[idxs_in1,:,:]
    LHF2_good = LHF2_in_1[idxs_in_2,:,:]
    scales1 = get_LHFScale(LHF1_good);
    scales2 = get_LHFScale(LHF2_good);
    max_scale = torch.max(scales1,scales2);
    min_scale = torch.min(scales1, scales2);
    mean_scale = 0.5 * (max_scale + min_scale)
    eps = 1e-12;
    if t != 0:
        dist_loss = torch.sqrt(torch.sum((LHF1_good[:,0:2,2] - LHF2_good[:,0:2,2])**2, dim = 1) + eps) / V(mean_scale.data);
    else:
        dist_loss = 0
    if sc != 0 :
        scale_loss = torch.log1p( (max_scale-min_scale)/(mean_scale))
    else:
        scale_loss = 0
    if aff != 0:
        A1 = LHF1_good[:,:2,:2] / scales1.view(-1,1,1).expand(scales1.size(0),2,2);
        A2 = LHF2_good[:,:2,:2] / scales2.view(-1,1,1).expand(scales2.size(0),2,2);
        shape_loss = ((A1 - A2)**2).mean(dim = 1).mean(dim = 1);
    else:
        shape_loss = 0;
    loss = t * dist_loss + sc * scale_loss + aff *shape_loss;
    #print dist_loss, scale_loss, shape_loss
    return loss, idxs_in1, idxs_in_2, LHF2_in_1[:,0:2,:]
def LAFMagicFro(LAFs1, LAFs2, H1to2, xy_th  = 5.0, scale_log = 0.4):
    LHF2_in_1 = reprojectLAFs(LAFs2, torch.inverse(H1to2), True)
    LHF1 = LAFs_to_H_frames(LAFs1)
    idxs_in1, idxs_in_2 = get_closest_correspondences_idxs(LHF1, LHF2_in_1, xy_th, scale_log)
    if len(idxs_in1) == 0:
        print('Warning, no correspondences found')
        return None
    LHF1_good = LHF1[idxs_in1,:,:]
    LHF2_good = LHF2_in_1[idxs_in_2,:,:]
    scales1 = get_LHFScale(LHF1_good);
    scales2 = get_LHFScale(LHF2_good);
    max_scale = torch.max(scales1,scales2);
    min_scale = torch.min(scales1, scales2);
    mean_scale = 0.5 * (max_scale + min_scale)
    eps = 1e-12;
    dist_loss = (torch.sqrt((LHF1_good.view(-1,9) - LHF2_good.view(-1,9))**2 + eps) / V(mean_scale.data).view(-1,1).expand(LHF1_good.size(0),9)).mean(dim=1); 
    loss = dist_loss;
    #print dist_loss, scale_loss, shape_loss
    return loss, idxs_in1, idxs_in_2, LHF2_in_1[:,0:2,:]
def pr_l(x):
    return x.mean().data.cpu().numpy()[0]
def add_1(A):
    add = torch.eye(2).unsqueeze(0).expand(A.size(0),2,2)
    add = torch.cat([add, torch.zeros(A.size(0),2,1)], dim = 2)
    if A.is_cuda:
        add = add.cuda()
    return add
def identity_loss(A):
    return torch.clamp(torch.sqrt((A - add_1(A))**2 + 1e-15).view(-1,6).mean(dim = 1) - 0.3*0, min = 0.0, max = 100.0).mean()



