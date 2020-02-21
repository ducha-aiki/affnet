import torch
import torch.nn as nn
import numpy as np
import math
import torch.nn.functional as F
from torch.autograd import Variable
from copy import deepcopy
from Utils import GaussianBlur, batch_eig2x2, line_prepender, batched_forward
from LAF import LAFs2ell,abc2A, angles2A, generate_patch_grid_from_normalized_LAFs, extract_patches, get_inverted_pyr_index, denormalizeLAFs, extract_patches_from_pyramid_with_inv_index, rectifyAffineTransformationUpIsUp
from LAF import get_pyramid_and_level_index_for_LAFs, normalizeLAFs, checkTouchBoundary
from HandCraftedModules import HessianResp, AffineShapeEstimator, OrientationDetector, ScalePyramid, NMS3dAndComposeA
import time

class ScaleSpaceAffinePatchExtractor(nn.Module):
    def __init__(self, 
                 border = 16,
                 num_features = 500,
                 patch_size = 32,
                 mrSize = 3.0,
                 nlevels = 3,
                 num_Baum_iters = 0,
                 init_sigma = 1.6,
                 th = None,
                 RespNet = None, OriNet = None, AffNet = None):
        super(ScaleSpaceAffinePatchExtractor, self).__init__()
        self.mrSize = mrSize
        self.PS = patch_size
        self.b = border;
        self.num = num_features
        self.nlevels = nlevels
        self.num_Baum_iters = num_Baum_iters
        self.init_sigma = init_sigma
        self.th = th;
        if th is not None:
            self.num = -1
        else:
            self.th = 0
        if RespNet is not None:
            self.RespNet = RespNet
        else:
            self.RespNet = HessianResp()
        if OriNet is not None:
            self.OriNet = OriNet
        else:
            self.OriNet= OrientationDetector(patch_size = 19);
        if AffNet is not None:
            self.AffNet = AffNet
        else:
            self.AffNet = AffineShapeEstimator(patch_size = 19)
        self.ScalePyrGen = ScalePyramid(nLevels = self.nlevels, init_sigma = self.init_sigma, border = self.b)
        return
    
    def multiScaleDetector(self,x, num_features = 0):
        t = time.time()
        self.scale_pyr, self.sigmas, self.pix_dists = self.ScalePyrGen(x)
        ### Detect keypoints in scale space
        aff_matrices = []
        top_responces = []
        pyr_idxs = []
        level_idxs = []
        det_t = 0
        nmst = 0
        for oct_idx in range(len(self.sigmas)):
            #print oct_idx
            octave = self.scale_pyr[oct_idx]
            sigmas_oct = self.sigmas[oct_idx]
            pix_dists_oct = self.pix_dists[oct_idx]
            low = None
            cur = None
            high = None
            octaveMap = (self.scale_pyr[oct_idx][0] * 0).byte()
            nms_f = NMS3dAndComposeA(w = octave[0].size(3),
                                     h =  octave[0].size(2),
                                     border = self.b, mrSize = self.mrSize)
            for level_idx in range(1, len(octave)-1):
                if cur is None:
                    low = torch.clamp(self.RespNet(octave[level_idx - 1], (sigmas_oct[level_idx - 1 ])) - self.th, min = 0)
                else:
                    low = cur
                if high is None:
                    cur =  torch.clamp(self.RespNet(octave[level_idx ], (sigmas_oct[level_idx ])) - self.th, min = 0)
                else:
                    cur = high
                high = torch.clamp(self.RespNet(octave[level_idx + 1], (sigmas_oct[level_idx + 1 ])) - self.th, min = 0)
                top_resp, aff_matrix, octaveMap_current  = nms_f(low, cur, high,
                                                                 num_features = num_features,
                                                                 octaveMap = octaveMap,
                                                                 scales = sigmas_oct[level_idx - 1:level_idx + 2])
                if top_resp is None:
                    continue
                octaveMap = octaveMap_current
                aff_matrices.append(aff_matrix), top_responces.append(top_resp)
                pyr_id = Variable(oct_idx * torch.ones(aff_matrix.size(0)))
                lev_id = Variable((level_idx - 1) * torch.ones(aff_matrix.size(0))) #prevBlur
                if x.is_cuda:
                    pyr_id = pyr_id.cuda()
                    lev_id = lev_id.cuda()
                pyr_idxs.append(pyr_id)
                level_idxs.append(lev_id)
        all_responses = torch.cat(top_responces, dim = 0)
        aff_m_scales = torch.cat(aff_matrices,dim = 0)
        pyr_idxs_scales = torch.cat(pyr_idxs,dim = 0)
        level_idxs_scale = torch.cat(level_idxs, dim = 0)
        if (num_features > 0) and (num_features < all_responses.size(0)):
            all_responses, idxs = torch.topk(all_responses, k = num_features);
            LAFs = torch.index_select(aff_m_scales, 0, idxs)
            final_pyr_idxs = pyr_idxs_scales[idxs]
            final_level_idxs = level_idxs_scale[idxs]
        else:
            return all_responses, aff_m_scales, pyr_idxs_scales , level_idxs_scale
        return all_responses, LAFs, final_pyr_idxs, final_level_idxs,
    
    def getAffineShape(self, final_resp, LAFs, final_pyr_idxs, final_level_idxs, num_features = 0):
        pe_time = 0
        affnet_time = 0
        pyr_inv_idxs = get_inverted_pyr_index(self.scale_pyr, final_pyr_idxs, final_level_idxs)
        t = time.time()
        patches_small = extract_patches_from_pyramid_with_inv_index(self.scale_pyr, pyr_inv_idxs, LAFs, PS = self.AffNet.PS)
        pe_time+=time.time() - t
        t = time.time()
        base_A = torch.eye(2).unsqueeze(0).expand(final_pyr_idxs.size(0),2,2)
        if final_resp.is_cuda:
            base_A = base_A.cuda()
        base_A = Variable(base_A)
        is_good = None
        n_patches = patches_small.size(0)
        for i in range(self.num_Baum_iters):
            t = time.time()
            A = batched_forward(self.AffNet, patches_small, 256)
            is_good_current = 1
            affnet_time += time.time() - t
            if is_good is None:
                is_good = is_good_current
            else:
                is_good = is_good * is_good_current
            base_A = torch.bmm(A, base_A); 
            new_LAFs = torch.cat([torch.bmm(base_A,LAFs[:,:,0:2]), LAFs[:,:,2:] ], dim =2)
            #print torch.sqrt(new_LAFs[0,0,0]*new_LAFs[0,1,1] - new_LAFs[0,1,0] *new_LAFs[0,0,1]) * scale_pyr[0][0].size(2)
            if i != self.num_Baum_iters - 1:
                pe_time+=time.time() - t
                t = time.time()
                patches_small =  extract_patches_from_pyramid_with_inv_index(self.scale_pyr, pyr_inv_idxs, new_LAFs, PS = self.AffNet.PS)
                pe_time+= time.time() - t
                l1,l2 = batch_eig2x2(A)      
                ratio1 =  torch.abs(l1 / (l2 + 1e-8))
                converged_mask = (ratio1 <= 1.2) * (ratio1 >= (0.8)) 
        l1,l2 = batch_eig2x2(base_A)
        ratio = torch.abs(l1 / (l2 + 1e-8))
        idxs_mask =  ((ratio < 6.0) * (ratio > (1./6.))) * checkTouchBoundary(new_LAFs)
        num_survived = idxs_mask.float().sum()
        if (num_features > 0) and (num_survived.data.item() > num_features):
            final_resp =  final_resp * idxs_mask.float() #zero bad points
            final_resp, idxs = torch.topk(final_resp, k = num_features);
        else:
            idxs = Variable(torch.nonzero(idxs_mask.data).view(-1).long())
            final_resp = final_resp[idxs]
        final_pyr_idxs = final_pyr_idxs[idxs]
        final_level_idxs = final_level_idxs[idxs]
        base_A = torch.index_select(base_A, 0, idxs)
        LAFs = torch.index_select(LAFs, 0, idxs)
        new_LAFs = torch.cat([torch.bmm(base_A, LAFs[:,:,0:2]),
                               LAFs[:,:,2:]], dim =2)
        print ('affnet_time',affnet_time)
        print ('pe_time', pe_time)
        return final_resp, new_LAFs, final_pyr_idxs, final_level_idxs  
    
    def getOrientation(self, LAFs, final_pyr_idxs, final_level_idxs):
        pyr_inv_idxs = get_inverted_pyr_index(self.scale_pyr, final_pyr_idxs, final_level_idxs)
        patches_small = extract_patches_from_pyramid_with_inv_index(self.scale_pyr, pyr_inv_idxs, LAFs, PS = self.OriNet.PS)
        max_iters = 1
        ### Detect orientation
        for i in range(max_iters):
            angles = self.OriNet(patches_small)
            if len(angles.size()) > 2:
                LAFs = torch.cat([torch.bmm( LAFs[:,:,:2], angles), LAFs[:,:,2:]], dim = 2)
            else:
                LAFs = torch.cat([torch.bmm( LAFs[:,:,:2], angles2A(angles).view(-1,2,2)), LAFs[:,:,2:]], dim = 2)
            if i != max_iters:
                patches_small = extract_patches_from_pyramid_with_inv_index(self.scale_pyr, pyr_inv_idxs, LAFs, PS = self.OriNet.PS)        
        return LAFs
    def extract_patches_from_pyr(self, dLAFs, PS = 41):
        pyr_idxs, level_idxs = get_pyramid_and_level_index_for_LAFs(dLAFs, self.sigmas, self.pix_dists, PS)
        pyr_inv_idxs = get_inverted_pyr_index(self.scale_pyr, pyr_idxs, level_idxs)
        patches = extract_patches_from_pyramid_with_inv_index(self.scale_pyr,
                                                      pyr_inv_idxs,
                                                      normalizeLAFs(dLAFs, self.scale_pyr[0][0].size(3), self.scale_pyr[0][0].size(2)), 
                                                      PS = PS)
        return patches
    def forward(self,x, do_ori = False):
        ### Detection
        t = time.time()
        num_features_prefilter = self.num
        if self.num_Baum_iters > 0:
            num_features_prefilter = int(1.5 * self.num);
        responses, LAFs, final_pyr_idxs, final_level_idxs = self.multiScaleDetector(x,num_features_prefilter)
        print (time.time() - t, 'detection multiscale')
        t = time.time()
        LAFs[:,0:2,0:2] =   self.mrSize * LAFs[:,:,0:2]
        if self.num_Baum_iters > 0:
            responses, LAFs, final_pyr_idxs, final_level_idxs  = self.getAffineShape(responses, LAFs, final_pyr_idxs, final_level_idxs, self.num)
        print (time.time() - t, 'affine shape iters')
        t = time.time()
        if do_ori:
            LAFs = self.getOrientation(LAFs, final_pyr_idxs, final_level_idxs)
            #pyr_inv_idxs = get_inverted_pyr_index(self.scale_pyr, final_pyr_idxs, final_level_idxs)
        #patches = extract_patches_from_pyramid_with_inv_index(scale_pyr, pyr_inv_idxs, LAFs, PS = self.PS)
        #patches = extract_patches(x, LAFs, PS = self.PS)
        #print time.time() - t, len(LAFs), ' patches extraction'
        return denormalizeLAFs(LAFs, x.size(3), x.size(2)), responses
