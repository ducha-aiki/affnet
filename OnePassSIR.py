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
from HandCraftedModules import HessianResp, AffineShapeEstimator, OrientationDetector, ScalePyramid, NMS3dAndComposeA,NMS3dAndComposeAAff
import time

class OnePassSIR(nn.Module):
    def __init__(self, 
                 border = 16,
                 num_features = 500,
                 patch_size = 32,
                 mrSize = 3.0,
                 nlevels = 3,
                 th = None,#16.0/ 3.0,
                 num_Baum_iters = 0,
                 init_sigma = 1.6,
                 RespNet = None, OriNet = None, AffNet = None):
        super(OnePassSIR, self).__init__()
        self.mrSize = mrSize
        self.PS = patch_size
        self.b = border;
        self.num = num_features
        self.th = th;
        if th is not None:
            self.num = -1
        else:
            self.th = 0
        self.nlevels = nlevels
        self.num_Baum_iters = num_Baum_iters
        self.init_sigma = init_sigma
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
    
    def multiScaleDetectorAff(self,x, num_features = 0):
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
            nms_f = NMS3dAndComposeAAff(w = octave[0].size(3),
                                     h =  octave[0].size(2),
                                     border = self.b, mrSize = self.mrSize)
            #oct_aff_map =  F.upsample(self.AffNet(octave[0]), (octave[0].size(2), octave[0].size(3)),mode='bilinear')
            oct_aff_map =  self.AffNet(octave[0])
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
                                                                 scales = sigmas_oct[level_idx - 1:level_idx + 2],
                                                                 aff_resp = oct_aff_map)
                if top_resp is None:
                    continue
                octaveMap = octaveMap_current
                not_touch_boundary_idx = checkTouchBoundary(torch.cat([aff_matrix[:,:2,:2] *3.0, aff_matrix[:,:,2:]], dim =2))
                aff_matrices.append(aff_matrix[not_touch_boundary_idx.byte()]), top_responces.append(top_resp[not_touch_boundary_idx.byte()])
                pyr_id = Variable(oct_idx * torch.ones(aff_matrices[-1].size(0)))
                lev_id = Variable((level_idx - 1) * torch.ones(aff_matrices[-1].size(0))) #prevBlur
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
            if i != max_iters-1:
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
    def forward(self,x, do_ori = True):
        ### Detection
        t = time.time()
        num_features_prefilter = self.num
        responses, LAFs, final_pyr_idxs, final_level_idxs = self.multiScaleDetectorAff(x,num_features_prefilter)
        print time.time() - t, 'detection multiscale'
        t = time.time()
        LAFs[:,0:2,0:2] =   self.mrSize * LAFs[:,:,0:2]
        if do_ori:
            LAFs = self.getOrientation(LAFs, final_pyr_idxs, final_level_idxs)
        #pyr_inv_idxs = get_inverted_pyr_index(scale_pyr, final_pyr_idxs, final_level_idxs)
        #patches = extract_patches_from_pyramid_with_inv_index(scale_pyr, pyr_inv_idxs, LAFs, PS = self.PS)
        #patches = extract_patches(x, LAFs, PS = self.PS)
        #print time.time() - t, len(LAFs), ' patches extraction'
        return denormalizeLAFs(LAFs, x.size(3), x.size(2)), responses
