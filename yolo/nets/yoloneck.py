import torch
import torch.nn as nn
import torch.functional as F
from collections import OrderedDict
import os
import hydra
import time

from .backbone import backbone_fn
from  utilities.custom import FPN,SPP


class YoloNeck(nn.Module):
    def __init__(self, config,device='cuda'):
        super(YoloNeck, self).__init__()
        self.device=device

        if config['neck']['spp_bottleneck'] is True:  
            bottleneck = True #if spp before fpn and btlnc==True then embs have normal dims[-1]
        else:
            bottleneck = False #if spp before fpn and btlnc==False then embs have [(3+1) * emb[k].shape[1] + emb[k-1].shape[1]) dims

        if config['neck']['fpn'] is True:
            self.do_fpn=True
            # self.fpn0 = FPN(1024,device='cuda')
            self.fpn1 = FPN(512,bottleneck,device=self.device)
            self.fpn2 = FPN(256,bottleneck,device=self.device)
        else:
            self.do_fpn = False
        
        if config['neck']['spp'] is True:
            self.do_spp = True
            self.spp = SPP(config['neck']['pyramids'],bottleneck)
        else:
            self.do_spp = False

    def forward(self, embeddings):
        #x2:256, x1=512, x0=1024
        # (x0,x1,x2) = embeddings

        if self.do_spp is True:
            embeddings = self.spp(embeddings)
        
        if self.do_fpn is True:
            fxs = self.fpn2(embeddings) #256
            x2_out=fxs[2].clone()
            fused_embeddings = tuple(torch.cat([e,f],dim=1) for e,f in zip(embeddings,fxs))

            fxs = self.fpn1(fused_embeddings) #512

            # (x0, _, _) = self.fpn0((x0, x1, x2)) #1024
        
        return embeddings[0],fxs[1].clone(),x2_out
