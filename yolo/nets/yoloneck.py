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

        if config['neck']['fpn'] is True:
            self.do_fpn=True
            # self.fpn0 = FPN(1024,device='cuda')
            self.fpn1 = FPN(512,device=self.device)
            self.fpn2 = FPN(256,device=self.device)
        else:
            self.do_fpn=False
        
        if config['neck']['spp'] is True:
            self.do_spp=True
            self.spp = SPP(config['neck']['pyramids'])
        else:
            self.do_spp=False

    def forward(self, embeddings):
        #x2:256, x1=512, x0=1024
        (x0,x1,x2) = embeddings
        if self.do_spp is True:
            (x0, x1, x2) = self.spp((x0, x1, x2))

        if self.do_fpn is True:
            (_, _, x2) = self.fpn2((x0, x1, x2)) #256
            (_, x1, _) = self.fpn1((x0, x1, x2)) #512
            # (x0, _, _) = self.fpn0((x0, x1, x2)) #1024

        

        return (x0,x1,x2)
