from torch import optim
from nets.yolohead import YoloHead
import os
import torch
from torch import nn
from collections import OrderedDict
from apex.parallel import DistributedDataParallel as DDP
from apex import amp
import time


def save_model(model,optimizer,mAP,epoch,name):

    if not os.path.exists('checkpoints/'):
        os.makedirs('checkpoints/')
    
    checkpoint=f'checkpoints/{name}.tar'

    torch.save({'epoch': epoch,
                'model_state_dict': model.module.state_dict() if type(model) is DDP else model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'optimizer_name':optimizer.name,
                'mAP': mAP
                }, checkpoint)


def get_model(cfg):
    model = YoloHead(cfg)
    # model.load_state_dict(torch.load('weights/yolov3_orig.pth'))
    model=model.cuda()
    epoch=0
    mAP=0
    optimizer=None

    if not os.path.exists('checkpoints/'):
        if cfg['optimizer']['name'] == 'sgd':
            optimizer=optim.SGD(model.parameters(), lr=cfg['optimizer']['lr'], momentum=cfg['optimizer']['momentum'],weight_decay=cfg['optimizer']['weight_decay'])
            
        elif cfg['optimizer']['name'] == 'adam':
            optimizer=optim.Adam(model.parameters(), lr=cfg['optimizer']['lr'],weight_decay=cfg['optimizer']['weight_decay'])
            
        optimizer_name=cfg['optimizer']['name']
    else:
        cp_name=cfg['experiment']['cp']
        checkpoint = torch.load(f'checkpoints/{cp_name}.tar')
        try:
            model.load_state_dict(checkpoint['model_state_dict'])
        except RuntimeError:
            new_state_dict = OrderedDict()
            for k, v in checkpoint['model_state_dict'].items():
                name = k[7:] # remove 'module.' of dataparallel
                new_state_dict[name]=v
            model.load_state_dict(new_state_dict)
        optimizer_name=checkpoint['optimizer_name']
        epoch=checkpoint['epoch']
        mAP=checkpoint['mAP']

        if cfg['experiment']['override_optimizer'] is False:
            if optimizer_name=='sgd':
                optimizer=optim.SGD(model.parameters(), lr=0)
            elif optimizer_name == 'adam':
                optimizer=optim.Adam(model.parameters(), lr=0)
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        else:
            if cfg['optimizer']['name'] == 'sgd':
                optimizer=optim.SGD(model.parameters(), lr=cfg['optimizer']['lr'], momentum=cfg['optimizer']['momentum'],weight_decay=cfg['optimizer']['weight_decay'])
            
            elif cfg['optimizer']['name'] == 'adam':
                optimizer=optim.Adam(model.parameters(), lr=cfg['optimizer']['lr'],weight_decay=cfg['optimizer']['weight_decay'])

    model, optimizer = amp.initialize(model,optimizer, 
                                      opt_level=cfg.apex_opt)
    optimizer.name = optimizer_name

    try:
        model = DDP(model)
    except AssertionError:
        pass
    return model,optimizer,mAP,epoch
