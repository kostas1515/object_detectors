import sys
from apex import amp
import numpy as np
import torch
import torch.distributed as dist

def valid_one_epoch(dataloader,model,yolo_loss,cfg):
    model.eval()
    batch_loss=torch.zeros(1).cuda()
    metrics = torch.zeros(6,device='cuda')
    stats = torch.zeros(5,device='cuda')
    torch.backends.cudnn.benchmark = True
    inp_dim=cfg.dataset.inp_dim
    yolo_loss.set_img_size(inp_dim)
    counter = 0 
    with torch.no_grad():
        for imgs,targets in dataloader:
            for param in model.parameters():
                param.grad = None
            targets = [{k: v.to('cuda') for k, v in t.items()} for t in targets]
            out=model(imgs.cuda())
            counter = counter + 1
            outcome=yolo_loss(out,targets)
            batch_loss= batch_loss + outcome[0]
            sub_losses = outcome[1]
            metrics = metrics + sub_losses
            stats = stats + outcome[2]
            avg_l = metrics / counter
            avg_stats = stats / counter
        stats_msg = {'iter':(counter), 'iou':avg_stats[0].item(),'pos_conf':avg_stats[1].item(),'neg_conf':avg_stats[2].item(),'pos_class':avg_stats[3].item(),'neg_class':avg_stats[4].item()}
        if cfg.verbose is True:
            print(stats_msg)
            print(f'\rAVG loss is:{avg_l.sum()}, xy is:{avg_l[0]}, wh is:{avg_l[1]},iou is:{avg_l[2]}, pos_conf is:{avg_l[3]}, neg_conf is:{avg_l[4]}, class is:{avg_l[5]}')
        
        return batch_loss