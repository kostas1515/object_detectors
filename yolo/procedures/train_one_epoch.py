import sys
from apex import amp
import numpy as np
import time
import torch
import logging



def train_one_epoch(dataloader,model,optimizer,yolo_loss,rank):
    if rank ==0:
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.DEBUG)
        fh = logging.FileHandler('epoch.log','a')
        fh.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        fh.setFormatter(formatter)
        logger.addHandler(fh)
        logger.propagate = False

    model.train()
    batch_loss=0
    counter = 0
    metrics = torch.zeros(6,device='cuda')
    stats = torch.zeros(5,device='cuda')
    for imgs,targets in dataloader:
        for param in model.parameters():
            param.grad = None
        batch_loss=0
        imgs=imgs.to('cuda',non_blocking=True)
        targets = [{k: v.to('cuda',non_blocking=True) for k, v in t.items()} for t in targets]
        out=model(imgs)
        outcome=yolo_loss(out,targets)
        batch_loss= outcome[0]
        sub_losses = outcome[1]
        metrics = metrics + sub_losses
        stats = stats + outcome[2]
        
        with amp.scale_loss(batch_loss, optimizer) as scaled_loss:
            scaled_loss.backward()
        optimizer.step()
        counter += 1
        avg_l = metrics / counter
        avg_stats = stats / counter
        
        if rank == 0:
            msg=f'Iteration:{counter},Loss is:{sub_losses.sum()}, xy is:{sub_losses[0]},wh is:{sub_losses[1]},iou is:{sub_losses[2]},pos_conf is:{sub_losses[3]}, neg_conf is:{sub_losses[4]},class is:{sub_losses[5]}'
            if torch.isnan(batch_loss):
                logger.warning(msg)
            else:
                stats_msg = {'iou':avg_stats[0].item(),'pos_conf':avg_stats[1].item(),'neg_conf':avg_stats[2].item(),'pos_class':avg_stats[3].item(),'neg_class':avg_stats[4].item()}
                logger.info(stats_msg)

        if (rank == 0) & ((counter % 10) == 0):
            sys.stdout.write(f'\rAVG loss is:{avg_l.sum()}, xy is:{avg_l[0]}, wh is:{avg_l[1]},iou is:{avg_l[2]}, pos_conf is:{avg_l[3]}, neg_conf is:{avg_l[4]}, class is:{avg_l[5]}')
            sys.stdout.flush()

    return avg_l,avg_stats