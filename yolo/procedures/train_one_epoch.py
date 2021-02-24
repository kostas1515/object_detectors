import sys
from apex import amp
import os
import numpy as np
import time
import torch
import logging
from torch.utils.tensorboard import SummaryWriter
import math



def train_one_epoch(dataloader,model,optimizer,yolo_loss,epoch,cfg):
    board = cfg.track_epoch
    verbose = cfg.verbose
    rank = cfg.rank
    freq = cfg.freq
    batch_size = cfg.dataset.tr_batch_size
    dataset_len = len(dataloader.dataset) / cfg.gpus
    iterations = math.ceil(epoch * (dataset_len / batch_size))

    
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)
    fh = logging.FileHandler('epoch.log','a')
    fh.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    logger.propagate = False
    if (board is True) and (rank==0):
        writer = SummaryWriter('track_epoch')

    # main training procedure
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
        #forw
        out=model(imgs)
        outcome=yolo_loss(out,targets)
        #stats
        batch_loss= outcome[0]
        sub_losses = outcome[1]
        metrics = metrics + sub_losses
        stats = stats + outcome[2]
        counter += 1
        avg_l = metrics / counter
        avg_stats = stats / counter

        if (torch.isnan(batch_loss)):
            msg=f'Rank: {rank}, Iteration:{counter + iterations},Loss is:{sub_losses.sum()}, xy is:{sub_losses[0]},wh is:{sub_losses[1]},iou is:{sub_losses[2]},pos_conf is:{sub_losses[3]}, neg_conf is:{sub_losses[4]},class is:{sub_losses[5]}'
            logger.warning(msg)
            print(msg)
            return None
        try:
            with amp.scale_loss(batch_loss, optimizer) as scaled_loss:
                scaled_loss.backward()
        except ZeroDivisionError:
            print(batch_loss)
            logger.warning(batch_loss)
            return None
                
        optimizer.step()
        
        # logging displaying and tensorboard
        if rank == 0:
            if (verbose is True):
                stats_msg = {'iter':(counter -1 + iterations), 'iou':avg_stats[0].item(),'pos_conf':avg_stats[1].item(),'neg_conf':avg_stats[2].item(),'pos_class':avg_stats[3].item(),'neg_class':avg_stats[4].item()}
                logger.info(stats_msg)
                if ((counter % freq) == 0):
                    sys.stdout.write(f'\rAVG loss is:{avg_l.sum()}, xy is:{avg_l[0]}, wh is:{avg_l[1]},iou is:{avg_l[2]}, pos_conf is:{avg_l[3]}, neg_conf is:{avg_l[4]}, class is:{avg_l[5]}')
                    sys.stdout.flush()

            if board is True:
                writer.add_scalar('Avg Loss/loss', avg_l.sum(),iterations + counter-1)
                writer.add_scalar('XY Loss/loss', avg_l[0],iterations + counter-1)
                writer.add_scalar('WH Loss/loss', avg_l[1],iterations + counter-1)
                writer.add_scalar('IOU Loss/loss', avg_l[2],iterations + counter-1)
                writer.add_scalar('Pos_Conf Loss/loss', avg_l[3],iterations + counter-1)
                writer.add_scalar('Neg_Conf Loss/loss', avg_l[4],iterations + counter-1)
                writer.add_scalar('Class Loss/loss', avg_l[5],iterations + counter-1)
                writer.add_scalar('IOU/train', avg_stats[0].item(),iterations + counter-1)
                writer.add_scalar('Pos Conf/train', avg_stats[1].item(),iterations + counter-1)
                writer.add_scalar('Neg Conf/train', avg_stats[2].item(),iterations + counter-1)
                writer.add_scalar('Pos Class/train', avg_stats[3].item(),iterations + counter-1)
                writer.add_scalar('Neg Class/train', avg_stats[4].item(),iterations + counter-1)

    return avg_l,avg_stats