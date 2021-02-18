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
        fh = logging.FileHandler('epoch.log','w')
        fh.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        fh.setFormatter(formatter)
        logger.addHandler(fh)
        logger.propagate = False

    model.train()
    batch_loss=0
    counter = 0
    metrics=np.zeros(6)
    for imgs,targets in dataloader:
        for param in model.parameters():
            param.grad = None
        batch_loss=0
        imgs=imgs.to('cuda',non_blocking=True)
        targets = [{k: v.to('cuda',non_blocking=True) for k, v in t.items()} for t in targets]
        out=model(imgs)
        outcome=yolo_loss(out,targets)
        batch_loss= outcome[0]
        metrics = metrics + np.array(outcome[1:])
        
        with amp.scale_loss(batch_loss, optimizer) as scaled_loss:
            scaled_loss.backward()
        optimizer.step()
        counter += 1
        avg = metrics / counter

        if rank == 0:
            msg=f'Iteration:{counter},Loss is:{np.array(outcome[1:]).sum()}, xy is:{outcome[1]},wh is:{outcome[2]},iou is:{outcome[3]},',f'pos_conf is:{outcome[4]}, neg_conf is:{outcome[5]},class is:{outcome[6]}'
            if torch.isnan(batch_loss):
                logger.warning(msg)
            else:
                logger.info(msg)

        if (rank == 0) & ((counter % 10) == 0):
            sys.stdout.write(f'\rAVG loss is:{avg.sum()}, xy is:{avg[0]}, wh is:{avg[1]},iou is:{avg[2]}, pos_conf is:{avg[3]}, neg_conf is:{avg[4]}, class is:{avg[5]}')
            sys.stdout.flush()

    avg = torch.stack(avg.tolist()).cuda()
    return avg