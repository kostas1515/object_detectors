import sys
from apex import amp
import numpy as np
import time
import torch
def train_one_epoch(dataloader,model,optimizer,yolo_loss,rank):
    model.train()
    batch_loss=0
    counter = 0
    metrics=np.zeros(6)
    for imgs,targets in dataloader:
        for param in model.parameters():
            param.grad = None
        batch_loss=0
        targets = [{k: v.to('cuda') for k, v in t.items()} for t in targets]
        out=model(imgs.cuda())
        outcome=yolo_loss(out,targets)
        batch_loss= outcome[0]
        metrics = metrics + np.array(outcome[1:])
        
        with amp.scale_loss(batch_loss, optimizer) as scaled_loss:
            scaled_loss.backward()
        optimizer.step()
        counter += 1
        avg = metrics / counter
        if (rank == 0) & ((counter % 100) == 0):
            sys.stdout.write(f'\rloss is:{batch_loss.item()}, xy is:{avg[0]}, wh is:{avg[1]},iou is:{avg[2]}, pos_conf is:{avg[3]}, neg_conf is:{avg[4]}, class is:{avg[5]}')
            sys.stdout.flush()