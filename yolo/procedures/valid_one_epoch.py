import sys
from apex import amp
import numpy as np
import torch
import torch.distributed as dist

def valid_one_epoch(dataloader,model,yolo_loss,rank):
    model.eval()
    batch_loss=torch.zeros(1).cuda()
    torch.backends.cudnn.benchmark = True
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
            
            if (rank == 0) &((counter % 10) ==0):
                print(outcome[0])

        return batch_loss