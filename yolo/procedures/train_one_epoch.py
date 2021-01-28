import sys
from apex import amp
def train_one_epoch(dataloader,model,optimizer,yolo_loss,rank):
    model.train()
    batch_loss=0
    counter = 0 
    for imgs,targets in dataloader:
        for param in model.parameters():
            param.grad = None
        batch_loss=0
        
        out=model(imgs.cuda())
        for k,y in enumerate(out):
            outcome=yolo_loss[k](y,targets)
            batch_loss=batch_loss+outcome[0]
        with amp.scale_loss(batch_loss, optimizer) as scaled_loss:
            scaled_loss.backward()
        optimizer.step()
        counter += 1
        if (rank == 0) & ((counter % 100) == 0):
            sys.stdout.write(f'\r{batch_loss.item()}')
            sys.stdout.flush()