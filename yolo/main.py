import hydra
from omegaconf import DictConfig, OmegaConf
from torch import optim
import procedures
import torch
import os
import torch.distributed as dist
from nets.yolo_forw import YOLOForw
from procedures.init_dataset import get_dataloaders
from procedures.train_one_epoch import train_one_epoch
from procedures.valid_one_epoch import valid_one_epoch
from procedures.test_one_epoch import test_one_epoch
from procedures.eval_results import eval_results,save_results
from procedures.initialize import save_model,get_model
import logging
import torch.multiprocessing as mp

log = logging.getLogger(__name__)

def setup(rank, world_size):
    """Initializes distributed process group.
    Arguments:
        rank: the rank of the current process.
        world_size: the total number of processes.
        backend: the backend used for distributed processing.
    """
    os.environ["MASTER_ADDR"] = 'localhost'
    os.environ["MASTER_PORT"] = '12355'
    dist.init_process_group(backend='nccl', init_method='env://', rank=rank, world_size=world_size)


def cleanup():
    """Cleans up distributed backend resources."""
    dist.destroy_process_group()


@hydra.main(config_path="hydra",config_name="config")
def main(cfg: DictConfig) -> None:
    os.environ['owd'] = hydra.utils.get_original_cwd()
    mp.spawn(pipeline, nprocs=cfg.gpus, args=(cfg,))

def pipeline(rank,cfg):
    setup(rank, cfg.gpus)
    torch.cuda.set_device(rank)
    torch.manual_seed(0)
    cfg.rank=rank
    dset_config=cfg['dataset']
    mAP_best=0
    mAP = 0
    last_epoch=0
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.benchmark = True

    #model,optimizer
    model,optimizer,mAP_best,last_epoch=get_model(cfg)
    #dataloaders
    train_loader,test_loader = get_dataloaders(cfg)
    print(len(train_loader.dataset))       
    
    #criterion
    criterion = YOLOForw(cfg['yolo']).cuda()

    epochs=100
    batch_loss = torch.zeros(1)
    for i in range(epochs):
        if i>30:
            cfg.metric =='mAP'
        train_one_epoch(train_loader,model,optimizer,criterion,rank)
        if cfg.metric =='mAP':
            results=test_one_epoch(test_loader,model,criterion)
            save_results(results,rank)
            dist.barrier()
            if rank==0:
                mAP=eval_results(i+last_epoch,dset_config['dset_name'],dset_config['val_annotations'])
                print(f'map is {mAP}')
                save_model(model,optimizer,mAP,i+last_epoch,'last')
                if mAP>mAP_best:
                    save_model(model,optimizer,mAP,i,'best')
                    mAP_best=mAP
        else:
            batch_loss = valid_one_epoch(test_loader,model,criterion,rank)
            dist.all_reduce(batch_loss, op=torch.distributed.ReduceOp.SUM, async_op=False)
            batch_loss = batch_loss / len(test_loader.dataset)
            if rank==0:
                print(f'batch_loss is {batch_loss}')
                save_model(model,optimizer,mAP,i+last_epoch,'last')
    cleanup()

if __name__=='__main__':
    main()