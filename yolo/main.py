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
from procedures.test_one_epoch import test_one_epoch
from procedures.eval_results import eval_results,save_results
from procedures.initialize import save_model,get_model




def setup(master_addr: str, master_port: str, rank: int, world_size: int, backend: str):
    """Initializes distributed process group.
    Arguments:
        rank: the rank of the current process.
        world_size: the total number of processes.
        backend: the backend used for distributed processing.
    """
    os.environ["MASTER_ADDR"] = master_addr
    os.environ["MASTER_PORT"] = master_port
    dist.init_process_group(backend=backend, init_method='env://', rank=rank, world_size=world_size)


def cleanup():
    """Cleans up distributed backend resources."""
    dist.destroy_process_group()


@hydra.main(config_path="hydra",config_name="config")
def main(cfg: DictConfig) -> None:
    setup(cfg.master_addr, cfg.master_port, cfg.rank, cfg.world_size, cfg.backend)
    torch.cuda.set_device(cfg.rank)
    dset_config=cfg['dataset']
    mAP_best=0
    mAP = 0
    last_epoch=0

    #model,optimizer
    model,optimizer,mAP_best,last_epoch=get_model(cfg)
    #dataloaders
    train_loader,test_loader = get_dataloaders(cfg)           
    
    #criterion
    criterion = YOLOForw(cfg['yolo'])


    epochs=60
    for i in range(epochs):
        if cfg.only_test is False:
            train_one_epoch(train_loader,model,optimizer,criterion,cfg.rank)
        results=test_one_epoch(test_loader,model,criterion)
        save_results(results,cfg.rank)
        dist.barrier()
        if cfg.rank==0:
            mAP=eval_results(i+last_epoch,dset_config['dset_name'],dset_config['val_annotations'])
            if cfg.only_test is False:
                save_model(model,optimizer,mAP,i+last_epoch,'last')
                if mAP>mAP_best:
                    save_model(model,optimizer,mAP,i,'best')
                    mAP_best=mAP
            print(f'map is {mAP}')


    cleanup()
    return mAP

if __name__=='__main__':
    main()