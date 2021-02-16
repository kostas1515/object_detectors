import hydra
from omegaconf import DictConfig, OmegaConf
from torch import optim
import procedures
import torch
import os
from nets.yolo_forw import YOLOForw
from procedures.init_dataset import get_dataloaders
from procedures.train_one_epoch import train_one_epoch
from procedures.valid_one_epoch import valid_one_epoch
from procedures.test_one_epoch import test_one_epoch
from procedures.eval_results import eval_results,save_results
from procedures.initialize import get_model
import logging
import time

log = logging.getLogger(__name__)

@hydra.main(config_path="hydra",config_name="hyperopt")
def main(cfg: DictConfig) -> None:
    os.environ['owd'] = hydra.utils.get_original_cwd()

    torch.cuda.set_device(cfg.job_id % cfg.gpus)
    dset_config=cfg['dataset']
    mAP = 0
    last_epoch=0
    cfg.rank= (cfg.job_id % cfg.gpus)
    rank = cfg.rank

    #model,optimizer
    model,optimizer,_,last_epoch=get_model(cfg)
    #dataloaders
    train_loader,test_loader = get_dataloaders(cfg)           
    print('done dts')
    #criterion
    criterion = YOLOForw(cfg['yolo'])
    print('done criterion')
    epochs=1
    batch_loss = torch.zeros(1)
    for i in range(epochs):
        if cfg.only_test is False:
            print('start train')
            train_one_epoch(train_loader,model,optimizer,criterion,rank)

        if cfg.metrics =='mAP':
            results=test_one_epoch(test_loader,model,criterion)
            save_results(results,rank)
            mAP=eval_results(i+last_epoch,dset_config['dset_name'],dset_config['val_annotations'])
            print(f'map is {mAP}')
            return mAP
        else:
            batch_loss = valid_one_epoch(test_loader,model,criterion,rank)
            if torch.isnan(batch_loss):
                batch_loss = torch.tensor([1e6],device='cuda')

            log.info(f"RANK{rank}, lambda_xy:{cfg.yolo.lambda_xy}, lambda_wh: {cfg.yolo.lambda_wh}, \
                lambda_iou:{cfg.yolo.lambda_iou}, ignore_threshold:{cfg.yolo.ignore_threshold},\
                lambda_conf:{cfg.yolo.lambda_conf}, lambda_no_conf:{cfg.yolo.lambda_no_conf},\
                lambda_cls: {cfg.yolo.lambda_cls}, iou_type:{cfg.yolo.iou_type}, valid_loss={batch_loss.item()}")

            valid_loss = batch_loss.item()
            for i in range(10):
                try:
                    del model,batch_loss,optimizer,train_loader,test_loader,criterion
                except UnboundLocalError:
                    pass

            return valid_loss
if __name__=='__main__':
    main()