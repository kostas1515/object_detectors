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
from procedures.eval_results import eval_results,save_partial_results
from procedures.initialize import get_model,load_checkpoint
import logging
import time

log = logging.getLogger(__name__)

@hydra.main(config_path="hydra",config_name="hyperopt")
def main(cfg: DictConfig) -> None:
    os.environ['owd'] = hydra.utils.get_original_cwd()
    pid =os.getpid()
    torch.cuda.set_device(pid % cfg.gpus)
    cfg.rank=pid % cfg.gpus
    dset_config=cfg['dataset']
    mAP = 0
    last_epoch=0
    rank = cfg.rank

    #model,optimizer
    model,optimizer,_,last_epoch=get_model(cfg)
    #checkpoint
    if cfg.resume is True:
        metrics,last_epoch = load_checkpoint(model,optimizer,cfg)
    #dataloaders
    train_loader,test_loader = get_dataloaders(cfg)           

    #criterion
    criterion = YOLOForw(cfg['yolo'])
    epochs=1
    batch_loss = torch.zeros(1)
    for i in range(epochs):
        if cfg.only_test is False:
            status = train_one_epoch(train_loader,model,optimizer,criterion,i,cfg)
            if status is None:
                msg=f"RANK{rank}, lambda_xy:{cfg.yolo.lambda_xy}, lambda_wh: {cfg.yolo.lambda_wh},"+\
                    f"lambda_iou:{cfg.yolo.lambda_iou}, ignore_threshold:{cfg.yolo.ignore_threshold},"+\
                    f"lambda_conf:{cfg.yolo.lambda_conf}, lambda_no_conf:{cfg.yolo.lambda_no_conf},"+\
                    f"alpha:{cfg.yolo.alpha}, gamma:{cfg.yolo.gamma},lambda_cls: {cfg.yolo.lambda_cls}, iou_type:{cfg.yolo.iou_type}, failed"
                log.info(msg)
                del model,batch_loss,optimizer,train_loader,test_loader,criterion
                return - 10000000

        if cfg.metric =='mAP':
            results=test_one_epoch(test_loader,model,criterion,cfg)
            mAP=eval_results(results,dset_config['dset_name'],dset_config['val_annotations'])
            msg=f"RANK{rank}, lambda_xy:{cfg.yolo.lambda_xy}, lambda_wh: {cfg.yolo.lambda_wh},"+\
                    f"lambda_iou:{cfg.yolo.lambda_iou}, ignore_threshold:{cfg.yolo.ignore_threshold},"+\
                    f"lambda_conf:{cfg.yolo.lambda_conf}, lambda_no_conf:{cfg.yolo.lambda_no_conf},"+\
                    f"alpha:{cfg.yolo.alpha}, gamma:{cfg.yolo.gamma},lambda_cls: {cfg.yolo.lambda_cls}, iou_type:{cfg.yolo.iou_type}, mAP={mAP}"
            log.info(msg)
            del model,batch_loss,optimizer,train_loader,test_loader,criterion
            return mAP
        else:
            batch_loss = valid_one_epoch(test_loader,model,criterion)
            if torch.isnan(batch_loss):
                batch_loss = torch.tensor([1e8],device='cuda')
            valid_loss = -batch_loss.item()
            msg=f"RANK{rank}, lambda_xy:{cfg.yolo.lambda_xy}, lambda_wh: {cfg.yolo.lambda_wh},"+\
                    f"lambda_iou:{cfg.yolo.lambda_iou}, ignore_threshold:{cfg.yolo.ignore_threshold},"+\
                    f"lambda_conf:{cfg.yolo.lambda_conf}, lambda_no_conf:{cfg.yolo.lambda_no_conf},"+\
                    f"alpha:{cfg.yolo.alpha}, gamma:{cfg.yolo.gamma},lambda_cls: {cfg.yolo.lambda_cls}, iou_type:{cfg.yolo.iou_type}, val_loss={valid_loss}"
            log.info(msg)

            del model,batch_loss,optimizer,train_loader,test_loader,criterion
            return valid_loss
if __name__=='__main__':
    main()