from dsets import coco_dataset,lvis_dataset
from dsets.transformations import ResizeToTensor,COCO91_80,Class1_0,Augment
from torchvision import transforms 
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
import os
from utilities import helper


def get_dataloaders(cfg):
    cwd = os.getenv('owd')
    config=cfg['dataset']
    augment=config.augment
    dset_name=config['dset_name']
    inp_dim=config['inp_dim']
    tr_batch_size=config['tr_batch_size']
    ts_batch_size=config['ts_batch_size']
    tr_subset=config['tr_subset']
    ts_subset=config['ts_subset']
    num_workers=config['num_workers']
    
    root= os.path.join(cwd,config['root'])
    train_annotations = os.path.join(cwd,config['train_annotations'])
    val_annotations = os.path.join(cwd,config['val_annotations'])
    ds_train=None
    ds_val=None
    transformationsList_train=[]
    transformationsList_test=[]
    
    if augment>0:
        transformationsList_train.append(Augment(augment))

    transformationsList_train.append(ResizeToTensor(inp_dim))
    transformationsList_test.append(ResizeToTensor(inp_dim))

    if dset_name=='coco':
        transformationsList_train.append(COCO91_80())
        transformationsList_test.append(COCO91_80())

        ds_train = coco_dataset.CocoDetection(root = root,
                                              annFile = train_annotations,
                                              subset=tr_subset,
                                              transform=transforms.Compose(transformationsList_train))
        
        ds_val = coco_dataset.CocoDetection(root = root,
                                            annFile = val_annotations,
                                            subset=ts_subset,
                                            transform=transforms.Compose(transformationsList_test))


    elif dset_name=='lvis':
        transformationsList_train.append(Class1_0())
        transformationsList_test.append(Class1_0())

        ds_train = lvis_dataset.LVISDetection(root = root,
                                              annFile = train_annotations,
                                              subset=tr_subset,
                                              transform=transforms.Compose(transformationsList_train))
        
        ds_val = lvis_dataset.LVISDetection(root = root,
                                            annFile = val_annotations,
                                            subset=ts_subset,
                                            transform=transforms.Compose(transformationsList_test))     
    if config.num_workers>0:
        mp_context='fork'
    else:
        mp_context=None
    if cfg.gpus>1:
        train_sampler = DistributedSampler(ds_train,num_replicas=cfg.gpus,rank=cfg.rank)
        test_sampler = DistributedSampler(ds_val,num_replicas=cfg.gpus,rank=cfg.rank)

        train_loader = DataLoader(dataset=ds_train,batch_size=tr_batch_size,
                                  shuffle=False,num_workers=num_workers,collate_fn=helper.collate_fn,
                                  pin_memory=True,sampler=train_sampler, multiprocessing_context=mp_context)

        test_loader = DataLoader(dataset=ds_val,batch_size=ts_batch_size,
                                  shuffle=False,num_workers=num_workers,collate_fn=helper.collate_fn,
                                  pin_memory=True,sampler=test_sampler,multiprocessing_context=mp_context)
    else:
        print("something went wrong with dist sampler, use standart loader")

        train_loader = DataLoader(dataset=ds_train,batch_size=tr_batch_size,
                                  shuffle=True,num_workers=num_workers,collate_fn=helper.collate_fn,
                                  pin_memory=True,multiprocessing_context=mp_context)

        test_loader = DataLoader(dataset=ds_val,batch_size=ts_batch_size,
                                 shuffle=False,num_workers=num_workers,collate_fn=helper.collate_fn,
                                 pin_memory=True,multiprocessing_context=mp_context)


    test_loader.dset_name = dset_name

    return train_loader,test_loader