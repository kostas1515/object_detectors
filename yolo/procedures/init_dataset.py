from dsets import coco_dataset,lvis_dataset
from dsets.transformations import ResizeToTensor,COCO91_80,Class1_0
from torchvision import transforms 
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
import os
from utilities import helper


def get_dataloaders(cfg):
    cwd = os.getenv('owd')
    config=cfg['dataset']
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

    if dset_name=='coco':
        ds_train = coco_dataset.CocoDetection(root = root,
                                              annFile = train_annotations,
                                              subset=tr_subset,
                                              transform=transforms.Compose([ResizeToTensor(inp_dim),
                                                                           COCO91_80()]))
        
        ds_val = coco_dataset.CocoDetection(root = root,
                                            annFile = val_annotations,
                                            subset=ts_subset,
                                            transform=transforms.Compose([ResizeToTensor(inp_dim),
                                                                         COCO91_80()]))


    elif dset_name=='lvis':
        ds_train = lvis_dataset.LVISDetection(root = root,
                                              annFile = train_annotations,
                                              subset=tr_subset,
                                              transform=transforms.Compose([ResizeToTensor(inp_dim),
                                                                           Class1_0()]))
        
        ds_val = lvis_dataset.LVISDetection(root = root,
                                            annFile = val_annotations,
                                            subset=ts_subset,
                                            transform=transforms.Compose([ResizeToTensor(inp_dim),
                                                                         Class1_0()]))     
        
    try:
        train_sampler = DistributedSampler(ds_train)
        test_sampler = DistributedSampler(ds_val)

        train_loader = DataLoader(dataset=ds_train,batch_size=tr_batch_size,
                                  shuffle=False,num_workers=num_workers,collate_fn=helper.collate_fn,
                                  pin_memory=True,sampler=train_sampler, multiprocessing_context='fork')

        test_loader = DataLoader(dataset=ds_val,batch_size=ts_batch_size,
                                  shuffle=False,num_workers=num_workers,collate_fn=helper.collate_fn,
                                  pin_memory=True,sampler=test_sampler,multiprocessing_context='fork')
    except AssertionError:
        train_loader = DataLoader(dataset=ds_train,batch_size=tr_batch_size,
                                  shuffle=True,num_workers=num_workers,collate_fn=helper.collate_fn,
                                  pin_memory=True,multiprocessing_context='fork')

        test_loader = DataLoader(dataset=ds_val,batch_size=ts_batch_size,
                                 shuffle=False,num_workers=num_workers,collate_fn=helper.collate_fn,
                                 pin_memory=True,multiprocessing_context='fork')


    test_loader.dset_name = dset_name

    return train_loader,test_loader