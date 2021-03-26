# Loss functions
from typing import OrderedDict
import torch
import torch.nn as nn
from pycocotools.coco import COCO
from lvis import LVIS
import numpy as np
from scipy.sparse import coo_matrix, hstack,vstack
from sklearn.feature_extraction.text import TfidfTransformer
from tqdm import tqdm
import os
import pandas as pd
from torchvision.ops import FeaturePyramidNetwork 

def smooth_BCE(eps=0.1):  # https://github.com/ultralytics/yolov3/issues/238#issuecomment-598028441
    # return positive, negative label smoothing BCE targets
    return 1.0 - 0.5 * eps, 0.5 * eps


class BCEBlurWithLogitsLoss(nn.Module):
    # BCEwithLogitLoss() with reduced missing label effects.
    def __init__(self, alpha=0.05):
        super(BCEBlurWithLogitsLoss, self).__init__()
        self.loss_fcn = nn.BCEWithLogitsLoss(reduction='none')  # must be nn.BCEWithLogitsLoss()
        self.alpha = alpha

    def forward(self, pred, true):
        loss = self.loss_fcn(pred, true)
        pred = torch.sigmoid(pred)  # prob from logits
        dx = pred - true  # reduce only missing label effects
        # dx = (pred - true).abs()  # reduce missing label and false label effects
        alpha_factor = 1 - torch.exp((dx - 1) / (self.alpha + 1e-4))
        loss *= alpha_factor
        return loss.mean()


class FocalLoss(nn.Module):
    # Wraps focal loss around existing loss_fcn(), i.e. criteria = FocalLoss(nn.BCEWithLogitsLoss(), gamma=1.5)
    def __init__(self, loss_fcn, gamma=1.5, alpha=0.25):
        super(FocalLoss, self).__init__()
        self.loss_fcn = loss_fcn  # must be nn.BCEWithLogitsLoss()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = loss_fcn.reduction
        self.loss_fcn.reduction = 'none'  # required to apply FL to each element

    def forward(self, pred, true):
        loss = self.loss_fcn(pred, true)
        # p_t = torch.exp(-loss)
        # loss *= self.alpha * (1.000001 - p_t) ** self.gamma  # non-zero power for gradient stability

        # TF implementation https://github.com/tensorflow/addons/blob/v0.7.1/tensorflow_addons/losses/focal_loss.py
        pred_prob = torch.sigmoid(pred)  # prob from logits
        p_t = true * pred_prob + (1 - true) * (1 - pred_prob)
        alpha_factor = true * self.alpha + (1 - true) * (1 - self.alpha)
        modulating_factor = (1.0 - p_t) ** self.gamma
        loss *= alpha_factor * modulating_factor

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:  # 'none'
            return loss


class QFocalLoss(nn.Module):
    # Wraps Quality focal loss around existing loss_fcn(), i.e. criteria = FocalLoss(nn.BCEWithLogitsLoss(), gamma=1.5)
    def __init__(self, loss_fcn, gamma=1.5, alpha=0.25):
        super(QFocalLoss, self).__init__()
        self.loss_fcn = loss_fcn  # must be nn.BCEWithLogitsLoss()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = loss_fcn.reduction
        self.loss_fcn.reduction = 'none'  # required to apply FL to each element

    def forward(self, pred, true):
        loss = self.loss_fcn(pred, true)

        pred_prob = torch.sigmoid(pred)  # prob from logits
        alpha_factor = true * self.alpha + (1 - true) * (1 - self.alpha)
        modulating_factor = torch.abs(true - pred_prob) ** self.gamma
        loss *= alpha_factor * modulating_factor

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:  # 'none'
            return loss


class ClassAttention(nn.Module):
    def __init__(self,in_h,in_w,num_classes,num_anchors,reduction_ration=16,device='cuda'):
        super(ClassAttention, self).__init__()
        self.num_classes=num_classes
        self.num_anchors=num_anchors

        fcl=torch.nn.Sequential(
            torch.nn.AvgPool3d((num_anchors,in_h,in_w)),
            torch.nn.Softmax(dim=1),
            torch.nn.Flatten(-4),
            torch.nn.Linear(num_classes, reduction_ration),
            torch.nn.ReLU(),
            torch.nn.Linear(reduction_ration, num_classes),
            torch.nn.Sigmoid()
        )
        self.fcl=fcl.to(device)

    def forward(self, class_logits):
        bs = class_logits.shape[0]
        in_h = class_logits.shape[-2]
        in_w = class_logits.shape[-1]
        class_attention = self.fcl(class_logits.view(bs,self.num_anchors,self.num_classes+5, in_h, in_w).permute(0,2,1,3,4).contiguous().contiguous()[:,5:,:,:,:])

        # class_attention = class_logits)

        return class_attention


class IDFTransformer():
    def __init__(self,annfile,dset_name,device='cuda'):
        cwd = os.getenv('owd')
        annfile = os.path.join(cwd,annfile)
        idf_path = dset_name+"_files"
        idf_path = os.path.join(cwd,idf_path)
        self.device=device 
        if not os.path.exists(idf_path):
            os.mkdir(idf_path)
        if not os.path.exists(os.path.join(idf_path,'idf.csv')):
            df = pd.DataFrame()
            if dset_name=='coco':
                coco=COCO(annfile)
                ims=[]
                last_cat = coco.getCatIds()[-1]
                num_classes = last_cat +1 # for bg
                self.num_classes=num_classes
                for imgid in coco.getImgIds():
                    anids = coco.getAnnIds(imgIds=imgid)
                    categories=[]
                    for anid in anids:
                        categories.append(coco.loadAnns(int(anid))[0]['category_id'])
                    ims.append(categories)
            else:
                lvis=LVIS(annfile)
                ims=[]
                last_cat = lvis.get_cat_ids()[-1]
                num_classes = last_cat +1 # for bg
                self.num_classes=num_classes
                for imgid in lvis.get_img_ids():
                    anids = lvis.get_ann_ids(img_ids=[imgid])
                    categories=[]
                    for annot in lvis.load_anns(anids):
                        categories.append(annot['category_id'])
                    ims.append(categories)
            
            final=0
            k=0
            print('calculating idf ...')
            for im in tqdm(ims):
                cats = np.array(im,dtype=np.int)
                cats = np.bincount(cats,minlength=self.num_classes)
                cats=np.array([cats])
                cats = coo_matrix(cats)
                if k==0:
                    final= cats
                else:
                    final = vstack([cats, final])
                k=k+1

            mask = final.sum(axis=0)>0
            mask=mask.tolist()[0]
            final = final.tocsr()[:,mask]
            self.num_classes = final.shape[1]
            self.idf_transformer = TfidfTransformer()
            self.idf_transformer.fit(final)
            df['idf_weights']= self.idf_transformer.idf_
            df.to_csv(os.path.join(idf_path,'idf.csv'))
            self.idf_weights = torch.tensor(self.idf_transformer.idf_,dtype=torch.float,device=self.device)
            weight_norm = torch.norm(self.idf_weights)
            self.idf_weights = self.idf_weights / weight_norm
        else:
            df=pd.read_csv(os.path.join(idf_path,'idf.csv'))
            weights=np.array(df['idf_weights'])
            self.idf_weights = torch.tensor(weights,dtype=torch.float,device=self.device)
            weight_norm = torch.norm(self.idf_weights)
            self.idf_weights = self.idf_weights / weight_norm


    def __call__(self,target_classes): 
        target_classes=target_classes.cpu().numpy()
        tc= np.bincount(target_classes,minlength=self.num_classes)
        res = self.idf_transformer.transform(tc.reshape(1,-1))
        weights = np.squeeze(res.A)[target_classes]
        weights=torch.tensor(weights,device = self.device)

        return weights

class FPN(nn.Module):
    def __init__(self,channels,device='cuda'):
        super(FPN, self).__init__()
        self.m =FeaturePyramidNetwork([256, 512, 1024], channels).to(device)
    
    def forward(self, layers):
        x = OrderedDict()
        x['feat0'] = layers[2]
        x['feat1'] = layers[1]
        x['feat2'] = layers[0]
        mx = self.m(x)
        out=(mx['feat2'],mx['feat1'],mx['feat0'])

        return out