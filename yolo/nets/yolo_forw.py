import torch
from torch._C import device
import torch.nn as nn
import numpy as np
import math
import os
from utilities import custom, helper
import pandas as pd
from torchvision.ops import boxes


class YOLOForw(nn.Module):
    def __init__(self, config):
        super(YOLOForw, self).__init__()
        cfg=config.yolo
        self.anchors = config.dataset['anchors']
        self.num_anchors = len(self.anchors)
        self.num_classes = cfg['classes']
        self.bbox_attrs = 5 + self.num_classes
        self.img_size = cfg['img_size']

        self.ignore_threshold = cfg['ignore_threshold']
        self.lambda_iou = cfg['lambda_iou']

        self.lambda_xy = cfg['lambda_xy']
        self.lambda_wh = cfg['lambda_wh']
        self.lambda_conf = cfg['lambda_conf']
        self.lambda_no_conf = cfg['lambda_no_conf']
        self.lambda_cls = cfg['lambda_cls']
        self.reduction = cfg['reduction']

        #TFIDF
        self.device = torch.device('cuda')
        self.tfidf_norm = cfg.tfidf_norm
        self.tfidf_batch = cfg.tfidf_batch
        weights = torch.ones(self.num_classes,device=self.device)
        self.idf = custom.IDFTransformer(config.dataset.train_annotations,config.dataset.dset_name,device=self.device)
        self.idf_logits=torch.tensor(1).cuda()

        
        self.iou_type = cfg['iou_type']
        self.wh_loss = nn.MSELoss(reduction = self.reduction)
        self.xy_loss = nn.MSELoss(reduction = self.reduction)
        self.pobj_loss = nn.BCEWithLogitsLoss(reduction = self.reduction)
        self.pobj_loss = custom.FocalLoss(self.pobj_loss,gamma=cfg.gamma,alpha=cfg.alpha)
        self.nobj_loss = nn.BCEWithLogitsLoss(reduction = "None")
        self.nobj_loss = custom.FocalLoss(self.nobj_loss,gamma=cfg.gamma,alpha=cfg.alpha)


        if cfg.tfidf[0]==1:
            weights = self.idf.idf_weights[cfg.tfidf_variant]
            if (self.tfidf_norm != 0):
                weights = weights / torch.norm(weights, p=self.tfidf_norm)
        elif cfg.tfidf[0]==2: #effective class samples
            cwd = os.getenv('owd')
            beta = 0.9999
            cls_num_list = self.idf.idf_weights['instance_freq'].tolist()
            effective_num = 1.0 - np.power(beta, cls_num_list)
            per_cls_weights = (1.0 - beta) / np.array(effective_num)
            per_cls_weights = per_cls_weights / np.sum(per_cls_weights) * len(cls_num_list)
            weights = torch.FloatTensor(per_cls_weights).cuda()
            
        if cfg.tfidf[1]==1:
            self.idf_logits=self.idf.idf_weights[cfg.tfidf_variant]
            if (self.tfidf_norm != 0):
                self.idf_logits = self.idf_logits / \
                    torch.norm(self.idf_logits, p=self.tfidf_norm)
     
        if cfg.class_loss==0:
            self.class_loss = nn.BCEWithLogitsLoss(reduction = self.reduction,pos_weight=weights)
        elif cfg.class_loss==1:
            self.class_loss = nn.CrossEntropyLoss(reduction = self.reduction,weight=weights)
        elif cfg.class_loss == 2:
            self.class_loss = nn.BCEWithLogitsLoss(
                reduction=self.reduction, pos_weight=weights)
            self.class_loss = custom.EQLoss(
                self.class_loss, img_freq=self.idf.idf_weights['img_freq'], gamma=cfg.gamma, alpha=cfg.alpha)



    def forward(self, input, targets=None):
        raw_pred=[]
        cxypwh =[]
        inw_inh=[]
        strides=[]
        #TFIDF - Minibatch calculation
        if (self.tfidf_batch is True) & (targets is not None):
            self.idf_logits = self.idf(targets)
            if (self.tfidf_norm != 0):
                self.idf_logits = self.idf_logits / \
                    torch.norm(self.idf_logits, p=self.tfidf_norm)
        # no_obj_conf_weights=[] # that will be [16,4,1] for the scales of yolo
        for k,input in enumerate(input):
            bs = input.size(0)
            in_h = input.size(2)
            in_w = input.size(3)
            stride_h = self.img_size / in_h
            stride_w = self.img_size / in_w
            scaled_anchors = torch.tensor([(a_w / stride_w, a_h / stride_h) for a_w, a_h in self.anchors[k]],device=self.device)
            # no_obj_conf_weights.append((4**(2-k))*torch.ones([in_h*in_w*scaled_anchors.shape[0]],device=self.device))
            prediction = input.view(bs,scaled_anchors.shape[0],self.bbox_attrs, in_h, in_w).permute(0, 3, 4, 1, 2).contiguous()
            # prediction = prediction.permute(0,2,3,1, 4).contiguous()
            prediction = torch.reshape(prediction,[bs,-1,self.bbox_attrs]).contiguous()
            grid_x = torch.linspace(0, in_w-1, in_w,device=self.device).repeat(in_w, 1).repeat(scaled_anchors.shape[0], 1, 1).permute(1,2,0) + 0.5
            grid_y = torch.linspace(0, in_h-1, in_h,device=self.device).repeat(in_h, 1).t().repeat(scaled_anchors.shape[0], 1, 1).permute(1,2,0) + 0.5
            grid_x=torch.reshape(grid_x,[-1])/in_w
            grid_y=torch.reshape(grid_y,[-1])/in_h
            anchor_w = scaled_anchors[:,0]/in_w
            anchor_w = anchor_w.repeat(1, in_h * in_w)
            anchor_w=torch.reshape(anchor_w,[-1])
            anchor_h = scaled_anchors[:,1]/in_h
            anchor_h = anchor_h.repeat(1, in_h * in_w)
            anchor_h=torch.reshape(anchor_h,[-1])
            cxypwh.append(torch.stack((grid_x,grid_y,anchor_w,anchor_h),axis=1))
            raw_pred.append(prediction)
            inw_inh.append(torch.ones(grid_y.shape,device=self.device)*in_w)
        inw_inh=torch.cat(inw_inh,axis=0)
        raw_pred=torch.cat(raw_pred,axis=1)
        cxypwh=torch.cat(cxypwh,axis=0)

        if targets is not None:
            tgt,tcls,obj_mask, noobj_mask = self.get_target(targets, cxypwh, inw_inh, ignore_threshold=self.ignore_threshold)
            final=torch.cat([raw_pred[k,i] for k,i in enumerate(obj_mask)])
            true_pred,gt = self.transform_pred(final,tgt,cxypwh,inw_inh,obj_mask)
            iou = helper.bbox_iou(true_pred,gt,self.iou_type)
            
            
            loss_xy = self.lambda_xy * self.xy_loss(torch.sigmoid(final[:,:2]),tgt[:,:2])
            loss_wh = self.lambda_wh * self.wh_loss(final[:,2:4],tgt[:,2:4])
            pos_conf_loss = self.lambda_conf * self.pobj_loss(final[:,4],torch.ones(final.shape[0],device=self.device))
            no_obj = raw_pred[noobj_mask][:,4]

            neg_conf_loss = self.lambda_no_conf * self.nobj_loss(no_obj,torch.zeros(no_obj.shape,device=self.device))

            if (type(self.class_loss)==nn.modules.loss.CrossEntropyLoss):
                class_loss = self.lambda_cls * self.class_loss(self.idf_logits.unsqueeze(0)*final[:,5:], tcls.max(axis=1)[1])
            else:
                class_loss = self.lambda_cls * \
                    self.class_loss(self.idf_logits.unsqueeze(0)
                                    * final[:, 5:], tcls)

            if self.reduction =="sum":
                iou_loss = self.lambda_iou *  (1 - iou).sum()
                neg_conf_loss = neg_conf_loss.sum()
            else:
                iou_loss = self.lambda_iou *  (1 - iou).mean()
                neg_conf_loss = neg_conf_loss.mean()
            

            loss = loss_xy + loss_wh + iou_loss + pos_conf_loss + neg_conf_loss + class_loss 

            #stats
            stats=self.get_stats(true_pred,iou,no_obj,tcls)
            sub_losses=torch.stack([loss_xy.detach().clone(),loss_wh.detach().clone(),
                                    iou_loss.detach().clone(),pos_conf_loss.detach().clone(),
                                    neg_conf_loss.detach().clone(),class_loss.detach().clone()])
                                    
            if self.reduction =="sum":
                loss = loss / gt.shape[0]
                sub_losses= sub_losses / gt.shape[0]
                 
            return loss,sub_losses,stats
        else:
            strides = (self.img_size / inw_inh).unsqueeze(1)
            inw_inh = inw_inh.unsqueeze(1)
            xy=(torch.sigmoid(raw_pred[...,0:2]) + cxypwh[:,:2] * inw_inh - 0.5)*strides
            wh=torch.exp(raw_pred[...,2:4]) * cxypwh[:,2:4] * inw_inh * strides
            conf=torch.sigmoid(raw_pred[:,:,4:5])
            if (type(self.class_loss)==nn.modules.loss.CrossEntropyLoss):
                cls = torch.softmax(self.idf_logits.unsqueeze(0).unsqueeze(0)*raw_pred[:,:,5:],axis=2)
            else:
                cls = torch.sigmoid(self.idf_logits.unsqueeze(
                    0).unsqueeze(0)*raw_pred[:, :, 5:])
            output = torch.cat((xy,wh,conf,cls),axis=2)

            return output.data

    def get_target(self, targets, cxypwh, inw_inh, ignore_threshold=0.5):
        obj_mask=[]
        noobj_mask=[]
        tgt=[]
        tcls=[]
        for b,target in enumerate(targets):
            bbox=target['bbox']
            tcls.append(torch.nn.functional.one_hot(target['category_id'],self.num_classes).float())
            iou=helper.bbox_iou(bbox.unsqueeze(1),cxypwh.unsqueeze(0),iou_type=self.iou_type)
            iou_mask=iou.max(axis=1)[1]
            gt=cxypwh[iou_mask].cuda()
            in_wh=inw_inh[iou_mask]

            gx=(bbox[:,0] * in_wh)-(bbox[:,0] * in_wh).long()
            gy=(bbox[:,1] * in_wh)-(bbox[:,1] * in_wh).long()
            gx = torch.clamp(gx,0.0001,0.9999)
            gy = torch.clamp(gy,0.0001,0.9999)
            
            gw=torch.log(bbox[:,2]/gt[:,2] + 1e-16)
            gh=torch.log(bbox[:,3]/gt[:,3] + 1e-16)

            tgt.append(torch.stack([gx,gy,gw,gh],axis=1))
            nobj_mask=((iou<ignore_threshold).prod(axis=0))
            nobj_mask[iou_mask]=False
            noobj_mask.append(nobj_mask)
            obj_mask.append(iou_mask)

        noobj_mask=torch.stack(noobj_mask,axis=0).bool()
        tgt=torch.cat(tgt,axis=0)
        tcls=torch.cat(tcls,axis=0)
        return tgt,tcls,obj_mask, noobj_mask


    def transform_pred(self,raw_pred,tgt,cxypwh,inw_inh,mask):
        cxypwh=torch.cat([cxypwh[i] for k,i in enumerate(mask)])
        inw_inh=torch.cat([inw_inh[i] for k,i in enumerate(mask)])
        inw_inh=inw_inh.unsqueeze(1)
        strides = self.img_size / inw_inh

        xy=(torch.sigmoid(raw_pred[:,0:2]) + cxypwh[:,:2] * inw_inh - 0.5)*strides
        wh=torch.exp(raw_pred[:,2:4]) * cxypwh[:,2:4] * inw_inh * strides
        conf=torch.sigmoid(raw_pred[:,4:5])

        if (type(self.class_loss)==nn.modules.loss.CrossEntropyLoss):
            cls = torch.softmax(raw_pred[:,5:],axis=1)
        else:
            cls = torch.sigmoid(raw_pred[:,5:])
        true_pred = torch.cat((xy,wh,conf,cls),axis=1)
        
        xy = (tgt[:,:2] + cxypwh[:,:2] * inw_inh - 0.5) * strides
        wh = torch.exp(tgt[:,2:4]) * cxypwh[:,2:4] * inw_inh * strides
        gt = torch.cat([xy,wh],axis=1)

        return true_pred,gt

    def get_stats(self,tp,i,nobj,gt):

        labels = gt.clone().detach().bool()
        no_obj = nobj.clone().detach()
        iou = i.clone().detach()
        true_pred = tp.clone().detach()
        no_obj_conf = torch.sigmoid(no_obj).mean()
        avg_iou = iou.mean()
        pos_conf = true_pred[:,4].mean()

        pos_class=true_pred[:,5:][labels].mean()
        neg_class=true_pred[:,5:][~labels].mean()
        stats = torch.stack([avg_iou,pos_conf,no_obj_conf,pos_class,neg_class])
        
        
        return stats

    def set_img_size(self,img_size):
        self.img_size = img_size
