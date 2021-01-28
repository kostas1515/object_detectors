import torch
import torch.nn as nn
import numpy as np
import math
from utilities import custom

from torchvision.ops import boxes


class YOLOLoss(nn.Module):
    def __init__(self, cfg,head):
        super(YOLOLoss, self).__init__()
        self.anchors = cfg['anchors'][head]
        self.num_anchors = len(self.anchors)
        self.num_classes = cfg['classes']
        self.bbox_attrs = 5 + self.num_classes
        self.img_size = cfg['img_size']

        self.ignore_threshold = cfg['ignore_threshold']
        self.lambda_xy = cfg['lambda_xy']
        self.lambda_wh = cfg['lambda_wh']
        self.lambda_conf = cfg['lambda_conf']
        self.lambda_no_conf = cfg['lambda_no_conf']
        self.lambda_cls = cfg['lambda_cls']


        self.device= torch.device('cuda')
        # self.iou_loss = custom.IoULoss()
        self.wh_loss = nn.MSELoss()
        self.xy_loss = nn.BCEWithLogitsLoss()
        self.conf_loss = nn.BCEWithLogitsLoss()
        self.class_loss = nn.BCEWithLogitsLoss()

    def forward(self, input, targets=None):
        bs = input.size(0)
        in_h = input.size(2)
        in_w = input.size(3)
        stride_h = self.img_size / in_h
        stride_w = self.img_size / in_w
        scaled_anchors = [(a_w / stride_w, a_h / stride_h) for a_w, a_h in self.anchors]

        prediction = input.view(bs,self.num_anchors,self.bbox_attrs, in_h, in_w).permute(0, 1, 3, 4, 2).contiguous()

        # Get outputs
        x = prediction[..., 0]                         # Center x
        y = prediction[..., 1]                         # Center y
        w = prediction[..., 2]                         # Width
        h = prediction[..., 3]                         # Height
        conf = prediction[..., 4]                      # Conf
        pred_cls = prediction[..., 5:]                 # Cls pred.

        if targets is not None:
            #  build target
            mask, noobj_mask, tx, ty, tw, th, tconf, tcls = self.get_target(targets, scaled_anchors,in_w, in_h,self.ignore_threshold)
            mask, noobj_mask = mask.cuda(), noobj_mask.cuda()
            mask = mask == 1
            noobj_mask = noobj_mask == 1
            tx, ty, tw, th = tx.cuda(), ty.cuda(), tw.cuda(), th.cuda()
            tconf, tcls = tconf.cuda(), tcls.cuda()
            #  losses.
            loss_x = self.xy_loss(x[mask], tx[mask])
            loss_y = self.xy_loss(y[mask], ty[mask])
            loss_w = self.wh_loss(w[mask], tw[mask])
            loss_h = self.wh_loss(h[mask], th[mask])
            loss_conf = self.lambda_conf * self.conf_loss(conf[mask],tconf[mask]) + \
                self.lambda_no_conf * self.conf_loss(conf[noobj_mask], tconf[noobj_mask] * 0.0)
            loss_cls = self.class_loss(pred_cls[mask == 1], tcls[mask == 1])
            #  total loss = losses * weight
            loss = loss_x * self.lambda_xy + loss_y * self.lambda_xy + \
                loss_w * self.lambda_wh + loss_h * self.lambda_wh + \
                loss_conf + loss_cls * self.lambda_cls

            return loss, loss_x.detach(), loss_y.detach(), loss_w.detach(),\
                loss_h.detach(), loss_conf.detach(), loss_cls.detach()
        else:
            x = torch.sigmoid(prediction[..., 0])
            y = torch.sigmoid(prediction[..., 1])
            w = prediction[..., 2]
            h = prediction[..., 3]
            conf = torch.sigmoid(prediction[..., 4])
            pred_cls = torch.sigmoid(prediction[..., 5:])

            FloatTensor = torch.cuda.FloatTensor if x.is_cuda else torch.FloatTensor
            LongTensor = torch.cuda.LongTensor if x.is_cuda else torch.LongTensor
            # Calculate offsets for each grid
            grid_x = torch.linspace(0, in_w-1, in_w).repeat(in_w, 1).repeat(
                bs * self.num_anchors, 1, 1).view(x.shape).type(FloatTensor)
            grid_y = torch.linspace(0, in_h-1, in_h).repeat(in_h, 1).t().repeat(
                bs * self.num_anchors, 1, 1).view(y.shape).type(FloatTensor)
            # Calculate anchor w, h
            anchor_w = FloatTensor(scaled_anchors).index_select(1, LongTensor([0]))
            anchor_h = FloatTensor(scaled_anchors).index_select(1, LongTensor([1]))
            anchor_w = anchor_w.repeat(bs, 1).repeat(1, 1, in_h * in_w).view(w.shape)
            anchor_h = anchor_h.repeat(bs, 1).repeat(1, 1, in_h * in_w).view(h.shape)
            # Add offset and scale with anchors
            pred_boxes = FloatTensor(prediction[..., :4].shape)
            pred_boxes[..., 0] = x.data + grid_x
            pred_boxes[..., 1] = y.data + grid_y
            pred_boxes[..., 2] = torch.exp(w.data) * anchor_w
            pred_boxes[..., 3] = torch.exp(h.data) * anchor_h
            # Results
            _scale = torch.Tensor([stride_w, stride_h] * 2).type(FloatTensor)
            output = torch.cat((pred_boxes.view(bs, -1, 4) * _scale,
                                conf.view(bs, -1, 1), pred_cls.view(bs, -1, self.num_classes)), -1)
            return output.data

    def get_target(self, targets, anchors, in_w, in_h, ignore_threshold=0.5):
        bs = len(targets)

        mask = torch.zeros(bs, 3, in_h, in_w, requires_grad=False,device=self.device)
        noobj_mask = torch.ones(bs, 3, in_h, in_w, requires_grad=False,device=self.device)
        tx = torch.zeros(bs, 3, in_h, in_w, requires_grad=False,device=self.device)
        ty = torch.zeros(bs, 3, in_h, in_w, requires_grad=False,device=self.device)
        tw = torch.zeros(bs, 3, in_h, in_w, requires_grad=False,device=self.device)
        th = torch.zeros(bs, 3, in_h, in_w, requires_grad=False,device=self.device)
        tconf = torch.zeros(bs, 3, in_h, in_w, requires_grad=False,device=self.device)
        tcls = torch.zeros(bs, 3, in_h, in_w, self.num_classes, requires_grad=False,device=self.device)

        for b,target in enumerate(targets):
            bbox=target['bbox'].cuda()
            categories=target['category_id'].cuda()

            gx = bbox[:,0] * in_w
            gy = bbox[:,1] * in_h
            gx = torch.clamp(gx,0,in_w-1e-4)
            gy = torch.clamp(gy,0,in_h-1e-4)

            gw = bbox[:,2] * in_w
            gh = bbox[:,3] * in_h
            #get crid coords
            gi = gx.long()
            gj = gy.long()

            gt_box = torch.zeros(bbox.shape,dtype=torch.float,requires_grad=False,device=self.device)
            gt_box[:,2] = gw
            gt_box[:,3] = gh

            anchor_shapes = torch.FloatTensor(np.concatenate((np.zeros((3, 2)),np.array(anchors)), 1)).cuda()

            anch_ious = boxes.box_iou(gt_box, anchor_shapes)

            # Where the overlap is larger than threshold set mask to zero (ignore)
            for i, anchor_ious in enumerate(anch_ious):
                noobj_mask[b, anchor_ious > ignore_threshold, gj[i], gi[i]] = 0
            # Find the best matching anchor box
            best_n = torch.max(anch_ious,axis=1)[1]

            # Masks
            mask[b, best_n, gj, gi] = True
            noobj_mask[b, best_n, gj, gi] = False
            # Coordinates
            tx[b, best_n, gj, gi] = gx - gi
            ty[b, best_n, gj, gi] = gy - gj
            # Width and height
            tw[b, best_n, gj, gi] = torch.log(gw/torch.tensor(anchors,device=self.device)[best_n][:,0] + 1e-16)
            th[b, best_n, gj, gi] = torch.log(gh/torch.tensor(anchors,device=self.device)[best_n][:,1] + 1e-16)
            # object
            tconf[b, best_n, gj, gi] = 1
            # One-hot encoding of label
            tcls[b, best_n, gj, gi, categories] = 1
        return mask, noobj_mask, tx, ty, tw, th, tconf, tcls
