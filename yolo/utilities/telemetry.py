from skimage import io
import seaborn as sns
import torch
from torchvision.ops import boxes
from utilities import helper
import cv2
import numpy as np

class Telemetry():


    def __init__(self,net_out,image,targets,num_classes=80):

        self.image = image
        self.img_size=image.shape[-1]
        self.out = [n.detach() for n in net_out]
        self.targets = targets
        self.anchors =[[[116, 90], [156, 198], [373, 326]],
                        [[30, 61], [62, 45], [59, 119]],
                        [[10, 13], [16, 30], [33, 23]]]
        self.num_classes=num_classes
        self.bbox_attrs=self.num_classes+5
        self.num_anchors=len(self.anchors)
        self.true_pred=[]
        with open('./coco_files/coco.names') as f:
            self.coco_names=f.readlines()
            self.coco_names=[c.rstrip() for c in self.coco_names]

        for i in range(3):
            input=self.out[i]
            bs = input.size(0)
            in_h = input.size(2)
            in_w = input.size(3)

            stride_h = self.img_size / in_h
            stride_w = self.img_size / in_w
            scaled_anchors = [(a_w / stride_w, a_h / stride_h) for a_w, a_h in self.anchors[i]]

            prd = input.view(bs,self.num_anchors,self.bbox_attrs, in_h, in_w).permute(0, 1, 3, 4, 2).contiguous()
            # Get outputs
            prd[...,0] = torch.sigmoid(prd[..., 0])          # Center x
            prd[...,1] = torch.sigmoid(prd[..., 1])          # Center y
            prd[...,2] = prd[..., 2]                         # Width
            prd[...,3] = prd[..., 3]                         # Height
            prd[...,4] = torch.sigmoid(prd[..., 4])          # Conf
            prd[...,5:] = torch.sigmoid(prd[..., 5:])        # Cls pred.
            self.out[i]=prd

            FloatTensor = torch.cuda.FloatTensor if prd.is_cuda else torch.FloatTensor
            LongTensor = torch.cuda.LongTensor if prd.is_cuda else torch.LongTensor
            # Calculate offsets for each grid
            grid_x = torch.linspace(0, in_w-1, in_w).repeat(in_w, 1).repeat(
                bs * self.num_anchors, 1, 1).view(prd[...,0].shape).type(FloatTensor)
            grid_y = torch.linspace(0, in_h-1, in_h).repeat(in_h, 1).t().repeat(
                bs * self.num_anchors, 1, 1).view(prd[...,1].shape).type(FloatTensor)
            # Calculate anchor w, h
            anchor_w = FloatTensor(scaled_anchors).index_select(1, LongTensor([0]))
            anchor_h = FloatTensor(scaled_anchors).index_select(1, LongTensor([1]))
            anchor_w = anchor_w.repeat(bs, 1).repeat(1, 1, in_h * in_w).view(prd[...,2].shape)
            anchor_h = anchor_h.repeat(bs, 1).repeat(1, 1, in_h * in_w).view(prd[...,3].shape)
            # Add offset and scale with anchors
            pred_boxes = FloatTensor(prd[..., :4].shape)
            pred_boxes[..., 0] = prd[...,0].data + grid_x
            pred_boxes[..., 1] = prd[...,1].data + grid_y
            pred_boxes[..., 2] = torch.exp(prd[...,2].data) * anchor_w
            pred_boxes[..., 3] = torch.exp(prd[...,3].data) * anchor_h
            # Results
            _scale = torch.Tensor([stride_w, stride_h] * 2).type(FloatTensor)

            output = torch.cat((pred_boxes.view(bs, -1, 4) * _scale,
                                            prd[...,4].view(bs, -1, 1), prd[...,5:].view(bs, -1, self.num_classes)), -1)
            self.true_pred.append(output)



    def show_img(self,k):
        img2show = self.image[k]
        mean = torch.tensor([[[0.485, 0.456, 0.406]]]).T
        std = torch.tensor([[[0.229, 0.224, 0.225]]]).T
        img2show = img2show*std +mean
        img2show = img2show*255
        img2show =  img2show.transpose(0,1)
        img2show =  img2show.transpose(1,2)
        img2show = img2show.numpy().astype(int)
        io.imshow(img2show)

    def show_gt(self,k,color = (0,0,0),linelen=1):
        img2show = self.image[k]
        mean = torch.tensor([[[0.485, 0.456, 0.406]]]).T
        std = torch.tensor([[[0.229, 0.224, 0.225]]]).T
        img2show = img2show*std +mean
        img2show = img2show*255
        img2show =  img2show.transpose(0,1)
        img2show =  img2show.transpose(1,2)
        img2show = img2show.numpy().astype(np.uint8)

        im = img2show.copy()

        bbox=self.targets[k]['bbox'].cuda()*self.img_size
        cords=helper.get_abs_coord(bbox)      
        for cord in cords:
            pt1, pt2 = (cord[0], cord[1]), (cord[2], cord[3])
            
            pt1 = int(pt1[0]), int(pt1[1])
            pt2 = int(pt2[0]), int(pt2[1])
            
            im = cv2.rectangle(im.copy(), pt1, pt2, color, linelen)
        
        io.imshow(im)



    def vis_attrib(self,img,scale,aspect,attrib=4):
        attrib2show=self.out[scale][img,aspect,:,:,attrib].cpu().numpy()
        ax = sns.heatmap(attrib2show)

    def vis_class(self,img,scale,aspect):
        
        classes=self.out[scale][img,aspect,:,:,5:].max(axis=-1)

        labels=(classes[1]).cpu().detach().numpy()
        values=(classes[0]).cpu().detach().numpy()

        ax = sns.heatmap(values,annot=labels)



    def vis_iou(self,img,scale,aspect):
        output=self.true_pred[scale]
        bbox=self.targets[img]['bbox'].cuda()*self.img_size
        res=boxes.box_iou(helper.get_abs_coord(output[img,:,:4]),helper.get_abs_coord(bbox)).max(axis=1)[0]
        in_h=(output.shape[1]/self.num_anchors)**0.5
        in_h=int(in_h)
        vis_res=res.view(self.num_anchors,in_h,in_h)
        ax = sns.heatmap(vis_res[aspect].cpu().numpy())



    def vis_performance(self,img,scale,aspect):
        output=self.true_pred[scale]
        bbox=self.targets[img]['bbox'].cuda()*self.img_size
        res=boxes.box_iou(helper.get_abs_coord(output[img,:,:4]),helper.get_abs_coord(bbox)).max(axis=1)[0]
        in_h=(output.shape[1]/self.num_anchors)**0.5
        in_h=int(in_h)
        iou=res.view(self.num_anchors,in_h,in_h)[aspect]

        confidence=self.out[scale][img,aspect,:,:,4]
        classes=self.out[scale][img,aspect,:,:,5:].max(axis=-1)
        classes_values=classes[0]
        classes_labels=classes[1]
        gt_labels=self.targets[img]['category_id']

        mask=torch.stack([classes_labels==lab for lab in gt_labels])
        mask=mask.sum(axis=0)
        mask[mask>0]=1
        mask[mask==0]=-1

        performance_heatmap=confidence*iou*mask
        ax = sns.heatmap(performance_heatmap.cpu().numpy())

    
    def get_category_name(self,index):

        return self.coco_names[index]


    def draw_bbs(self,k,confidence=0.1,iou_threshold=0.5,color = (0,0,0),linelen=1):

        predictions=torch.cat(self.true_pred,axis=1)[k]
        predictions[:,:4]=helper.get_abs_coord(predictions[:,:4])

        score=predictions[:,4]*(predictions[:,5:].max(axis=1)[0])
        pred_mask=score>confidence
        pred_conf=predictions[pred_mask]

        indices=boxes.nms(pred_conf[:,:4],pred_conf[:,4],iou_threshold)
        pred_final=pred_conf[indices,:]


        img2show = self.image[k]
        mean = torch.tensor([[[0.485, 0.456, 0.406]]]).T
        std = torch.tensor([[[0.229, 0.224, 0.225]]]).T
        img2show = img2show*std +mean
        img2show = img2show*255
        img2show =  img2show.transpose(0,1)
        img2show =  img2show.transpose(1,2)
        img2show = img2show.numpy().astype(np.uint8)

        im = img2show.copy()
        
        cords=pred_final.clone().cpu().numpy()        
        for cord in cords:
            pt1, pt2 = (cord[0], cord[1]), (cord[2], cord[3])
            
            pt1 = int(pt1[0]), int(pt1[1])
            pt2 = int(pt2[0]), int(pt2[1])
            
            im = cv2.rectangle(im.copy(), pt1, pt2, color, linelen)
        
        io.imshow(im)