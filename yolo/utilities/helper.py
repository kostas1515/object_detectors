import torch
import numpy as np
import math
import os
import pandas as pd
from datetime import datetime

def coco80_to_coco91_class(label):
    x= [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 27, 28, 31, 32, 33, 34,
         35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63,
         64, 65, 67, 70, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 84, 85, 86, 87, 88, 89, 90]
    
    return x[int(label)]


def torch80_to_91(label):
    '''
    Input should be tensor
    '''
    x= torch.tensor([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 27, 28, 31, 32, 33, 34,
                     35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63,
                     64, 65, 67, 70, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 84, 85, 86, 87, 88, 89, 90])
    
    return x[label]



def get_idf(targets):
    num_of_images=len(targets)
    idf=torch.zeros(1203)
    for t in targets:
        label=torch.unique(t['category_id'])
        idf=idf+torch.bincount(label,minlength=1203)
        
    idf[idf==0]=num_of_images
    idf=torch.log(num_of_images/(idf+1))+1 #smooth
    idf=idf.cuda()
    
    
    return idf


def get_precomputed_idf(obj_idf,col_name):
    
    idf=np.array(obj_idf[col_name])
    idf=torch.tensor(idf,device='cuda',dtype=torch.float)
    idf=-torch.log(idf)
    
    
    return idf

def get_location_weights(gt_boxes):
    area=gt_boxes[:,2]*gt_boxes[:,3]
    weights=-torch.log(area)
    return torch.tensor(weights,device='cuda')


def dic2tensor(targets,key):
    
    tensor=torch.cat([t[key] for t in targets],dim=0)
    
    return tensor

def convert2onehot(labels):
    onehot=torch.zeros([labels.shape[0],1203],dtype=torch.float).cuda()
    onehot.zero_()
    onehot.scatter_(1, labels.unsqueeze(1), 1)
    return onehot.cuda()

def write_progress_stats(avg_losses,avg_stats,metrics,epoch):
    progress_path='progress'
    file_name=os.path.join(progress_path,'progress.csv')
    progress = {"Timestamp":datetime.now(),
                "Epoch":epoch,
                "Loss":avg_losses.sum(),
                "xy":avg_losses[0],
                "wh":avg_losses[1],
                "iou_loss":avg_losses[2],
                "pos_conf_loss":avg_losses[3],
                "neg_conf_loss":avg_losses[4],
                "class_loss":avg_losses[5],
                'iou':avg_stats[0],
                'pos_conf':avg_stats[1],
                'neg_conf':avg_stats[2],
                'pos_class':avg_stats[3],
                'neg_class':avg_stats[4],
                "mAP":metrics['mAP'],
                "val_loss": metrics['val_loss']}

    df= pd.DataFrame([progress])
    if os.path.exists(progress_path):
        df0 = pd.read_csv(file_name)
        pd.concat([df0,df],ignore_index=True).to_csv(file_name,index=False)
    else:
        os.mkdir(progress_path)
        df.to_csv(file_name,index=False)

    


def get_progress_stats(true_pred,no_obj,iou_list,targets):
    '''
    this function takes the Tranformed true prediction and the IoU list between the true prediction and the Gt boxes.
    Based on the IoU list it will calculate the mean IoU, the mean Positive Classification,  the mean negative Classification,
    the mean Positive objectness and the mean negative objectness.
    INPUTS: True_pred = Tensor:[N,BBs,4+1+C]
            no_obj_conf= Tensor [K]
            Targets = List of DICTs(N elements-> Key:Tenor[M,x]) , where M is the number of objects for that N, x depends on key.
            IoU List= List(N elements->Tensors[M,BBs])
    Outputs:DICT:{floats: avg_pos_conf, avg_neg_conf, avg_pos_class, avg_neg_class, avg_iou}  
    '''

    resp_true_pred=[]
    best_iou=[]
    no_obj2=no_obj.clone().detach()
    labels=dic2tensor(targets,'category_id')
    
    for i in range(len(iou_list)):
        best_iou_positions=iou_list[i].max(axis=1)[1]
        best_iou.append(iou_list[i].max(axis=1)[0])
        
        resp_true_pred.append(true_pred[i,:,:][best_iou_positions])
        
    resp_true_pred=torch.cat(resp_true_pred,dim=0)
    best_iou=torch.cat(best_iou,dim=0)
    
    
    nb_digits = resp_true_pred.shape[1]-5 # get the number of CLasses
    
    n_obj=resp_true_pred.shape[0]
    y = labels.view(-1,1)
    y_onehot = torch.BoolTensor(n_obj, nb_digits)
    y_onehot.zero_()
    y_onehot.scatter_(1, y, 1)
    
    pos_class=resp_true_pred[:,5:][y_onehot].mean().item()
    neg_class=resp_true_pred[:,5:][~y_onehot].mean().item()
    
    pos_conf=resp_true_pred[:,4].mean().item()
    avg_iou=best_iou.mean().item()
    
    neg_conf=torch.sigmoid(no_obj2).mean().item()
    
    return  {'iou':avg_iou,
             'pos_conf':pos_conf,
             'neg_conf':neg_conf,
             'pos_class':pos_class,
             'neg_class':neg_class}
    


def collate_fn(batch):
    pictures=[i[0] for i in filter(None,batch)]
    if pictures==[]:
        return (None,None)
    pictures=torch.cat(pictures, dim=0)
    
    targets=[i[1] for i in filter(None,batch)]
        
    return pictures,targets


def convert2_abs_xyxy(bboxes,shape,inp_dim=1):
        
    (h,w)=shape[:2]
    h=h/inp_dim
    w=w/inp_dim
    
    
    xmin=bboxes[:,0]*w
    ymin=bboxes[:,1]*h
    width=bboxes[:,2]*w
    height=bboxes[:,3]*h
    xmin=xmin-width/2
    ymin=ymin-height/2
    xmax=xmin+width
    ymax=ymin+height
    
    if (type(bboxes) is torch.Tensor):
        return torch.stack((xmin, ymin, xmax, ymax)).T
    else:
        return np.stack((xmin, ymin, xmax, ymax)).T



def convert2_rel_xcycwh(bboxes,shape):
        
    (h,w)=shape[:2]    
    xmin=bboxes[:,0]/w
    ymin=bboxes[:,1]/h
    width=bboxes[:,2]/w
    height=bboxes[:,3]/h
        
    xc = xmin+(width)/2 
    yc = ymin+(height)/2 
    
    if (type(bboxes) is torch.Tensor):
        return torch.stack((xc, yc, width, height)).T
    else:
        return np.stack((xc, yc, width, height)).T


def get_abs_coord(box):
    if torch.cuda.is_available():
        box=box.cuda()
    if (len(box.shape)==3):
        x1 = (box[:,:,0] - box[:,:,2]/2) 
        y1 = (box[:,:,1] - box[:,:,3]/2) 
        x2 = (box[:,:,0] + box[:,:,2]/2) 
        y2 = (box[:,:,1] + box[:,:,3]/2)
        return torch.stack((x1, y1, x2, y2),axis=2)
    else:
        x1 = (box[:,0] - box[:,2]/2) 
        y1 = (box[:,1] - box[:,3]/2) 
        x2 = (box[:,0] + box[:,2]/2) 
        y2 = (box[:,1] + box[:,3]/2)
        return torch.stack((x1, y1, x2, y2),axis=1)



def bbox_iou(bb1, bb2,iou_type,CUDA=True,xcycwh=True):
    # Returns the IoU of box1 to box2. box1 is 4, box2 is nx4

    if iou_type == 1:
        GIoU, DIoU, CIoU = (True,False,False)
    elif(iou_type == 2):
        GIoU, DIoU, CIoU = (False,True,False)
    elif(iou_type == 3):
        GIoU, DIoU, CIoU = (False,False,True)
    else:
        GIoU, DIoU, CIoU = (False,False,False)

    if xcycwh is True:
        box1=get_abs_coord(bb1)
        box2=get_abs_coord(bb2)
    else:
        box1 = bb1
        box2 = bb2
    
    if CUDA:
        box2 = box2.cuda()
        box1 = box1.cuda()

    # Get the coordinates of bounding boxes
    b1_x1, b1_y1, b1_x2, b1_y2 = box1[...,0], box1[...,1], box1[...,2], box1[...,3]
    b2_x1, b2_y1, b2_x2, b2_y2 = box2[...,0], box2[...,1], box2[...,2], box2[...,3]

    # Intersection area
    inter = (torch.min(b1_x2, b2_x2) - torch.max(b1_x1, b2_x1)).clamp(0) * \
            (torch.min(b1_y2, b2_y2) - torch.max(b1_y1, b2_y1)).clamp(0)

    # Union Area
    w1, h1 = b1_x2 - b1_x1, b1_y2 - b1_y1
    w2, h2 = b2_x2 - b2_x1, b2_y2 - b2_y1
    union = (w1 * h1 + 1e-16) + w2 * h2 - inter

    iou = inter / union  # iou
    if GIoU or DIoU or CIoU:
        cw = torch.max(b1_x2, b2_x2) - torch.min(b1_x1, b2_x1)  # convex (smallest enclosing box) width
        ch = torch.max(b1_y2, b2_y2) - torch.min(b1_y1, b2_y1)  # convex height
        if GIoU:  # Generalized IoU https://arxiv.org/pdf/1902.09630.pdf
            c_area = cw * ch + 1e-16  # convex area
            return iou - (c_area - union) / c_area  # GIoU
        if DIoU or CIoU:  # Distance or Complete IoU https://arxiv.org/abs/1911.08287v1
            # convex diagonal squared
            c2 = cw ** 2 + ch ** 2 + 1e-16
            # centerpoint distance squared
            rho2 = ((b2_x1 + b2_x2) - (b1_x1 + b1_x2)) ** 2 / 4 + ((b2_y1 + b2_y2) - (b1_y1 + b1_y2)) ** 2 / 4
            if DIoU:
                return iou - rho2 / c2  # DIoU
            elif CIoU:  # https://github.com/Zzh-tju/DIoU-SSD-pytorch/blob/master/utils/box/box_utils.py#L47
                v = (4 / math.pi ** 2) * torch.pow(torch.atan(w2 / h2) - torch.atan(w1 / h1), 2)
                with torch.no_grad():
                    alpha = v / (1 - iou + v)
                return iou - (rho2 / c2 + v * alpha)  # CIoU

    return iou

    
def nms_majority(P,thresh_iou=0.6):
    """
    Apply non-maximum suppression to avoid detecting too many
    overlapping bounding boxes for a given object.
    Args:
        boxes: (tensor) The location preds for the image 
            along with the class predscores, Shape: [num_boxes,5].
        thresh_iou: (float) The overlap thresh for suppressing unnecessary boxes.
    Returns:
        A list of filtered boxes, Shape: [ , 5]
    """
    # we extract coordinates for every 
    # prediction box present in P
    x1 = P[:, 0]
    y1 = P[:, 1]
    x2 = P[:, 2]
    y2 = P[:, 3]

    # we extract the confidence scores as well
    scores = P[:, 4]
    
    classes = (P[:, 5]).int()

    # calculate area of every block in P
    areas = (x2 - x1) * (y2 - y1)
    
    # sort the prediction boxes in P
    # according to their confidence scores
    order = scores.argsort()

    # initialise an empty list for 
    # filtered prediction boxes
    keep = []
    

    while len(order) > 0:
        
        # extract the index of the 
        # prediction with highest score
        # we call this prediction S
        idx = order[-1]


        # remove S from P
        order = order[:-1]
        
        keep.append(P[idx])
        # sanity check
        if len(order) == 0:
            break
        
        # select coordinates of BBoxes according to 
        # the indices in order
        xx1 = torch.index_select(x1,dim = 0, index = order)
        xx2 = torch.index_select(x2,dim = 0, index = order)
        yy1 = torch.index_select(y1,dim = 0, index = order)
        yy2 = torch.index_select(y2,dim = 0, index = order)

        # find the coordinates of the intersection boxes
        xx1 = torch.max(xx1, x1[idx])
        yy1 = torch.max(yy1, y1[idx])
        xx2 = torch.min(xx2, x2[idx])
        yy2 = torch.min(yy2, y2[idx])

        # find height and width of the intersection boxes
        w = xx2 - xx1
        h = yy2 - yy1
        
        # take max with 0.0 to avoid negative w and h
        # due to non-overlapping boxes
        w = torch.clamp(w, min=0.0)
        h = torch.clamp(h, min=0.0)

        # find the intersection area
        inter = w*h

        # find the areas of BBoxes according the indices in order
        rem_areas = torch.index_select(areas, dim = 0, index = order) 

        # find the union of every prediction T in P
        # with the prediction S
        # Note that areas[idx] represents area of S
        union = (rem_areas - inter) + areas[idx]
        
        # find the IoU of every prediction in P with S
        IoU = inter / union

        # keep the boxes with IoU less than thresh_iou
        mask = IoU < thresh_iou
        mask_neg = IoU > thresh_iou
        if (classes[order[mask_neg]].shape[0]>0):
            cats,cnts = torch.unique(classes[order[mask_neg]],return_counts=True)
            if(cnts.shape[0]>1):
                majority_vote=cats[cnts.max(axis=0)[1]].float()
                if (P[idx,5]!=majority_vote):
                    P[idx,5] = cats[cnts.max(axis=0)[1]].float()
            
        # push S in filtered predictions list
        

        order = order[mask]
    
    return torch.stack(keep,axis=0)