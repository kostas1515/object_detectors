from utilities import helper
import cv2
import torch
import imgaug as ia
from imgaug import parameters as iap
import imgaug.augmenters as iaa
from imgaug.augmentables.bbs import BoundingBox, BoundingBoxesOnImage
import numpy as np

class ResizeToTensor(object):
    """
    Image: Resizes and normalizes it to predefined yolo dimensions. Also it transposes it and adds extra dimension for batch.
    BOXES: Transformes them from absolute X0Y0WH to yolo dimension relative XcYcWH -> range: [0, 1]
    Also it calculates bbox area and put it in the sample
    """
    
    def __init__(self, scale):
        self.scale = scale

    def __call__(self, sample):
        
        img,target=sample
        targets={}
        img_size=torch.tensor(img.shape)

        bbs=torch.tensor(target['bbox'])
        labels=torch.tensor(target['category_id'],dtype=torch.int64)
        areas=torch.tensor(target['area'])
        img = cv2.resize(img, dsize=(self.scale, self.scale), interpolation=cv2.INTER_CUBIC)
        try:
            img =  img.transpose((2,0,1)) # H x W x C -> C x H x W
        except ValueError:
            img = cv2.cvtColor(img,cv2.COLOR_GRAY2RGB)
            img =  img.transpose((2,0,1))
            
        img = img/255.0       #Add a channel at 0 (for batch) | Normalise.
        img= torch.from_numpy(img).float()
        
        mean=torch.tensor([[[0.485, 0.456, 0.406]]]).T
        std=torch.tensor([[[0.229, 0.224, 0.225]]]).T
        img = (img-mean)/std
        img=img.unsqueeze(0) #Add a channel at 0 (for batch)
        
        bbs=helper.convert2_rel_xcycwh(bbs,img_size)

        targets["bbox"]=bbs
        targets["img_size"]=img_size
        targets["category_id"]=labels
        targets['area']=areas/(img_size[0]*img_size[1])
        targets['image_id']=torch.tensor(target['image_id'],dtype=torch.int64)
        sample=(img,targets)
        
        return sample



class COCO91_80():

    def __init__(self):
        data = {"0": 1, "1": 2, "2": 3, "3": 4, "4": 5, "5": 6, "6": 7, "7": 8, "8": 9, "9": 10, 
                "10": 11, "11": 13, "12": 14, "13": 15, "14": 16, "15": 17, "16": 18, "17": 19, "18": 20,
                "19": 21, "20": 22, "21": 23, "22": 24, "23": 25, "24": 27, "25": 28, "26": 31, "27": 32,
                "28": 33, "29": 34, "30": 35, "31": 36, "32": 37, "33": 38, "34": 39, "35": 40, "36": 41,
                "37": 42, "38": 43, "39": 44, "40": 46, "41": 47, "42": 48, "43": 49, "44": 50, "45": 51,
                "46": 52, "47": 53, "48": 54, "49": 55, "50": 56, "51": 57, "52": 58, "53": 59, "54": 60,
                "55": 61, "56": 62, "57": 63, "58": 64, "59": 65, "60": 67, "61": 70, "62": 72, "63": 73,
                "64": 74, "65": 75, "66": 76, "67": 77, "68": 78, "69": 79, "70": 80, "71": 81, "72": 82,
                "73": 84, "74": 85, "75": 86, "76": 87, "77": 88, "78": 89, "79": 90}


        self.data_inv = {v: int(k) for k, v in data.items()}
    
    def __call__(self,sample):
        img,targets=sample

        inv_l=[]
        for label in targets['category_id'].tolist():
            inv_l.append(self.data_inv[label])
        targets['category_id']=torch.tensor(inv_l,dtype=torch.int64)

        sample = img,targets
        return sample


class Class1_0():
    def __init__(self):
        pass
    
    def __call__(self,sample):
        img,targets=sample
        targets['category_id']-=1
        sample = img,targets
        return sample


class Augment(object):
    #use mosaic
    # TO DO FIX IT
    
    def __init__(self,num_of_augms=0):
        self.num_of_augms=num_of_augms
        self.aug=iaa.OneOf([
            iaa.Sequential([
                iaa.LinearContrast(alpha=(0.75, 1.5)),
                iaa.Fliplr(0.5)
            ]),
            iaa.Sequential([
                iaa.Grayscale(alpha=(0.1, 0.9)),
                iaa.Affine(
                translate_percent={"y": (-0.15, 0.15)}
            )
            ]),
            iaa.Sequential([
                iaa.LinearContrast((0.6, 1.4)),
                iaa.ShearX((-10, 10))
            ]),
            iaa.Sequential([
                iaa.GaussianBlur(sigma=(0, 1)),
                iaa.ShearY((-10, 10))
            ]),
            iaa.Sequential([
                iaa.Cutout(nb_iterations=(1, 2), size=0.1, squared=False),
                iaa.Multiply((0.8, 1.2), per_channel=0.25),
                iaa.Fliplr(0.5),
            ]),
            iaa.Sequential([
                iaa.LinearContrast((0.6, 1.4)),
                iaa.Affine(
                translate_percent={"x": (-0.25, 0.25)}
            )
            ]),
            iaa.Sequential([
                iaa.Cutout(nb_iterations=(1, 5), size=0.1, squared=False),
                iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 15), per_channel=0.5),
                iaa.Affine(
                scale={"x": (0.9, 1.1), "y": (0.9, 1.1)},
            )
            ]),
            iaa.Sequential([
                iaa.CoarseDropout((0.0, 0.05), size_percent=(0.02, 0.25)),
                iaa.GaussianBlur(sigma=(0, 3)),
                iaa.Affine(
                scale={"x": (0.8, 1.2), "y": (0.8, 1.2)}
            )
            ])
        ])
                        
    def __call__(self, sample):
        
        temp_img_,annot=sample
        temp_b_= annot['bbox']
        labels = annot['category_id']
        targets={}
        xmin=temp_b_[:,0]
        ymin=temp_b_[:,1]
        xmax=xmin+temp_b_[:,2]
        ymax=ymin+temp_b_[:,3]
        temp_b_ = np.stack((xmin, ymin, xmax, ymax)).T

        at_least_one_box=False
        
        
        
        while(at_least_one_box==False):
            bbs = BoundingBoxesOnImage([
            BoundingBox(x1=b[0], y1=b[1], x2=b[2], y2=b[3], label=l) for b,l in zip(temp_b_,labels)], shape=temp_img_.shape)
            
            image_aug, bbs_aug = self.aug(image=temp_img_, bounding_boxes=bbs)

            bbs_aug=bbs_aug.remove_out_of_image().clip_out_of_image()
            
            
            new_bboxes=bbs_aug.to_xyxy_array()
            new_labels=np.array([box.label for box in bbs_aug.bounding_boxes])
            
            if(new_labels.size>0):
                at_least_one_box=True
        
        xmin=new_bboxes[:,0]
        ymin=new_bboxes[:,1]
        w=new_bboxes[:,2] - xmin
        h=new_bboxes[:,3] - ymin
        areas = w*h

        targets["bbox"]=np.stack((xmin, ymin, w, h)).T
        targets["img_size"]=image_aug.size
        targets["category_id"]=new_labels
        targets['area']=areas
        targets['image_id']=np.array(annot['image_id'],dtype=np.int64)

        sample=(image_aug,targets)

        return sample