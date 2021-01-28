from utilities import helper
import cv2
import torch

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
        
        
        bbs=torch.tensor([t['bbox'] for t in target])
        labels=torch.tensor([t['category_id'] for t in target],dtype=torch.int64)
        areas=torch.tensor([t['area'] for t in target])
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
        targets['image_id']=torch.tensor(target[0]['image_id'],dtype=torch.int64)
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