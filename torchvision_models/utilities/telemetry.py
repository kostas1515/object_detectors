from skimage import io
import torch
import cv2
import numpy as np

class Telemetry():
    def __init__(self,out,images,targets,dset_name='coco'):
        self.out = out
        self.targets = targets
        self.images=[]
        for i in images:
            i=i.cpu()
            i = i*255
            i =  i.transpose(0,1)
            i =  i.transpose(1,2)
            i = i.numpy().astype(np.uint8)
            self.images.append(i)
        if dset_name=='coco':
            with open('../coco_files/coco.names') as f:
                self.cat_names=f.readlines()
                self.cat_names=[c.rstrip() for c in self.cat_names]
        elif dset_name=='lvis':
            with open('../lvis_files/lvis.names') as f:
                self.cat_names=f.readlines()
                self.cat_names=[c.rstrip() for c in self.cat_names]

    def show_im(self,k):
        img2show = self.images[k]
        im = img2show.copy()
        io.imshow(im)
        
    def show_gt(self,k,color = (0,0,0),linelen=1,show_labels=True):
        img2show = self.images[k]
        im = img2show.copy()
        img_size=im.shape[1]

        bbox=np.stack([t['bbox'] for t in self.targets[k]])
        bbox[:,2:4] += bbox[:,:2]
        labels = np.array([t['category_id'] for t in self.targets[k]]) -1
        k=0
        for cord in bbox:
            pt1, pt2 = (cord[0], cord[1]), (cord[2], cord[3])

            pt1 = int(pt1[0]), int(pt1[1])
            pt2 = int(pt2[0]), int(pt2[1])

            im = cv2.rectangle(im.copy(), pt1, pt2, color, linelen)
            lb= labels[k]
            if show_labels is True:
                lb= self.get_category_name(labels[k])
                im = cv2.putText(im.copy(), lb, pt1,0,1e-3 * img_size, color, linelen//2)
            k=k+1

        io.imshow(im)
        
    def draw_bbs(self,k,color = (0,0,0),linelen=1,show_labels=True):
        predictions=self.out[k]
        img2show = self.images[k]
        im = img2show.copy()
        img_size=im.shape[1]
        
        cords=predictions['boxes'].detach().numpy()
        labels = predictions['labels'].detach().numpy() -1
        k=0
        for cord in cords:
            pt1, pt2 = (cord[0], cord[1]), (cord[2], cord[3])
            
            pt1 = int(pt1[0]), int(pt1[1])
            pt2 = int(pt2[0]), int(pt2[1])
            
            im = cv2.rectangle(im.copy(), pt1, pt2, color, linelen)
            if show_labels is True:
                lb= self.get_category_name(labels[k])
                im = cv2.putText(im.copy(), lb, pt1, 0, 1e-3 * img_size, color, linelen//2)
            k=k+1
        
        io.imshow(im)
        
    def get_category_name(self,index):
        
        return self.cat_names[index]
        