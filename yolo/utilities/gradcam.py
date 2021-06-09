import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
import cv2
from skimage import io

class YOLOCAM(nn.Module):
    def __init__(self,model):
        super(YOLOCAM, self).__init__()
        
        # get the pretrained VGG19 network
        self.backbone = model.backbone
        
        self.embedding0 = model.embedding0
        
        self.embedding1_cbl = model.embedding1_cbl
        
        self.embedding1_upsample=model.embedding1_upsample
        
        self.embedding1 = model.embedding1
        
        self.embedding2_cbl = model.embedding2_cbl
        
        self.embedding2_upsample=model.embedding2_upsample
        
        self.embedding2 = model.embedding2
        
        # placeholder for the gradients
        self.gradients = None
    
    # hook for the gradients of the activations
    def activations_hook(self, grad):
        self.gradients = grad
        
    def forward(self, x):
        def _branch(_embedding, _in):
            for i, e in enumerate(_embedding):
                _in = e(_in)
                if i == 4:
                    out_branch = _in
            return _in, out_branch
        #  backbone
        x2, x1, x0 = self.backbone(x)
        
        

        #  yolo branch 0
        out0, out0_branch = _branch(self.embedding0, x0)
        #  yolo branch 1
        x1_in = self.embedding1_cbl(out0_branch)
        x1_in = self.embedding1_upsample(x1_in)
        x1_in = torch.cat([x1_in, x1], 1)
        out1, out1_branch = _branch(self.embedding1, x1_in)
        #  yolo branch 2
        x2_in = self.embedding2_cbl(out1_branch)
        x2_in = self.embedding2_upsample(x2_in)
        x2_in = torch.cat([x2_in, x2], 1)
        out2, out2_branch = _branch(self.embedding2, x2_in)
#         pred= [out0, out1, out2]
#         final = [p.view(p.shape[0],3,85, p.shape[-1], p.shape[-1]).permute(0, 3, 4, 1, 2).contiguous() for p in pred]
#         final2 = torch.cat([f.view((f.shape[0],-1,f.shape[-1])) for f in final],dim=1)
        h2,h1,h0 = out2.register_hook(self.activations_hook),out1.register_hook(self.activations_hook),out0.register_hook(self.activations_hook)
    
        return out0, out1, out2
        
    
    # method for the gradient extraction
    def get_activations_gradient(self):
        return self.gradients
    
    # method for the activation exctraction
    def get_activations(self, x):
        def _branch(_embedding, _in):
            for i, e in enumerate(_embedding):
                _in = e(_in)
                if i == 4:
                    out_branch = _in
            return _in, out_branch
        #  backbone
        x2, x1, x0 = self.backbone(x)

        #  yolo branch 0
        out0, out0_branch = _branch(self.embedding0, x0)
        #  yolo branch 1
        x1_in = self.embedding1_cbl(out0_branch)
        x1_in = self.embedding1_upsample(x1_in)
        x1_in = torch.cat([x1_in, x1], 1)
        out1, out1_branch = _branch(self.embedding1, x1_in)
        #  yolo branch 2
        x2_in = self.embedding2_cbl(out1_branch)
        x2_in = self.embedding2_upsample(x2_in)
        x2_in = torch.cat([x2_in, x2], 1)
        out2, out2_branch = _branch(self.embedding2, x2_in)
        
        return out0, out1, out2
    
class GradCam():
    
    def __init__(self,yolo_cam,net_out,images,targets,config):
        self.yolo_cam = yolo_cam
        self.num_anchors = len(config.yolo.anchors)
        self.num_classes = config.dataset.num_classes
        self.inp_dim = config.dataset.inp_dim
        self.images  = images
        self.targets = targets
        self.prd = [p.view(p.shape[0],self.num_anchors,5+self.num_classes, p.shape[-1], p.shape[-1]).permute(0, 3, 4, 1, 2).contiguous() for p in net_out]
        self.img_path = '../../../../datasets/coco/val2017/'
        
    def get_index(self,k,scale,aspect,index):
        
        self.prd[scale][k,index[0],index[1],aspect,index[2]+5].backward(retain_graph=True) 
        
        gradients = self.yolo_cam.get_activations_gradient()
        
        pooled_gradients = torch.mean(gradients, dim=[0, 2, 3])
        
        activations = self.yolo_cam.get_activations(self.images.cuda())[scale].detach()
        for i in range(255):
            activations[:, i, :, :] *= pooled_gradients[i]

        heatmap = torch.mean(activations, dim=1).squeeze()
        heatmap = np.maximum(heatmap.cpu(), 0)
        
        
        # normalize the heatmap
        heatmap = heatmap.float()
        heatmap /= torch.max(heatmap)

        # draw the heatmap
#         plt.matshow(heatmap.squeeze())

        hm=heatmap.numpy()[k]
    
        target_id = str(self.targets[k]['image_id'].item())
        templetate= 12 - len(target_id)
        img = cv2.imread(self.img_path+templetate*'0'+target_id+'.jpg')
        img = cv2.resize(img, (self.inp_dim, self.inp_dim))
        hm = cv2.resize(hm, (self.inp_dim, self.inp_dim))
        hm = np.uint8(255 * hm)
        hm = cv2.applyColorMap(hm, cv2.COLORMAP_JET)
        superimposed_img = hm * 0.4 + img
        
        im2 = np.uint16(superimposed_img)[:,:,::-1] 
        io.imshow(im2)

        return im2,heatmap.squeeze()
        
        
    def get_all_activations(self,k,scale,aspect,indices,class_names):
        
        positions = np.concatenate([np.array(indices),np.array([class_names])])
        superimposed_img=0
        offset_color = positions.shape[1]
        for t in positions.T:
            
            self.prd[scale][k,t[0],t[1],aspect,t[2]+5].backward(retain_graph=True) 

            gradients = self.yolo_cam.get_activations_gradient()

            pooled_gradients = torch.mean(gradients, dim=[0, 2, 3])

            activations = self.yolo_cam.get_activations(self.images.cuda())[scale].detach()
            for i in range(255):
                activations[:, i, :, :] *= pooled_gradients[i]

            heatmap = torch.mean(activations, dim=1).squeeze()
            heatmap = np.maximum(heatmap.cpu(), 0)


            heatmap = heatmap.float()
            heatmap /= torch.max(heatmap)

            # draw the heatmap
    #         plt.matshow(heatmap.squeeze())

            hm=heatmap.numpy()[k]

            target_id = str(self.targets[k]['image_id'].item())
            templetate= 12 - len(target_id)
            img = cv2.imread(self.img_path+templetate*'0'+target_id+'.jpg')
            img = cv2.resize(img, (self.inp_dim, self.inp_dim))
            hm = cv2.resize(hm, (self.inp_dim, self.inp_dim))
            hm = np.uint8(255 * hm)
            hm = cv2.applyColorMap(hm, cv2.COLORMAP_JET)
            superimposed_img = superimposed_img + hm * (0.4/offset_color) 
        superimposed_img = superimposed_img +img
        im2 = np.uint16(superimposed_img)[:,:,::-1] 
        io.imshow(im2)

        return im2,heatmap.squeeze()