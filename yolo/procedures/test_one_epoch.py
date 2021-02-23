import torch
from utilities import helper
from torchvision.ops import boxes
import itertools

def test_one_epoch(dataloader,model,yolo_loss,cfg):
    confidence=cfg.yolo.inf_confidence
    iou_threshold=cfg.yolo.inf_iou_threshold
    inp_dim=yolo_loss.img_size
    model.eval()
    results = []
    dset_name = dataloader.dset_name
#     torch.backends.cudnn.benchmark = True
    with torch.no_grad():
        for batch_idx, (images, targets) in enumerate(dataloader):
            # measure data loading time
            
            images=images.to('cuda',non_blocking=True)
            targets = [{k: v.to('cuda',non_blocking=True) for k, v in t.items()} for t in targets]
            
            out=model(images)
            predictions=yolo_loss(out)

            predictions[:,:,:4]=helper.get_abs_coord(predictions[:,:,:4])
            score=predictions[:,:,4]*(predictions[:,:,5:].max(axis=2)[0])
            pred_mask=score>confidence
            pred_conf=[(predictions[e][m]) for e,m in enumerate(pred_mask)]
            indices=[boxes.nms(pred_conf[i][:,:4],pred_conf[i][:,4],iou_threshold) for i in range(len(pred_conf))]
            pred_final=[pred_conf[i][indices[i],:] for i in range(len(pred_conf))]
            pred_final=list(filter(lambda t:t.shape[0]!=0,pred_final))


            for i,atrbs in enumerate(pred_final):
                xmin=atrbs[:,0]/inp_dim * targets[i]['img_size'][1]
                ymin=atrbs[:,1]/inp_dim * targets[i]['img_size'][0]
                xmax=atrbs[:,2]/inp_dim * targets[i]['img_size'][1]
                ymax=atrbs[:,3]/inp_dim * targets[i]['img_size'][0]
                w=xmax-xmin
                h=ymax-ymin

                scores=(atrbs[:,4]*atrbs[:,5:].max(axis=1)[0]).tolist()
                labels=(atrbs[:,5:].max(axis=1)[1])
                if dset_name=='coco':
                    labels=helper.torch80_to_91(labels).tolist()
                elif dset_name=='lvis':
                    labels = (labels - 1).tolist()
                bboxes=torch.stack((xmin, ymin, w, h),axis=1)
                areas=(bboxes[:,2]*bboxes[:,3]).tolist()
                bboxes=bboxes.tolist()
                temp=[{'bbox':b,'area':a,
                                 'category_id':l,
                                 'score':s,
                                 'image_id':targets[i]['image_id'].item()}
                                  for b,a,l,s in zip(bboxes,areas,labels,scores)]

                results=list(itertools.chain(results, temp))

    return results