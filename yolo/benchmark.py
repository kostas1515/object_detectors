from torch.utils.data import DataLoader
import torch
from nets.yolohead import YoloHead
from torchvision.ops import boxes
from nets.yolo_loss import YOLOLoss
import sys
import json
from utilities import telemetry
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from dsets import coco_dataset,transformations
from torchvision import transforms
from utilities import helper
import time
import pandas as pd

for kk in range(2):
    batch_list=[2,4,8,16,32,64,128]
    forw_timings=[]
    cpu_timings=[]
    app_timings=[]
    map_per_batch=[]

    for batch in batch_list:

        app_dur=time.time()
        annotations="../../../datasets/coco/annotations/instances_val2017.json"
        root='../../../datasets/coco/images'
        inp_dim=416

        cc_val = coco_dataset.CocoDetection(root = root,
                            annFile = annotations,
                            subset=1,
                            transform=transforms.Compose([
                                transformations.ResizeToTensor(inp_dim),
                                transformations.COCO91_80()
                                                ]))

        test_dataloader=DataLoader(cc_val,batch_size=batch,shuffle=False, num_workers=2,collate_fn=helper.collate_fn)
        test_dataloader.dset_name='coco'


        config = {"backbone": {"backbone_name": "darknet_53",
                                "backbone_pretrained": "./weights/darknet53_weights_pytorch.pth"},
                "yolo": {"anchors": [[[116, 90], [156, 198], [373, 326]],
                            [[30, 61], [62, 45], [59, 119]],
                            [[10, 13], [16, 30], [33, 23]]],
                        "classes": 80}
                }


        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = YoloHead(config)
        model.load_state_dict(torch.load('weights/yolov3_orig.pth'))
        model=model.to(device)
        model.train()
        losses=[]
        cc_res=[]
        confidence=0.1
        iou_threshold=0.6

        for i in range(3):
            losses.append(YOLOLoss(config['yolo']['anchors'][i],80,[416,416]))

        forw_sum=0
        with torch.no_grad():
            for batch_idx, (images, targets) in enumerate(test_dataloader):
                # measure data loading time

                images = images.to(device)
            
                targets2=[]
                for t in targets:
                    dd={}
                    for k, v in t.items():
                        if(k!='img_size'):
                            dd[k]=v.to(device)
                        else:
                            dd[k]=v
                    targets2.append(dd)

        #         targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
                targets=targets2

                forw_dur=time.time()
                out=model(images)
                forw_dur=time.time() - forw_dur
                forw_sum=forw_sum+forw_dur

                outcome=[]
                for k,y in enumerate(out):
                    outcome.append(losses[k](y))

                predictions=torch.cat(outcome,axis=1)
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

                    scores=atrbs[:,4]*atrbs[:,5:].max(axis=1)[0]
                    labels=(atrbs[:,5:].max(axis=1)[1])
                    bboxes=torch.stack((xmin, ymin, w, h),axis=1)

                    for k in range(bboxes.shape[0]):
                        out={}
                        out['bbox']=(bboxes[k,:4]).cpu().tolist()
                        out['area']=(bboxes[k,2]*bboxes[k,3]).item()
                        out['category_id']=helper.coco80_to_coco91_class(labels[k].item())
                        out['score']=(scores[k]).item()
                        out['image_id']=targets[i]['image_id'].item()
                        cc_res.append(out)
                    
        app_dur=time.time()-app_dur     

        cpu_time=time.time()
        json.dump(cc_res, open('./{}_bbox_results.json'.format('cc'), 'w'), indent=4)
        cocoGt=COCO('../../../datasets/coco/annotations/instances_val2017.json')

        resFile='./cc_bbox_results.json'
        cocoDt=cocoGt.loadRes(resFile)
        imgIds=sorted(cocoGt.getImgIds())
        cocoDt.loadAnns()


        # # running evaluation
        cocoEval = COCOeval(cocoGt,cocoDt,'bbox')
        cocoEval.evaluate()
        cocoEval.accumulate()
        cocoEval.summarize()
        cpu_time=time.time()-cpu_time

        print(f'Forw_time(gpu):{forw_sum}, cpu_part:{cpu_time}, app_time:{app_dur}')
        forw_timings.append(forw_sum)
        cpu_timings.append(cpu_time)
        app_timings.append(app_dur)
        map_per_batch.append(cocoEval.stats[0])

    df=pd.DataFrame()
    df['batch']=batch_list
    df['forw_time']=forw_timings
    df['cpu_time']=cpu_timings
    df['app_time']=app_timings
    df['map']=map_per_batch
    df.to_csv('benchmark_res.csv')