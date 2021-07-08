import math
import sys
sys.path.insert(1, '../') 
import time
import torch
from lvis import LVIS
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from lvis import LVISEval
from detection.coco_utils import LVISDetection,CocoDetection
from tvision import frcnn,retinanet,mask_rcnn,ssd
from detection import utils,transforms
from torch.utils.data import DataLoader
from collections import OrderedDict
import argparse
import itertools
import json
import os
import pandas as pd

@torch.no_grad()
def evaluate(model, data_loader, device):
    cpu_device = torch.device("cpu")
    model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'
    results = []
    for images, targets in metric_logger.log_every(data_loader, 100, header):
        images = list(img.to(device) for img in images)

        torch.cuda.synchronize()
        model_time = time.time()
        outputs = model(images)
        outputs = [{k: v.to(cpu_device) for k, v in t.items()} for t in outputs]
        non_empt_el= [i for i,t in enumerate(targets) if len(t)>0]
        targets= [t for i,t in enumerate(targets) if i in non_empt_el]
        outputs= [o for i,o in enumerate(outputs) if i in non_empt_el]
        for i,o in enumerate(outputs):
            o['boxes'][:,2:4] -= o['boxes'][:,0:2]
            areas = (o['boxes'][:,2] * o['boxes'][:,3]).tolist()
            boxes=o['boxes'].tolist()
            scores=o['scores'].tolist()
            labels=o['labels'].tolist()
            temp=[{'bbox':b,'area':a,
                                 'category_id':l,
                                 'score':s,
                                 'image_id':targets[i]['image_id']}
                                  for b,a,l,s in zip(boxes,areas,labels,scores)]
            results=list(itertools.chain(results, temp))
        evaluator_time = time.time()    
        model_time = time.time() - model_time
        evaluator_time = time.time() - evaluator_time
        metric_logger.update(model_time=model_time, evaluator_time=evaluator_time)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    # accumulate predictions from all images
    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='coco', help='dataset')
    parser.add_argument('--model', default='maskrcnn_resnet50_fpn', help='model')
    parser.add_argument('--device', default='cuda', help='device')
    parser.add_argument('-b', '--batch-size', default=1, type=int,
                        help='images per gpu, the total batch size is $NGPU x batch_size')

    parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')

    parser.add_argument('--confidence_threshold',type=float, default=0.05, help='used during inference')
    parser.add_argument('--iou_threshold',type=float, default=0.5, help='used during inference')
    parser.add_argument('--max-detections',type=int, default=100, help='used during inference')
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--tfidf', default=None, type=str, help='tfidf variant')

    args = parser.parse_args()
    exp_name = (args.resume).split("/")[-2]
    out_dir = os.path.join('../jsons',args.dataset,exp_name)
    utils.mkdir(out_dir)
        
    
    if args.dataset=='lvis':
        root='../../../../datasets/coco/'
        annotations="../../../../datasets/coco/annotations/lvis_v1_val.json"
        dset = LVISDetection(root,annotations,transforms=transforms.ToTensor())
        num_classes=1204
    elif args.dataset=='coco':
        root='../../../../datasets/coco/val2017'
        annotations="../../../../datasets/coco/annotations/instances_val2017.json"
        dset = CocoDetection(root,annotations,transforms=transforms.ToTensor())
        num_classes=91
    else:
        sys.exit("Dataset not recognisable")
        
    if (args.tfidf):
        tfidf= pd.read_csv(f'../{args.dataset}_files/idf_{num_classes}.csv')[args.tfidf]
        tfidf = torch.tensor(tfidf,device='cuda').unsqueeze(0)
    else:
        tfidf = torch.ones(num_classes,device='cuda',dtype=torch.float).unsqueeze(0)
    
    if args.model == 'fasterrcnn_resnet50_fpn':
        model = frcnn.fasterrcnn_resnet50_fpn(pretrained=False,num_classes=num_classes,tfidf=tfidf,
                                              box_score_thresh=float(args.confidence_threshold),
                                              box_nms_thresh=float(args.iou_threshold),
                                              box_detections_per_img=int(args.max_detections))
    elif args.model == 'retinanet_resnet50_fpn':
        model = retinanet.retinanet_resnet50_fpn(pretrained=False,num_classes=num_classes,tfidf=tfidf,
                                                 box_score_thresh=float(args.confidence_threshold),
                                                 box_nms_thresh=float(args.iou_threshold),
                                                 box_detections_per_img=int(args.max_detections))
    elif args.model == 'maskrcnn_resnet50_fpn':
        model = mask_rcnn.maskrcnn_resnet50_fpn(pretrained=False,num_classes=num_classes,tfidf=tfidf,
                                                box_score_thresh=float(args.confidence_threshold),
                                                box_nms_thresh=float(args.iou_threshold),
                                                box_detections_per_img=int(args.max_detections))
    elif args.model == 'ssd300_vgg16':
        model = ssd.ssd300_vgg16(pretrained=False,num_classes=num_classes,tfidf= tfidf)

    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu')
        model.load_state_dict(checkpoint['model'])
        print(f'checkpoint loaded from:{args.resume}')

    model.eval()
    model=model.cuda()


    test_loader = DataLoader(dataset=dset,batch_size=args.batch_size,
                                      shuffle=False,num_workers=args.workers,collate_fn=utils.collate_fn,
                                      pin_memory=True)
    
    results = evaluate(model, test_loader, args.device)
    res_path=os.path.join(out_dir,(args.resume).split("/")[-1].split(".")[0]+".json")
    json.dump(results, open(res_path, 'w'), indent=4)
    
    if args.dataset=='coco':
        cocoGt=COCO(annotations)
        try:
            cocoDt=cocoGt.loadRes(res_path)
        except IndexError:
            print('empty list return zero map')
        cocoDt.loadAnns()

        #  running evaluation
        cocoEval = COCOeval(cocoGt,cocoDt,'bbox')
        cocoEval.evaluate()
        cocoEval.accumulate()
        cocoEval.summarize()

        mAP=cocoEval.stats[0]

    elif(args.dataset=='lvis'):
        try:
            lvis_eval = LVISEval(annotations, res_path, 'bbox')
        except IndexError:
            print('empty list return zero map')
        lvis_eval.run()
        metrics=lvis_eval.get_results()
        lvis_eval.print_results()
        mAP=metrics['AP']