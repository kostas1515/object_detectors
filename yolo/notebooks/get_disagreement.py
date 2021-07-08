import itertools
import sys,os
# import matplotlib.pyplot as plt
import argparse
# insert at 1, 0 is the script path (or '' in REPL)
sys.path.insert(1, '../')
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import os
import numpy as np
import json
import torch
from torchvision.ops import boxes
from statsmodels.stats.contingency_tables import mcnemar
from prettytable import PrettyTable

parser = argparse.ArgumentParser(description='get disagreement')

parser.add_argument('--compare_list', nargs='+', help='Experiments to compare')
parser.add_argument('--dset_name', type=str, help='Dataset name, either coco or lvis',default='coco')
parser.add_argument('--rids',  nargs='+', help='Columns to compare',default=[0])
parser.add_argument('--iou_threshold',  type=float, help='IoU threshold to compute TP',default=0.5)
parser.add_argument('--confidence',  type=float, help='Apply confidence thresholding',default=0.0)
parser.add_argument('--alpha',  type=float, help='Statistical Significance level',default=0.05)

args = parser.parse_args()

baseline=f'../output/{args.compare_list[0]}/bbox_results/coco/results_{args.rids[0]}.json'
res1=f'../output/{args.compare_list[1]}/bbox_results/coco/results_{args.rids[1]}.json'

if args.dset_name =='coco':
    gt = "../../../../datasets/coco/annotations/instances_val2017.json"
else:
    print('not implemented')
    sys.exit(0)

cocoGt=COCO(gt)
coco_det1=cocoGt.loadRes(res1)
coco_det1.loadAnns()
coco_base=cocoGt.loadRes(baseline)
coco_base.loadAnns()

def convert_gt_to_numpy(input_list):
    res = np.array([list(itertools.chain(il['bbox'],[float(il['category_id'])])) for il in input_list])
    res[:,2:4]+=res[:,0:2]
    return res

def convert_dt_to_numpy(input_list):
    res = np.array([list(itertools.chain(il['bbox'],[il['score'],float(il['category_id'])])) for il in input_list])
    res[:,2:4]+=res[:,0:2]
    return res
img_ids = cocoGt.getImgIds()
conf=args.confidence
correct1=0
correct2=0
yes_yes=0
no_yes=0
yes_no=0
no_no=0
missed=0
iou_true_positives=args.iou_threshold
total_val_instances=0
for img_id in img_ids:
    try:
        gt_ann = convert_gt_to_numpy(cocoGt.loadAnns(cocoGt.getAnnIds(img_id)))
        gt_ann=torch.tensor(gt_ann)
        base_ann = convert_dt_to_numpy(coco_base.loadAnns(coco_base.getAnnIds(img_id)))

        det1_ann = convert_dt_to_numpy(coco_det1.loadAnns(coco_det1.getAnnIds(img_id)))

        det1_ann=torch.tensor(det1_ann[det1_ann[:,4]>conf])
        base_ann=torch.tensor(base_ann[base_ann[:,4]>conf])

        iou1=boxes.box_iou(base_ann[:,:4],gt_ann[:,:4])
        iou_values,iou_indices  = iou1.max(axis=0)[0],iou1.max(axis=0)[1]
        mask1 = iou_values>=iou_true_positives
        if base_ann[iou_indices[mask1]].shape[0]==0:
            correct1=[]
        else:
            correct1=gt_ann[:,4][mask1] ==(base_ann[iou_indices[mask1]][:,5])

        iou1=boxes.box_iou(det1_ann[:,:4],gt_ann[:,:4])
        iou_values,iou_indices  = iou1.max(axis=0)[0],iou1.max(axis=0)[1]
        mask2 = iou_values>=iou_true_positives
        if det1_ann[iou_indices[mask2]].shape[0]==0:
            correct2=[]
        else:
            correct2=gt_ann[:,4][mask2] ==(det1_ann[iou_indices[mask2]][:,5])
        

        gt =torch.arange(gt_ann.shape[0])
        total_val_instances+=gt.shape[0]
        lista1=gt[mask1][correct1].tolist()
        lista2=gt[mask2][correct2].tolist()
        yes_no += len(set(lista1) - set(lista2))
        no_yes += len(set(lista2) - set(lista1))
        yes_yes += len(set(lista1) & set(lista2))
        no_no += len(set(torch.arange(gt_ann.shape[0]).tolist())-(set(lista1) | set(lista2)))
    except IndexError:
        missed+=1
        pass
    

x = PrettyTable()
table = [[yes_yes, yes_no],[no_yes, no_no]]
result = mcnemar(table, exact=False,correction=True)
print(args.compare_list)
x.field_names = [f"IoU@{iou_true_positives}", "Correct2","Inorrect2"]
x.add_rows(
    [
        ["Correct1", yes_yes, yes_no,],
        ["Incorrect1", no_yes, no_no],
    ]
)
print(x)
print("McNemar's test:")
print('statistic=%.4f, p-value=%.4f' % (result.statistic, result.pvalue))
alpha = args.alpha
if result.pvalue > alpha:
    print('Same proportions of errors (fail to reject H0)')
else:
    print('Different proportions of errors (reject H0)')