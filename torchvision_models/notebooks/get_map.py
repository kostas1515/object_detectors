import pandas as pd
import sys,os
import matplotlib.pyplot as plt
import argparse
# insert at 1, 0 is the script path (or '' in REPL)
sys.path.insert(1, '../')
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import os
import numpy as np
import pandas as pd
import json
from matplotlib.pyplot import figure

parser = argparse.ArgumentParser(description='mAP comparisons')

parser.add_argument('--compare_list', nargs='+', help='Experiments to compare')
parser.add_argument('--dset_name', type=str, help='Dataset name, either coco or lvis',default='coco')
parser.add_argument('--categories', nargs='+', help='Columns to compare',default=['all'])
parser.add_argument('--rids',  nargs='+', help='Columns to compare',default=[0])
parser.add_argument('--metric_columns', nargs='+', help='Columns to compare',default=[0])
args = parser.parse_args()
metric_names=np.array(['AP','AP50','AP75','APs','APm','APl','AR1','AR10','AR100','ARs','ARm','ARl'])
cwd="./"

if args.dset_name=='coco':
    validation_path = "../../../../datasets/coco/annotations/instances_val2017.json"
    num_categories=80
validation_path=os.path.join(cwd,validation_path)
cocoGt=COCO(validation_path)
rids=[int(rid) for rid in args.rids]
results={}
results['columns'] = metric_names[args.metric_columns].tolist()
for counter,exp_name in enumerate(args.compare_list):
    resFile=f'../jsons/{args.dset_name}/{exp_name}/model_{rids[counter]}.json'
    cocoDt=cocoGt.loadRes(resFile)
    cocoDt.loadAnns()
    #  running evaluation
    cocoEval = COCOeval(cocoGt,cocoDt,'bbox')
    if not args.categories:
        cocoEval.evaluate()
        cocoEval.accumulate()
        cocoEval.summarize()
        temp={}
        temp[0]=[np.array(cocoEval.stats)[args.metric_columns]].tolist()
        results[exp_name] = temp
    elif args.categories[0] == 'all':
        categories= [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 27, 28, 31, 32, 33, 34,
         35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63,
         64, 65, 67, 70, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 84, 85, 86, 87, 88, 89, 90]
        temp={}
        for i in results['columns']:
            temp[i]=[]
        for cat in categories:
            cocoEval.params.catIds = [cat]
            cocoEval.evaluate()
            cocoEval.accumulate()
            cocoEval.summarize()
            for k,i in enumerate(results['columns']):
                temp[i].append(cocoEval.stats[k])
        results[exp_name]=temp
    else:
        categories= args.categories
        temp={}
        for i in results['columns']:
            temp[i]=[]
        for cat in categories:
            cocoEval.params.catIds = [cat]
            cocoEval.evaluate()
            cocoEval.accumulate()
            cocoEval.summarize()
            for k,i in enumerate(results['columns']):
                temp[i].append(cocoEval.stats[k])
        results[exp_name]=temp

path2save='./mAP_viz/'+"_".join(args.compare_list)
if not os.path.exists(path2save):
    os.mkdir(path2save)

print(results)
json.dump(results, open(os.path.join(path2save,"results.json"), 'w' ) )
df = pd.read_csv('../coco_files/idf.csv')
coco_names=[n.rstrip() for n in open("../coco_files/coco.names").readlines()]
indices=df['idf_weights'].sort_values().index.values
coco_names=np.array(coco_names)[indices]

difference = 0
for mc in results['columns']:
    for exp_name in args.compare_list:
        difference = np.array(results[exp_name][mc]) - difference
        figure(num=None, figsize=(20, 10), dpi=80, facecolor='w', edgecolor='k')
        plt.xticks(rotation=90)
        plt.bar(coco_names,difference[indices])
        plt.savefig(os.path.join(path2save,mc+'.png'))
        plt.close()
    
    
    
    



            
    


