from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from lvis import LVISEval
import os
import json
import hydra
import pickle
import itertools


def save_results(results,rank):
    path='bbox_results/temp_res'
    if not os.path.exists(path):
        os.makedirs(path)

    temp_name=os.path.join(path,'{}.json'.format(rank))
    with open(temp_name,'wb') as f:
        pickle.dump(results, f)
    


def eval_results(epoch,dset_name,validation_path):
    results=[]
    mAP = -1
    directory = 'bbox_results/temp_res'
    for filename in os.listdir(directory):
        if filename.endswith(".json"):
            temp_name = os.path.join(directory, filename)
            with open(temp_name, 'rb') as f:
                results=list(itertools.chain(results, pickle.load(f)))
                
    cwd = os.getenv('owd')  
    validation_path=os.path.join(cwd,validation_path)
    
    if not os.path.exists(f'bbox_results/{dset_name}/'):
        os.makedirs(f'bbox_results/{dset_name}/')

    json.dump(results, open(f'./bbox_results/{dset_name}/results_{epoch}.json', 'w'), indent=4)
    resFile=f'./bbox_results/{dset_name}/results_{epoch}.json'

    if dset_name=='coco':
        cocoGt=COCO(validation_path)

        
        cocoDt=cocoGt.loadRes(resFile)
        imgIds=sorted(cocoGt.getImgIds())
        cocoDt.loadAnns()


        # # running evaluation
        cocoEval = COCOeval(cocoGt,cocoDt,'bbox')
        cocoEval.evaluate()
        cocoEval.accumulate()
        cocoEval.summarize()

        mAP=cocoEval.stats[0]

    elif(dset_name=='lvis'):
    
        lvis_eval = LVISEval(validation_path, resFile, 'bbox')
        lvis_eval.run()
        metrics=lvis_eval.get_results()
        mAP=metrics['AP']

    return (mAP)