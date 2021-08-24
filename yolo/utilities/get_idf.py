from pycocotools.coco import COCO
import sys
sys.path.insert(1, '../') # add .. to path to discover lvis folder
from lvis import LVIS
import numpy as np
from scipy.sparse import coo_matrix, hstack,vstack
from tqdm import tqdm
import os
import pandas as pd
import math
import argparse 
from scipy.special import ndtri

parser = argparse.ArgumentParser(description='Cmpute IDF for dataset')

parser.add_argument('--dset_name', type=str, help='Dataset name, either coco or lvis',default='lvis')

    
def get_idf(annfile,dset_name):
    df = pd.DataFrame()
    if dset_name=='coco':
        coco=COCO(annfile)
        ims=[]
        last_cat = coco.getCatIds()[-1]
        num_classes = last_cat +1 # for bg
        for imgid in coco.getImgIds():
            anids = coco.getAnnIds(imgIds=imgid)
            categories=[]
            for anid in anids:
                categories.append(coco.loadAnns(int(anid))[0]['category_id'])
            ims.append(categories)
    else:
        lvis=LVIS(annfile)
        ims=[]
        last_cat = lvis.get_cat_ids()[-1]
        num_classes = last_cat +1 # for bg
        for imgid in lvis.get_img_ids():
            anids = lvis.get_ann_ids(img_ids=[imgid])
            categories=[]
            for annot in lvis.load_anns(anids):
                categories.append(annot['category_id'])
            ims.append(categories)

    final=0
    k=0
    print('calculating idf ...')
    for im in tqdm(ims):
        cats = np.array(im,dtype=np.int)
        cats = np.bincount(cats,minlength=num_classes)
        cats=np.array([cats])
        cats = coo_matrix(cats)
        if k==0:
            final= cats
        else:
            final = vstack([cats, final])
        k=k+1

    mask = final.sum(axis=0)>0
    mask=mask.tolist()[0]
    final = final.tocsr()[:,mask]
    doc_freq = (final>0).sum(axis=0)
    instance_freq = final.sum(axis=0)
    pobs = doc_freq/final.shape[0]
    pobs=np.array(pobs)[0]
    df['smooth'] = (np.log((final.shape[0]+1)/(doc_freq+1))+1).tolist()[0]
    df['raw'] = (np.log((final.shape[0])/(doc_freq))).tolist()[0]
    df['prob'] = (np.log((final.shape[0]-doc_freq)/(doc_freq))).tolist()[0]
    df['normit'] = -ndtri(pobs)
    df['gombit'] = -np.log(-np.log(1-pobs))
    df['base2'] = -np.log2(pobs)
    df['base10'] = -np.log10(pobs)
    #obj
    N = instance_freq.sum()
    pobs = instance_freq/N
    pobs=np.array(pobs)[0]
    df['smooth_obj'] = (np.log((N+1)/(instance_freq+1))+1).tolist()[0]
    df['raw_obj'] = (np.log((N)/(instance_freq))).tolist()[0]
    df['prob_obj'] = (np.log((N-instance_freq)/(instance_freq))).tolist()[0]
    df['normit_obj'] = -ndtri(pobs)
    df['gombit_obj'] = -np.log(-np.log(1-pobs))
    df['base2_obj'] = -np.log2(pobs)
    df['base10_obj'] = -np.log10(pobs)
    df['img_freq'] = doc_freq.tolist()[0]
    df['instance_freq'] = instance_freq.tolist()[0]
    df.to_csv('idf.csv',index=False)

    
args = parser.parse_args()

if args.dset_name=='coco':
    path = "../../../../datasets/coco/annotations/instances_train2017.json"
else:
    path= "../../../../datasets/lvis/data/lvis_v1_train.json"

get_idf(path,args.dset_name)