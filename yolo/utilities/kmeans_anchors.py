from lvis import LVIS
import numpy as np
from tqdm import tqdm
import pandas as pd 
from sklearn.cluster import KMeans 



annfile ="../../../../datasets/coco/annotations/lvis_v1_train.json"
lvis=LVIS(annfile)

img_ids = lvis.get_img_ids()
xmin=[]
ymin=[]
xmax=[]
ymax=[]
width=[]
height=[]
category=[]
for id in tqdm(img_ids):
    img=lvis.load_imgs([id])[0]
    w=img['width']
    h=img['height']
    ann_ids = lvis.get_ann_ids([id])
    annotations = lvis.load_anns(ann_ids) 
    for ann in annotations:
        x0 = ann['bbox'][0]/w
        y0= ann['bbox'][1]/h
        x1 = x0 + ann['bbox'][2]/w
        y1 = y0 + ann['bbox'][3]/h
        xmin.append(x0)
        ymin.append(y0)
        xmax.append(x1)
        ymax.append(y1)
        width.append(ann['bbox'][2]/w)
        height.append(ann['bbox'][3]/h)
        category.append(ann['category_id'])

df = pd.DataFrame()
df['xmin'] = xmin
df['ymin'] = ymin
df['xmax'] = xmax
df['ymax'] = ymax
df['width'] = width
df['height'] = height
df['category'] = category

df['aspect_ratio'] = df['width']/df['height']
df['area'] = df['width']*df['height']
mask = (df['aspect_ratio'] > 0.2)*(df['aspect_ratio'] < 5)
x_coordinates=np.array([df['width'],df['height'],df['aspect_ratio'],df['area']]).T
centers=[]

area_mask = df['area']<0.01
kmeans = KMeans(n_clusters=3, random_state=0).fit(x_coordinates[mask*area_mask])
centers.append(kmeans.cluster_centers_)
area_mask = (df['area']>0.01) & (df['area']<0.1)
kmeans = KMeans(n_clusters=3, random_state=0).fit(x_coordinates[mask*area_mask])
centers.append(kmeans.cluster_centers_)
area_mask = (df['area']>0.1)
kmeans = KMeans(n_clusters=3, random_state=0).fit(x_coordinates[mask*area_mask])
centers.append(kmeans.cluster_centers_)

print(centers)