import os
import pandas as pd
import numpy as np
from tqdm import tqdm
from os.path import join as join
import seaborn as sns
import matplotlib.pyplot as plt

def calculate_distances_vectorized(dfA, dfB):
    a = dfA[['x', 'y']].values
    b = dfB[['x', 'y']].values
    distance_matrix = np.sqrt(np.sum((a[:, np.newaxis, :] - b[np.newaxis, :, :])**2, axis=2))
    return distance_matrix

mpp=0.25
core_width=7200
core_height=7200
thresh=5
base_dir='./registration_P1_seg'
text_dir=join(base_dir,'texts')
text_files=os.listdir(text_dir)
text_files.sort()
cell_df1 = pd.read_csv(join(base_dir, 'cluster_results.csv'))
cell_df1 = cell_df1[cell_df1['cluster'] != 'other'].copy()
cell_df1['core_id'] = cell_df1['Sample Name']
csv_core_li=[]
img_core_li=[]
for text_file in text_files:
    csv_core=pd.read_csv(os.path.join(text_dir,text_file))
    img_core=np.load(os.path.join(base_dir,'cores',text_file.split('.txt')[0]+'.npy'))
    csv_core['core_id']=csv_core['Name'].apply(lambda x:x.split('[')[1].split(']')[0])
    core_id=list(set(csv_core['core_id']))[0]
    csv_core=csv_core[['core_id','Centroid X px','Centroid Y px','Length px','Area px^2']].copy()
    csv_core.columns=['core_id','x','y','len','area']
    csv_core['x']=csv_core['x']*mpp
    csv_core['y']=csv_core['y']*mpp
    csv_core['len']=csv_core['len']*mpp
    csv_core['area']=csv_core['area']*mpp*mpp
    cluster_core=cell_df1[cell_df1['core_id']==core_id].copy()
    core_xy = cluster_core['Sample Name'].values[0].split('[')[-1].split(']')[0].split(',')
    core_xy=[float(v) for v in core_xy]
    core_xy_min=[core_xy[0]-(core_width*mpp)/2,core_xy[1]-(core_height*mpp)/2]
    cluster_core['Cell X Position']=cluster_core['Cell X Position']-core_xy_min[0]
    cluster_core['Cell Y Position']=cluster_core['Cell Y Position']-core_xy_min[1]
    cluster_core=cluster_core[['core_id','Cell X Position','Cell Y Position','label']].copy()
    cluster_core.columns=['core_id','x','y','label']
    cluster_core.reset_index(drop=True, inplace=True)
    distance_matrix = calculate_distances_vectorized(csv_core, cluster_core)
    min_distances = np.min(distance_matrix, axis=1)
    min_indices = np.argmin(distance_matrix, axis=1)
    csv_core['label']=cluster_core.loc[min_indices,'label'].values
    csv_core['min_distance']=min_distances
    csv_core=csv_core[csv_core['min_distance']<=thresh].copy()
    csv_core_li.append(csv_core)
    img_core = img_core[csv_core.index]
    img_core_li.append(csv_core)

csv_core_li = pd.concat(csv_core_li, ignore_index=True)
csv_core_li.to_csv(join(base_dir,'csv_core_li.csv'),index=False)
np.save(join(base_dir,'img_core_li.npy'),np.array(img_core_li))