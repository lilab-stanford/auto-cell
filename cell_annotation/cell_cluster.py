import os
import time
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from os.path import join as join
import seaborn as sns
from tqdm import tqdm
from sklearn.cluster import KMeans
import scanpy as sc
# import rapids_singlecell as rsc

marker_selected = ['Entire Cell Area', \
                   'Entire Cell CD3 (PPD650) Mean', \
                   'Entire Cell PANCK (PPD690) Mean', \
                   'Entire Cell CD20 (PPD620) Mean'
                   ]
column_selected=['Sample Name', 'Tissue Category','Cell ID','Cell X Position','Cell Y Position']
cell_csv_path='./Cell seg data'
cell_csvs = glob.glob(join(cell_csv_path, '*cell_seg_data.txt'))
cell_df_li=[]
for cell_csv in tqdm(cell_csvs[:]):
    cell_df = pd.read_csv(cell_csv, sep='\t')
    cell_df_li.append(cell_df)
cell_df = pd.concat(cell_df_li, axis=0)
for column in cell_df.columns:
    if column not in (column_selected + marker_selected):
        cell_df.drop(column, axis=1, inplace=True)
cell_df.reset_index(drop=True, inplace=True)
mask = cell_df.isna().sum(axis=1).astype(bool)
cell_df = cell_df[~mask].copy()
cell_df.reset_index(drop=True, inplace=True)

grouped = cell_df.groupby('Sample Name')
threshold = 0.997
quantiles = grouped['Entire Cell Area'].quantile(threshold)
filtered_df_list = []
filtered_df_list_o=[]
for patient, group in grouped:
    quantile_value = quantiles[patient]
    filtered_group = group[group['Entire Cell Area'] < quantile_value].copy()
    filtered_group_o = filtered_group.copy()
    for biomarker in ['Entire Cell CD3 (PPD650) Mean', \
                   'Entire Cell PANCK (PPD690) Mean', \
                   'Entire Cell CD20 (PPD620) Mean']:
        mean = filtered_group[biomarker].mean()
        std = filtered_group[biomarker].std()
        filtered_group[biomarker] = (filtered_group[biomarker] - mean) / std
    filtered_df_list.append(filtered_group)
    filtered_df_list_o.append(filtered_group_o)
cell_df = pd.concat(filtered_df_list)
cell_df.reset_index(drop=True, inplace=True)
feature_norm = cell_df[marker_selected].copy()
cell_df_o=pd.concat(filtered_df_list_o)
cell_df_o.reset_index(drop=True, inplace=True)

adata = sc.AnnData(cell_df[marker_selected])
# rsc.get.anndata_to_GPU(adata)
sc.pp.neighbors(adata)
sc.tl.leiden(adata,resolution=0.1)
# cell_df_o['cluster'] = list(adata.obs['leiden'].astype('int'))
cell_df['cluster'] = list(adata.obs['leiden'].astype('int'))
# sc.tl.rank_genes_groups(adata, "leiden", method="t-test")
# sc.pl.rank_genes_groups(adata, n_genes=25, sharey=False)
feature_norm=np.array(cell_df[marker_selected])
predicted=np.array(cell_df['cluster'])
# cluster_area = cell_df.groupby('cluster')['Entire Cell Area'].mean()
cell_df.to_csv('cluster_results.csv',index=False)
# UMAP
import umap
reducer = umap.UMAP(verbose=True)
embedding = reducer.fit_transform(cell_df[marker_selected])
embedding = np.array(embedding)
x1_axis = embedding[:,0]
x2_axis = embedding[:,1]
colors = ["#75fbfd", "#ea33f7", "#ea3323", "#A3A5A6"]
fig, ax = plt.subplots(figsize=(12, 12))
labels = np.array(cell_df['cluster'])
for label, color in zip(np.unique(labels), colors):
    indices = np.where(labels == label)
    ax.scatter(x1_axis[indices], x2_axis[indices], color=color, label=label, s=0.2,alpha=1)
plt.tight_layout()
plt.show()

# cell_df[marker_selected] = cell_df[marker_selected].rank(pct=True)
for i in list(set(cell_df['cluster'])):
    cluster = cell_df_o[cell_df_o['cluster'] == i]
    cluster_mean = cluster[marker_selected].mean()
    cluster_mean = cluster_mean.values
    angles = np.linspace(0, 2 * np.pi, len(cluster_mean), endpoint=False).tolist()
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))
    ax.bar(angles, cluster_mean, color=["#ea33f7", "#75fbfd", "#ea3323"], alpha=0.9, width=0.52, zorder=10)
    plt.title('Cluster '+str(i)+': '+str(len(cluster)), fontsize=24)
    ax.set_xticks(angles)
    ax.set_xticklabels([])
    plt.show()
cluster_mean = cell_df.groupby('cluster')[marker_selected].mean()
cluster_mean = cluster_mean.T
fig, ax = plt.subplots(figsize=(20, 10))
sns.heatmap(cluster_mean.iloc[:,:3], cmap='coolwarm', ax=ax)
plt.tight_layout()
plt.show()
