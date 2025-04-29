import hist
from hist import Hist

import numpy as np
import awkward as ak
import uproot
import vector
vector.register_awkward()
import os
import shutil
import zipfile
import tarfile
import urllib
import requests
from tqdm import tqdm

import matplotlib.pyplot as plt

def _pad(a, maxlen, value=0, dtype='float32'):
    if isinstance(a, np.ndarray) and a.ndim >= 2 and a.shape[1] == maxlen:
        return a
    elif isinstance(a, ak.Array):
        if a.ndim == 1:
            a = ak.unflatten(a, 1)
        a = ak.fill_none(ak.pad_none(a, maxlen, clip=True), value)
        return ak.values_astype(a, dtype)
    else:
        x = (np.ones((len(a), maxlen)) * value).astype(dtype)
        for idx, s in enumerate(a):
            if not len(s):
                continue
            trunc = s[:maxlen].astype(dtype)
            x[idx, :len(trunc)] = trunc
        return x
    
def _download(url, fname, chunk_size=1024):
    '''https://gist.github.com/yanqd0/c13ed29e29432e3cf3e7c38467f42f51'''
    resp = requests.get(url, stream=True)
    total = int(resp.headers.get('content-length', 0))
    with open(fname, 'wb') as file, tqdm(
        desc=fname,
        total=total,
        unit='iB',
        unit_scale=True,
        unit_divisor=1024,
    ) as bar:
        for data in resp.iter_content(chunk_size=chunk_size):
            size = file.write(data)
            bar.update(size)

example_file = 'JetClass_example_100k.root'
example_file2 = 'HToWW2Q1L_100.root'

tree = uproot.open(example_file)['tree']

table = tree.arrays()
feat_list = ['jet_pt', 'jet_eta', 'jet_phi', 'jet_energy', 'jet_nparticles', 'jet_sdmass', 'jet_tau1', 'jet_tau2', 'jet_tau3', 'jet_tau4']
label_list = ['label_QCD', 'label_Hbb', 'label_Hcc', 'label_Hgg', 'label_H4q', 'label_Hqql', 'label_Zqq', 'label_Wqq', 'label_Tbqq', 'label_Tbl']


# data = table['jet_eta'].to_numpy()
# for i in range(len(data)):
#     if(data[i]>400):
#         print(data[i],i)

bin_num = 50
hpt = hist.Hist(
    hist.axis.Regular(bins= bin_num, start=0, stop =1000, label= feat_list[0], name= feat_list[0]),
    hist.axis.StrCategory(label_list, name="labels", label=label_list),
)


data = table[feat_list[0]].to_numpy()

label = table[label_list[3]].to_numpy()
mask = data[label == 1]

# masked = np.ma.masked_array(data,mask = np.logical_not(label))
sum = 0

# hpt.fill(masked, labels = label_list[1])
# hpt.fill(mask, labels = label_list[3])
# hpt.fill(data, labels = label_list[1])
# fig, ax = plt.subplots(figsize=(10,12))


# hpt[{"labels": label_list[1]}].plot1d(ax=ax, label = label_list[1])
# hpt[{"labels": label_list[3]}].plot1d(ax=ax, label = label_list[3])
# hpt[{"labels": label_list[1]}].plot1d(ax=ax, label = label_list[1])

# ax.legend()
# plt.savefig('jet_plots/test.png')

for j in range(len(feat_list)):
    bin_num = 50
    start_n = 0
    stop_n = 1000
    if(j == 1 or j == 2):
        start_n = -3
        stop_n = 3
    elif(j==3):
        stop_n = 3500
    elif(j==4):
        stop_n = 140
    elif(j==5):
        stop_n = 500
    elif(j==6):
        stop_n = 0.5
    elif(j==7):
        stop_n = 0.3
    elif(j>7):
        stop_n = 0.2

    
    
    hpt = hist.Hist(
        hist.axis.Regular(bins= bin_num, start= start_n, stop = stop_n, label= feat_list[j], name= feat_list[j]),
        hist.axis.StrCategory(label_list, name="labels", label=label_list),
    )

    print(feat_list[j])
    data = table[feat_list[j]].to_numpy()
    for i in range(len(label_list)):
        label = table[label_list[i]].to_numpy()
        masked = data[label == 1]
        hpt.fill(masked, labels = label_list[i])

    fig, ax = plt.subplots(figsize=(10,12))

    for i in range(len(label_list)):
        hpt[{"labels": label_list[i]}].plot1d(ax=ax, label = label_list[i])

    ax.legend()
    plt.savefig('jet_plots/' + feat_list[j] + '.png')

