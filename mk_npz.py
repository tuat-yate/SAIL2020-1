# -*- coding: utf-8 -*-
#
# mk_npz.py
# 
# EDFデータを SEG_SIZE の長さ毎に画像化し、その画素値から3次元のnpzファイルを作成
# 出力されるnpzの形式は，[n_segments, n_px_x, n_px_y]
#

import glob
import numpy as np
from mne import io
from CREDF import _CREDF as CREDF
from showSegment import eeg_to_fig
import re
import csv
import math

SEG_SIZE = 1 # seconds

'''
data = io.read_raw_edf(EDF_NAME, preload=True, stim_channel=None)
#start~stopまでのデータのみを取得、0~n秒までだったら、start=0,stop=sampling_frequency*n
data_eeg           = data.get_data(start=0,stop=255)
sampling_frequency = data.info['sfreq']
channel_names      = data.ch_names
#全体の秒数はn_times/sampling_frequency
n_times            = data.n_times
for i in range(len(data_eeg)):
    plt.plot(data_eeg[i])
plt.show()
'''
def get_seizures_edf_set():
    #devtestは./seizures_devtest.csv
    with open('./seizures_train.csv') as f:
        reader=csv.reader(f)
        data=[row for row in reader]
    edf_set=set()
    for i in range(1,len(data)):
        edf_set.add(re.sub(r'\.\/edf\/train\/[^\/]*\/[^\/]*\/[^\/]*\/[^\/]*\/','',data[i][11]).replace('.tse','.edf'))
    edf_list=list(edf_set)
    edf_list.sort()
    return edf_list

def cut_up(num):
    if not (re.search('.',num)):
        return int(num)
    else:
        return math.ceil(float(num))

def cut_down(num):
    if not (re.search('.',num)):
        return int(num)
    else:
        return math.floor(float(num))

def labeling_list(edf):
    with open('./seizures_train.csv') as f:
        reader=csv.reader(f)
        data=[row for row in reader]
    training_data=list()
    for i in range(1,len(data)):
        #fullpathを単純なファイル名に置換
        data[i][11]=re.sub(r'\.\/edf\/train\/[^\/]*\/[^\/]*\/[^\/]*\/[^\/]*\/','',data[i][11]).replace('.tse','.edf')
        #引数で与えられたCSVのstart,stopのみを別のリストに保存
        if(data[i][11]==edf.edf_name):
            start_n = cut_down(data[i][12]) * edf.samp_freq
            end_n = cut_up(data[i][13]) * edf.samp_freq
            training_data.append([start_n, end_n])
    label = np.zeros(int(edf.n_samples), dtype=bool)
    x = np.arange(int(edf.n_samples))
    for (start_n, end_n) in training_data:
        label = np.logical_or(label, np.logical_and(start_n < x, x < end_n))
    return label

target_edfs = get_seizures_edf_set()
edfs = []
formats = []
for i, path in enumerate(target_edfs):
    print(f"Loading EDF: {i+1}/{len(target_edfs)}")
    edf = CREDF("./train/" + path)
    edfs.append(edf)
    ch_names = ';'.join(edf.ch_names)
    samp_freq = int(edf.samp_freq)
    formats.append(ch_names + ';' + str(samp_freq))

unique_formats, counts = np.unique(formats, return_counts=True)
employment_format = unique_formats[np.argmax(counts)]
available_chs = [ch for ch in employment_format.split(';')][:-1]

eeg_chs = ['EEG FP1-REF', 'EEG FP2-REF', 'EEG F3-REF', 'EEG F4-REF',
           'EEG C3-REF', 'EEG C4-REF', 'EEG P3-REF', 'EEG P4-REF',
           'EEG O1-REF', 'EEG O2-REF', 'EEG F7-REF', 'EEG F8-REF',
           'EEG T3-REF', 'EEG T4-REF', 'EEG T5-REF', 'EEG T6-REF',
           'EEG A1-REF', 'EEG A2-REF', 'EEG FZ-REF', 'EEG CZ-REF',
           'EEG PZ-REF', 'EEG T1-REF', 'EEG T2-REF']

eeg_ch_idxs = [available_chs.index(ch) for ch in eeg_chs]

edfs = [edf for edf, fmt in zip(edfs, formats) if fmt == employment_format]



splited_eegs = [] #この中には[(n_segments, n_ch, fs*SEG_SIZE)]のデータが入る
labels = [] #この中には[n_segments]のラベル{0,1}が入る
edf_images = []

for i, edf in enumerate(edfs):
    print(f"Creating EEG segments: {i+1}/{len(edfs)}")
    
    # セグメント化
    dat = edf.split_into(edf.samp_freq*SEG_SIZE, eeg_ch_idxs)
    dat = dat.reshape(len(eeg_ch_idxs), -1, edf.samp_freq*SEG_SIZE)
    dat = dat.transpose([1, 0, 2]) #should be (n_segments, n_ch, fs*SEG_SIZE)
    splited_eegs.append(dat)
    
    #画像化
    images = np.vstack([eeg_to_fig(
        segment, eeg_chs, f'figs/{edf.edf_name.replace(".edf","")}_{j:03d}.png')[None, :] for j, segment in enumerate(dat)])
    edf_images.append(images)

    #ラベル作成
    splited_length = int(dat.shape[0] * dat.shape[2])
    splited_label = labeling_list(edf)[:splited_length]
    splited_label = splited_label.reshape(-1, edf.samp_freq*SEG_SIZE)
    label = splited_label.any(1) # fs*SEG_SIZE の長さの区間内に，一度でも
                                 # 発作ラベルがあればこのセグメントは陽性とする
    label = label.astype(float)
    labels.append(label)
    

# Example:
#   splited_edfs[0].shape is
#   (601, 31, 250) that means (n_segments, n_ch, fs*SEG_SIZE)

#   labels[0].shape is
#    (601, ) = [0,0,0,0,0,1,1,1,1,0,0,0,.....]

'''
1症例あたり，
[[start_n, end_n], [start_n, end_n],...] 注意 nの次元はサンプル点
label = np.zeros(601, dtype=bool)
t = np.arange(601)
for i in 注釈数:
    label = np.or(labels, np.and(start_n[i] < t, t < end_n[i]) )

labels = [label_0, label_1, ...]



X, y
X.shape <- (N_SAMPLE, 31, 250)
y.shape <- (N_SAMPLE) in {0,1}
'''



