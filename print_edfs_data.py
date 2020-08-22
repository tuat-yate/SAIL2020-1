import glob
import mne
import collections
import csv
import os
import re
datalist=list()
tmplist=list()

def update_seizures_dict(seizures_dict,seizure_type,seizure_times):
    if(seizure_type not in seizures_dict):
        seizures_dict[seizure_type]=seizure_times
    else:
        seizures_dict[seizure_type]+=seizure_times
    return seizures_dict
toprow=['ファイル名','チャンネル数','チャンネル','サンプリング周波数','ファイル総時間','発作種類数','発作種類1','種類1内発作総時間','発作種類2','種類2内発作総時間','発作種類3','種類3内発作総時間']
with open('./data_train.csv') as f:
    reader = csv.reader(f)
    list1=[row for row in reader]
for i in glob.glob("../train/*.edf"):
    data=mne.io.read_raw_edf(i,preload=True,stim_channel=None)
    tmplist=[os.path.basename(i),str(len(data.ch_names)),str(data.ch_names),str(data.info['sfreq']),str(float(data.n_times)/float(data.info['sfreq']))]
    seizures_dict=dict()
    for k in list1:
        if(os.path.basename(k[11]).replace('tse','edf')==os.path.basename(i)and(not(k[14]==''))):
            seizures_dict=update_seizures_dict(seizures_dict,k[14],float(k[13])-float(k[12]))  
    tmplist.append(len(seizures_dict)) 
    for dict_keys in seizures_dict.keys():
        tmplist.append(dict_keys)
        tmplist.append(seizures_dict[dict_keys])
    datalist.append(tmplist)
    tmplist=list()
#c=collections.Counter(datalist)
#print(c.most_common()[0])
with open('tmp_train.csv', 'w') as file:
    writer = csv.writer(file, lineterminator='\n')
    writer.writerow(toprow)
    writer.writerows(datalist)
