import glob
import mne
import collections
import csv
datalist=list()
edflist=list()
for i in glob.glob("../dev_test/*.edf"):
    data=mne.io.read_raw_edf(i,preload=True,stim_channel=None)
    #datalist.append(str(data.ch_names)+","+str(data.info['sfreq']))
    if(data.ch_names==['EEG FP1-REF', 'EEG FP2-REF', 'EEG F3-REF', 'EEG F4-REF', 'EEG C3-REF', 'EEG C4-REF', 'EEG P3-REF', 'EEG P4-REF', 'EEG O1-REF', 'EEG O2-REF', 'EEG F7-REF', 'EEG F8-REF', 'EEG T3-REF', 'EEG T4-REF', 'EEG T5-REF', 'EEG T6-REF', 'EEG T1-REF', 'EEG T2-REF', 'EEG FZ-REF', 'EEG CZ-REF', 'EEG PZ-REF', 'EEG EKG1-REF', 'EEG C3P-REF', 'EEG C4P-REF', 'EEG SP1-REF', 'EEG SP2-REF', 'EEG A1-REF', 'EEG A2-REF', 'EEG 31-REF', 'EEG 32-REF', 'IBI', 'BURSTS', 'SUPPR']and data.info['sfreq']==256.0):
        edflist.append(i)
#c=collections.Counter(datalist)
#print(c.most_common()[0])
with open('csv_data_devtest.csv', 'w') as file:
    writer = csv.writer(file, lineterminator='\n')
    writer.writerow(edflist)
