import glob
import mne
import collections
import csv
datalist=list()
for i in glob.glob("./train/*.edf"):
    data=mne.io.read_raw_edf(i,preload=True,stim_channel=None)
    datalist.append(str(data.ch_names)+","+str(data.info['sfreq']))
c=collections.Counter(datalist)
print(c.most_common()[0])
