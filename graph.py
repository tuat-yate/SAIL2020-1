import matplotlib.pyplot as plt
import sys
import mne
args =sys.argv
EDF_NAME = args[1]
data=mne.io.read_raw_edf(EDF_NAME)
#data_eeg=data.get_data(start=0,stop=int(data.info['sfreq'])*10)
data_eeg=data.get_data(start=int(data.info['sfreq'])*60,stop=int(data.info['sfreq'])*120)
for i in range(len(data.ch_names)):
    plt.plot([k*pow(10,6)+5*i for k in data_eeg[i]])
    plt.show()
#plt.xticks([0,int(data.info['sfreq'])-1],[0,1])