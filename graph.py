import matplotlib.pyplot as plt
import sys
import mne
args =sys.argv
EDF_NAME = args[1]
data=mne.io.read_raw_edf(EDF_NAME)
data_eeg=data.get_data(start=0,stop=int(data.info['sfreq']))
for i in range(0,len(data.ch_names)):
    plt.plot([k*pow(10,6)+5*i for k in data_eeg[i]])
plt.xticks([0,int(data.info['sfreq'])-1],[0,1])
plt.show()