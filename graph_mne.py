import matplotlib.pyplot as plt
import sys
import mne
import numpy as np
import os
args =sys.argv
EDF_NAME = args[1]
data=mne.io.read_raw_edf(EDF_NAME)
data_eeg=data.get_data()
data_eeg=data_eeg[:-3]
print(data_eeg.shape)

for j in range(int(data_eeg.shape[1]/256)):
    data_window=data_eeg[:,j*256:(j+1)*256]
    plt.clf()
    for i,k in enumerate(data_window):
        plt.plot(k+0.00010*i,color='black',linewidth=1)
    plt.ylim(-0.001,0.0040)
    #plt.show()
    plt.savefig('../img/'+os.path.splitext(os.path.basename(EDF_NAME))[0]+'/'+str(j)+'.jpg')
print(data_eeg.sum(axis=1))