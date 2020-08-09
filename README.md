# SAIL2020-1

# Demo

# Note
- csv_data_devtest.csv          ... dev_testのデータのうち有効なもののリストです。
- csv_data_train.csv            ... trainのデータのうち有効なもののリストです。
- seizure_dev_test_flags.csv    ... 有効なdev_testデータのflagのうちラベルがGNSZのリストです。
- seizure_train_flags.csv    ... 有効なtrainデータのflagのうちラベルがGNSZのリストです。
# Requirement
- matplotlib 3.2.2  
- mne 0.19.0

# Installation
はじめに、オープンソースの脳波データを[The Neural Data Consortium](https://www.isip.piconepress.com/projects/tuh_eeg/)からダウンロードする必要があります。e-mailを登録することでダウンロード用のパスワードが発行されます。
```
git clone https://github.com/tuat-yate/SAIL2020-1.git
pip3 install matplotlib
pip3 install mne==0.19.0
```
