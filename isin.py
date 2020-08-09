# isin.py
# ラベルcsvのうち有効データのもののみ残す
import csv
import os
outlist=list()
#有効データのリスト[['..\.edf'],...,['..\.edf']]
with open('csv_data_train.csv') as f:
    reader = csv.reader(f)
    list1=[row for row in reader]
# ラベルデータのリスト[['..\.edf',start,stop,],...,['..\.edf',start,stop,]]
with open('seizure_train_flags.csv') as f:
    reader = csv.reader(f)
    list2=[row for row in reader]
for i in list1:
    for k in list2:
        if(k[0]==i[0]):
            outlist.append(k)
'''
with open('csv_data_devtest.csv', 'w') as f:
    writer = csv.writer(f)
    writer.writerow(l)
'''
print(outlist)