import csv
import collections
#辞書をもらって数値を足す
def update_seizures_dict(seizures_dict,seizure_type,seizure_times):
    if(seizure_type==''):
        pass
    elif(seizure_type not in seizures_dict):
        seizures_dict[seizure_type]=float(seizure_times)
    else:
        seizures_dict[seizure_type]+=float(seizure_times)
    return seizures_dict

with open('./fulldata/full_data_train.csv') as f:
    reader = csv.reader(f)
    list1=[row for row in reader]
chlist=list()
for j in list1:
    chlist.append(j[2])
#6-9がデータ本体
all_edf_time=0
seizures_dict=dict()
for n,i in enumerate(list1):
    if(n==0):
        pass
    else:
        all_edf_time+=float(i[4])
        update_seizures_dict(seizures_dict,i[6],i[7])
        update_seizures_dict(seizures_dict,i[6],i[7])
for k in seizures_dict:
    all_edf_time -= seizures_dict[k]
print(all_edf_time)
print(seizures_dict)

c=collections.Counter(chlist)
print(c.most_common()[0])