import csv
import os
with open('csv_data_devtest.csv') as f:
    reader = csv.reader(f)
    l=[row for row in reader]
for i in l:
    i[0]=os.path.basename(i[0])
    i[0]=i[0].replace('.tse','.edf')
with open('csv_data_devtest.csv', 'w') as f:
    writer = csv.writer(f)
    writer.writerow(l)