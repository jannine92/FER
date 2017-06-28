import csv
import tensorflow as tf
import numpy as np


f = open('fer2013/Input_Dataset/fer2013.csv', newline='')
csv_reader = csv.reader(f)


# https://github.com/npinto/fer2013/blob/master/convert_fer2013.py
# splits CSV data into 3 separate csv data files

header = next(csv_reader, None)

rows = [row for row in csv_reader]


# split r[1] so that every value is in an own cell
with open('fer2013/Input_Dataset/train.csv', 'w', newline='') as output_csv_train:
    csv_writer_train = csv.writer(output_csv_train, delimiter=',', quoting=csv.QUOTE_NONE, escapechar='')
    train = [row[:-1] for row in rows if row[-1] == 'Training']
    #train_array = []
    for r in train:
        #r[0] = str(r[0] + ' ' + r[1])
        r[1] = r[1].replace(' ', ",")
        r[0] = str(r[0] + ',' + r[1])
        r[1] = ""
        row_as_list = str(r[0]).split(',')
        #r = r[0].lower().split(',')
        #r = r[0].split(',')
        #csv_writer.writerow([x.split(',') for x in r[0]])
        csv_writer_train.writerow(row_as_list)
        #print("r0: ", r[0], " r1: ", r[1])
        #r[0] = float(r[0])
        # image = map(float, r[0].split())
        #pixel_array = np.array(image)
        #csv_writer.writerow(pixel_array)
        #train_array.append(pixel_array)
    
    # csv_writer.writerows(train)
    #np.savetxt('fer2013/train1.csv', train_array, delimiter=',')
#csv.writer(open('fer2013/train1.csv', 'w', newline=''), quoting=csv.QUOTE_MINIMAL).writerows(x.split(',') for x in train)


# csv file for public test
with open('fer2013/Input_Dataset/test.csv', 'w', newline='') as output_csv_test:
    csv_writer_test = csv.writer(output_csv_test, delimiter=',', quoting=csv.QUOTE_NONE, escapechar='')
    test = [row[:-1] for row in rows if row[-1] == 'PublicTest']
    for r in test:
        r[1] = r[1].replace(' ', ",")
        r[0] = str(r[0] + ',' + r[1])
        r[1] = ""
        row_as_list = str(r[0]).split(',')
        csv_writer_test.writerow(row_as_list)
    

# csv file for private test
with open('fer2013/Input_Dataset/privatetest.csv', 'w', newline='') as output_csv_ptest:
    csv_writer_ptest = csv.writer(output_csv_ptest, delimiter=',', quoting=csv.QUOTE_NONE, escapechar='')
    ptest = [row[:-1] for row in rows if row[-1] == 'PrivateTest']
    for r in ptest:
        r[1] = r[1].replace(' ', ",")
        r[0] = str(r[0] + ',' + r[1])
        r[1] = ""
        row_as_list = str(r[0]).split(',')
        csv_writer_ptest.writerow(row_as_list)
    

f.close()


# csv file save all separated with ',': (Format: 0 | 70,80,82,72,...)
"""train = [row[:-1] for row in rows if row[-1] == 'Training']
for r in train:
    r[0] = str(r[0] + ' ' + r[1])
    r[1] = r[0].replace(' ', ",")
    r[1] = ""
csv.writer(open('fer2013/train1.csv', 'w', newline='')).writerows(train)"""



