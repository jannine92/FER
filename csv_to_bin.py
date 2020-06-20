import csv
import numpy as np
import ast
import os 
import tensorflow as tf
import sys

f = open('fer2013/fer2013.csv', newline='')
csv_reader = csv.reader(f)


''' Splits fer2013 CSV data into 3 separate csv data files and 
converts csv files to .bin
'''

header = next(csv_reader, None)

rows = [row for row in csv_reader]

# csv file for training 
train = [row[:-1] for row in rows if row[-1] == 'Training']
for r in train:
    r[0] = str(r[0] + ' ' + r[1])
    r[1] = ""
csv.writer(open('fer2013/train.csv', 'w', newline='')).writerows(train)


# csv file for public test 
test = [row[:-1] for row in rows if row[-1] == 'PublicTest']
for r in test:
    r[0] = str(r[0] + ' ' + r[1])
    r[1] = ""
csv.writer(open('fer2013/test.csv', 'w', newline='')).writerows(test)

# csv file for private test 
private_test = [row[:-1] for row in rows if row[-1] == 'PrivateTest']
for r in private_test:
    r[0] = str(r[0] + ' ' + r[1])
    r[1] = ""
csv.writer(open('fer2013/privatetest.csv', 'w', newline='')).writerows(private_test)

f.close()


# convert training csv to .bin 

trainFile = open('fer2013/train.csv', 'r', newline='')
csv_trainFile = csv.reader(trainFile)

with open('fer2013/train.bin', 'wb') as outFile:
    outCSVFile = csv.writer(outFile, dialect='excel')
    for row in csv_trainFile: 
        lineOut = np.fromstring(str(row[0]), dtype='uint8', sep=' ')
        lineOut.tofile(outFile)
    outFile.close()

trainFile.close()
print('convertion for training done')


# convert test csv to .bin
testFile = open('fer2013/test.csv')
csv_testFile = csv.reader(testFile)

with open('fer2013/test.bin', 'wb') as outFile:
    outCSVFile = csv.writer(outFile, dialect='excel')
    for row in csv_testFile: 
        lineOut = np.fromstring(str(row[0]), dtype='uint8', sep=' ')
        lineOut.tofile(outFile)
    outFile.close()

testFile.close()
print('convertion for test done')

# convert private test csv to .bin
pTestFile = open('fer2013/privatetest.csv')
csv_pTestFile = csv.reader(pTestFile)

with open('fer2013/privatetest.bin', 'wb') as outFile:
    outCSVFile = csv.writer(outFile, dialect='excel')
    for row in csv_pTestFile: 
        lineOut = np.fromstring(str(row[0]), dtype='uint8', sep=' ')
        lineOut.tofile(outFile)
    outFile.close()

pTestFile.close()
print('convertion for private test done')
