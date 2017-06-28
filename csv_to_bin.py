import csv
import numpy as np
import ast
import os 
import tensorflow as tf
import sys

f = open('fer2013/fer2013.csv', newline='')
csv_reader = csv.reader(f)


# https://github.com/npinto/fer2013/blob/master/convert_fer2013.py
# splits CSV data into 3 separate csv data files

header = next(csv_reader, None)

rows = [row for row in csv_reader]

# csv file for training 
train = [row[:-1] for row in rows if row[-1] == 'Training']
for r in train:
    r[0] = str(r[0] + ' ' + r[1])
    r[1] = ""
csv.writer(open('fer2013/train.csv', 'w', newline='')).writerows(train)
#csv.writer(open('fer2013/train.csv', 'w', newline='')).writerows([header[:-1]] + train)


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
        #print('row[0]: ' + str(row[0]))
        #lineParsed = ast.literal_eval(str(row[0]))
        #print('line parsed: ' ,str(lineParsed))
        lineOut = np.fromstring(str(row[0]), dtype='uint8', sep=' ')
        #print('lineOut: ', lineOut)
        #bytearray = bytes(lineOut)
        #outFile.writerows(bytearray)
        #lineOut = np.array(row,'uint8')
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
        #lineParsed = ast.literal_eval(str(row))
        #print(str(row))
        lineOut = np.fromstring(str(row[0]), dtype='uint8', sep=' ')
        #print('line: ',str(lineOut))
        #lineOut = np.array(row,'uint8')
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
        #lineParsed = ast.literal_eval(str(row))
        lineOut = np.fromstring(str(row[0]), dtype='uint8', sep=' ')
        #lineOut = np.array(row,'uint8')
        lineOut.tofile(outFile)
    outFile.close()

pTestFile.close()
print('convertion for private test done')

#trainInput = csv.reader(open('fer2013/train.csv', 'r'))

"""with open('fer2013/train.bin', 'wb') as outFile:
    outCSVFile = csv.writer(outFile, dialect='excel')
    for line in trainInput:
        #lineParsed = ast.literal_eval(line)
        lineOut = np.array(line,'float16')
        lineOut.tofile(outFile)
    outFile.close()
#trainInput.close()
print('convertion done')"""

"""
test = [row[:-1] for row in rows if row[-1] == 'PublicTest']
csv.writer(open('fer2013/test.csv', 'w+')).writerows([header[:-1]] + test)
#print(test)

private_test = [row[:-1] for row in rows if row[-1] =='PrivateTest']
csv.writer(open('fer2013/privatetest.csv', 'w+')).writerows([header[:-1]] + private_test)
#print(private_test)
"""
"""
local_directory = os.path.dirname(os.path.abspath(__file__))+ '\fer2013' + '/'

DATASET_CSV = local_directory + 'fer2013.csv'
DATASET_TRAIN_BIN = local_directory + 'fer2013_train.bin'
DATASET_TEST_BIN = local_directory + 'fer2013_test.bin'
DATASET_TEST2_BIN = local_directory + 'fer2013_test2.bin'


# https://stackoverflow.com/questions/12799454/convert-a-text-csv-binary-file-and-get-a-random-line-from-it-in-python-without-r
# this converts the file from text CSV to bin
    
with open("output.bin", 'wb') as outFile:
    outCSVFile = csv.writer(outFile, dialect='excel')
    for line in csv_reader:
        lineParsed = ast.literal_eval(line)
        lineOut = np.array(lineParsed,'float16')
        lineOut.tofile(outFile)
    outFile.close()

csv_reader.close()


height, width = 48

with open(csv_reader, 'rb') as csvfile:
        datareader = csv.reader(csvfile, delimiter=',')
        headers = datareader.next()
        print(headers)
        
        for row in datareader:
            emotion = row[0]
            pixels = map(tf.uint8, row[1].split()) #or: int instead of uint8
            usage = row[2]
            pixel_array = np.asarray(pixels)
            
            
            image = pixel_array.reshape(height, width)"""

# link for ruby: https://github.com/lijian8/emotion-recognition-1/blob/master/uint8-to-binary.rb