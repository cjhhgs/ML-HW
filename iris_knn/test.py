import numpy
import random

file = open("iris.txt")
data = file.readlines()
print(data)

index=[]
for i in range(0,15):
    x = random.randint(i*10+0,i*10+9)
    index.append(x)
print(index)

testData=[]
for i in range(0,15):
    testData.append(data[index[i]])

for i in reversed(index):
    del data[i]

trainSet = []
for line in data:
    line = line.strip()
    temp=line.split(',')
    list = []
    for i in range(4):
        x = float(temp[i])
        list.append(x)
    list.append(temp[4])
    trainSet.append(list)

testSet=[]
for line in testData:
    line = line.strip()
    temp=line.split(',')
    list=[]
    for i in range(0,4):
        list.append(float(temp[i]))
    list.append(temp[4])
    testSet.append(list)

print(trainSet[1][0:5])