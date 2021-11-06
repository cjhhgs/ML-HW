import numpy
import random


#在给定的150项数据中均匀随机取出15项作为测试集
def separateData(fileName):
    file = open(fileName)
    data = file.readlines()

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

    trainSet=[]
    for line in data:
        list=line.split(',')
        trainSet.append(list)
    
    testSet=[]
    for line in testData:
        list=line.split(',')
        testSet.append(list)

    return trainSet,testSet

def loadData():
    print(1)

#计算两向量欧式距离
def distance(x,y):
    d = 0
    for i in range(0,4):
        d+=(x[i]-y[i])**2
    d = d**0.5
    return d

def knn(trainSet,x,k):
    #计算待分类点x与训练集中每个点的距离，放入result中
    result=[]
    for i in trainSet:
        x = distance(x[0:3],i[0:3])
        result.append(x)
    
    #将排序后的索引放入sortIndex
    sortIndex = result.argsort()

    #将前k个邻近的点的种类放入lable数组
    lable=[]
    for i in range(0,k):
        lable.append(trainSet[sortIndex[i]][4])
    
    #计算每个种类出现的数量，放入字典类型count
    count = { 'Iris-setosa' : 0 , 'Iris-versicolor' : 0 , 'Iris-virginica' : 0 }
    for i in lable:
        count[i]+=1
    
    #找出出现最多的种类，判断x类型
    res = 'Iris-setosa'
    if count[res] < count['Iris-versicolor']:
        res = 'Iris-versicolor'
    
    if count[res] < count['Iris-virginica']:
        res = 'Iris-virginica'

    #判断knn算法的结果是否正确，返回结果
    if res == x[4]:
        return True
    else :
        return False

def test(fileName,k):
    trainSet,testSet=separateData(fileName)
    right=0
    for x in testSet:
        res = knn(trainSet,x,k)
        if res == True:
            right+=1
    res = right/15

    return res


if __name__ == '__main__':
    res = test("iris.txt",3)
    print(res)
