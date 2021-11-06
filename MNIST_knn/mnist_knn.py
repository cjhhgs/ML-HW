import numpy as np
import struct
import matplotlib.pyplot as plt
from multiprocessing import Process,Pool
import time


TRAIN_NUM = 4000
TEST_NUM = 100



#解析图片文件的通用函数
def decodeImages(file):
    
    # 读取二进制数据
    file = open(file, 'rb')
    data = file.read()

    # 解析文件头信息，依次为魔数、图片数量、每张图片高、每张图片宽
    offset = 0
    fmt = '>iiii' #unpack_from()函数中读取格式，表示读取4个32bit int型
    magicNumber, numImages, numRows, numCols = struct.unpack_from(fmt, data, offset)
    print('魔数:%d, 图片数量: %d张, 图片大小: %d*%d' % (magicNumber, numImages, numRows, numCols))

    # 解析图片数据集
    imageSize = numRows * numCols
    offset += 16  #指针位置移动。
    fmt = '>' + str(imageSize) + 'B'  #读取28*28个unsigned char类型的数据，即一个图片的所有像素
    images = np.empty((numImages, numRows, numCols)) #创建数组，存放图片像素信息
    #plt.figure()
    for i in range(numImages):
        if (i + 1) % 10000 == 0:
            print('已解析 %d' % (i + 1) + '张')
        #解析第i张图片，放入数组images中
        imageData = struct.unpack_from(fmt,data,offset)
        imageData=np.array(imageData)
        images[i]=imageData.reshape((numRows,numCols))
        offset += struct.calcsize(fmt) #指针移动

    return images


#解析标签文件的通用函数
def decodeLables(file):
    
    # 读取二进制数据
    file = open(file, 'rb')
    data = file.read()

    # 解析文件头信息，依次为魔数、图片数量
    offset = 0
    fmt = '>ii' #unpack_from()函数中读取格式，表示读取2个32bit int型
    magicNumber, numImages = struct.unpack_from(fmt, data, offset)
    print('魔数:%d, 图片数量: %d张' % (magicNumber, numImages))

    # 解析标签数据集
    offset += 8  #指针位置移动。
    fmt = '>B'  #读取1个unsigned char类型的数据，即数字
    lables = np.empty(numImages) #创建数组，存放图片像素信息
    #plt.figure()
    for i in range(numImages):
        #解析第i张图片，放入数组images中
        lableData = struct.unpack_from(fmt,data,offset)
        lables[i]=lableData[0]
        offset += struct.calcsize(fmt) #指针移动

    return lables

#计算两矩阵的欧氏距离的函数
def distance1(x,y):
    d=0
    for i in range(28):
        for j in range(28):
            temp = x[i][j]-y[i][j]
            d += temp**2
    d=d**0.5
    return d

#计算曼哈顿距离
def distance2(x,y):
    d=0
    for i in range(28):
        for j in range(28):
            temp = x[i][j]-y[i][j]
            d += abs(temp)
    return d

#利用knn算法推测x的数字，参数=（训练集图片，训练集标签，待测图片，k，待测图片序号）
def knn(trainImages,trainLables,x,k,procnum):
    
    print('第',procnum+1,'张图片测试开始')
    distances = []
    #计算x与trainImages中每个点的距离，放入数组
    for i in trainImages:
        temp = distance2(i,x)
        distances.append(temp)

    #找出前k个最小的值
    Lst = distances[:]                        
    indexs = []
    for i in range(k):
        index_i = Lst.index(min(Lst))    #得到列表的最小值，并得到该最小值的索引
        indexs.append(index_i)           #记录最小值索引
        Lst[index_i] = float('inf')      #将遍历过的列表最小值改为无穷大，下次不再选择
    

    #取前k个标签
    lables = []
    for i in range(k):
        lables.append(trainLables[indexs[i]])
    

    #为每个标签计数
    count = np.zeros(10)
    for i in lables:
        count[int(i)]+=1
    

    max = count[0]
    index = 0
    for i in range(1,10):
        if max < count[i]:
            max=count[i]
            index = i
    
    
    print('第',procnum+1,'张图片测试结束，结果为:',index)
    return index

#test函数，计算预测准确率，参数=（训练集图片，训练集标签，测试集图片，测试集标签，k）
def test(trainImages,trainLables,testImages,testLables,k):
    count = 0
    num = TEST_NUM  #测试的图片数量

    pool = Pool(8)  #创建一个最大进程数为8的进程池
    
    #开始测试每张图片
    result=[]
    for i in range(num):
        arg = (trainImages,trainLables,testImages[i],k,i)   #创建进程
        p = pool.apply_async(knn, args=arg)
        result.append(p)

    pool.close()
    pool.join()

    for i in range(num):
        p=result[i]
        res = p.get()       #返回结果
        print('第',i+1,'张图片预测结果为：',res,',  正确结果为：',int(testLables[i]))
        if testLables[i]==res:
            print('判断正确')
            count+=1
        else :
            print('判断错误')

    return count/num


    



if __name__ == '__main__':

    #获取训练集、测试集数据
    trainImages = decodeImages('train-images.idx3-ubyte')
    trainLables = decodeLables('train-labels.idx1-ubyte')
    testImages = decodeImages('t10k-images.idx3-ubyte')
    testLables = decodeLables('t10k-labels.idx1-ubyte')
    result=[]
    t1=time.time()
    for k in range(1,11):
        print('k=',k,'，开始测试')
        temp = test(trainImages[:TRAIN_NUM],trainLables[:TRAIN_NUM],testImages[:TEST_NUM],testLables[:TEST_NUM],k)
        print('测试结束')
        print('k=',k,'的正确率为：',temp)
        result.append(temp)
    
    t2=time.time()
    print('总用时(s)：',t2-t1)
    print(result)

    Y=np.array(result)
    X=range(1,11)
    plt.plot(X,Y)
    plt.xlabel('K')
    plt.ylabel('accuracy')
    plt.title('Result')
    plt.show()


    
#  # 查看前十个数据及其标签以读取是否正确
# for i in range(10):
#     print(trainLables[i])
#     plt.imshow(trainImages[i], cmap='gray')
#     plt.pause(0.000001)
#     plt.show()
# print('done')