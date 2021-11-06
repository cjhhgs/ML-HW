from sklearn import datasets
from sklearn.model_selection import train_test_split#用于模型划分
import numpy
import matplotlib.pyplot as plt
from multiprocessing import Pool
import time


testsize=0.4    #测试集比例
max_k = 20      #测试最大的k值

#两向量的欧式距离
def distance_1(x,y):
    d = 0
    for i in range(0,4):
        d+=(x[i]-y[i])**2
    d = d**0.5
    
    return d

#两向量的曼哈顿距离
def distance_2(x,y):
    d=0
    for i in range(4):
        d+=abs(x[i]-y[i])
    return d

#knn函数，利用knn算法判断x的类型
def knn(X_train,Y_train,x,k):
    #计算待分类点x与训练集中每个点的距离，放入result中
    result=[]
    for i in X_train:
        distance = distance_2(x,i)
        result.append(distance)
    
    #将排序后的索引放入sortIndex
    res=numpy.array(result)
    sortIndex = res.argsort()
    

    #将前k个邻近的点的种类放入lable数组
    lable=[]
    for i in range(0,k):
        lable.append(Y_train[sortIndex[i]])
    
    #计每个种类出现的数量
    count = numpy.zeros(3)
    for i in lable:
        count[i]+=1

    
    #找出出现最多的种类，判断x类型
    max_x = count[0]
    res=0
    for i in range(1,3):
        if max_x<count[i]:
            max_x=count[i]
            res=i

    return res


#test函数，利用训练集判断测试集的类型，并返回正确率
def test(X_train,Y_train,X_test,Y_test,k):
    print("开始测试k=",k,"时knn算法的正确率")
    count=0
    num = int(150*testsize)
    for i in range(0,num):
        res = knn(X_train,Y_train,X_test[i],k)
        if res == Y_test[i]:
            count+=1
        
    accuracy = count/(150*testsize)
    print("结束，正确率为:",accuracy)
    return accuracy

def test_mtprocess(X_train,Y_train,X_test,Y_test,k):
    print("开始测试k=",k,"时knn算法的正确率")
    count = 0
    num = int(150*testsize)  #测试的图片数量

    pool = Pool(8)  #创建一个最大进程数为8的进程池
    
    #开始测试每张图片
    result=[]
    for i in range(num):
        arg = (X_train,Y_train,X_test[i],k)   #创建进程
        p = pool.apply_async(knn, args=arg)
        result.append(p)

    pool.close()
    pool.join()

    for i in range(num):
        p=result[i]
        res = p.get()       #返回结果
        if Y_test[i]==res:
            count+=1
    
    print("结束，正确率为:",count/num)
    return count/num

def main_1():
    print('开始加载数据')
    #加载数据
    iris = datasets.load_iris()
    X=iris.data
    Y=iris.target
    print('加载完毕')
    #划分
    X_train,X_test,Y_train,Y_test=train_test_split(X, Y, test_size=testsize, random_state=0)
    print('划分完毕')

    print('开始测试')
    t1=time.time()
    #取k值为1~20进行实验，测试正确率
    result=[]
    max_acc=0   #最高的正确率
    index=[]    #index[]存放正确率最高的k值
    for k in range(1,max_k+1):
        #res=test_mtprocess(X_train,Y_train,X_test,Y_test,k)   #多进程版本
        res=test(X_train,Y_train,X_test,Y_test,k)              #普通版本
        result.append(res)
        #找出正确率最大的k值
        if max_acc<res:
            max_acc=res
            index.clear()
            index.append(k)
        elif max_acc==res:
            index.append(k)
    t2=time.time()
    print('用时：',t2-t1)
    print("正确率：",result)
    print("k=",index,"时，正确率最高，为",max_acc)
    
    #绘图
    X=range(1,max_k+1)
    Y=numpy.array(result)
    plt.plot(X,Y)
    plt.xlabel('K')
    plt.ylabel('accuracy')
    plt.title('Result')
    plt.show()


def main_2():
    print('开始加载数据')
    #加载数据
    iris = datasets.load_iris()
    X=iris.data
    Y=iris.target
    print('加载完毕')
    #划分
    X_train,X_test,Y_train,Y_test=train_test_split(X, Y, test_size=testsize, random_state=0)
    print('划分完毕')

    print('开始测试')
    
    #取k值为1~20进行实验，测试正确率

    max_acc=0   #最高的正确率
    index=[]    #index[]存放正确率最高的k值
    pool=Pool(8)
    result=[]   #正确率列表
    joblist=[]  #进程列表
    t1=time.time()

    #创建多进程
    for k in range(1,max_k+1):
        arg=(X_train,Y_train,X_test,Y_test,k)
        p = pool.apply_async(test,args=arg)
        joblist.append(p)

    pool.close()
    pool.join()

    #获取结果
    for k in range(1,max_k+1):
        p = joblist[k-1]
        res = p.get()
        result.append(res)
        if max_acc<res:
            max_acc=res
            index.clear()
            index.append(k)
        elif max_acc==res:
            index.append(k)


    t2=time.time()
    print('用时：',t2-t1)
    print("正确率：",result)
    print("k=",index,"时，正确率最高，为",max_acc)
    
    #绘图
    X=range(1,max_k+1)
    Y=numpy.array(result)
    plt.plot(X,Y)
    plt.xlabel('K')
    plt.ylabel('accuracy')
    plt.title('Result')
    plt.show()

if __name__ == '__main__':

    main_1()

