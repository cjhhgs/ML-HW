from itertools import count
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn import linear_model

if __name__ == '__main__':
    
    # print('开始加载数据')
    cancer =load_breast_cancer()
    cancer_x=cancer.data
    cancer_y=cancer.target
    # print("加载完毕，数据大小：")
    # print(cancer_x.shape)
    # print(cancer_y.shape)
    # print("前5个数据：")
    # for i in range(5):
    #     print(cancer_x[i],cancer_y[i])
    
    x_train,x_test,y_train, y_test=train_test_split(cancer_x,cancer_y,test_size=0.2)
    model = linear_model.LogisticRegression(max_iter=10000)
    model.fit(x_train,y_train)
    y_pred = model.predict(x_test)

    i = 0
    count=0
    for y in y_test:
        if(y==y_pred[i]):
            count+=1
        i+=1
    
    score = count/i
    print("正确率：")
    print(score)