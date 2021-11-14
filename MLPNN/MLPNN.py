from sklearn.neural_network import MLPClassifier
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

def accuracy(y_test,y_pred):
    i=0
    count=0
    for j in  y_test:
        k = y_pred[i]
        i=i+1
        if(k==j):
            count+=1
    return count/i

cancer =load_breast_cancer()
cancer_x=cancer.data
cancer_y=cancer.target
x_train,x_test,y_train, y_test=train_test_split(cancer_x,cancer_y,test_size=0.2)
    
print("start")
print("正确率：")

#'sgd' 指的是随机梯度下降；
clf = MLPClassifier(solver='sgd')
clf.fit(x_train,y_train)
y_pred = clf.predict(x_test)
print(accuracy(y_test,y_pred))

#'lbfgs' 是准牛顿方法族的优化器；
clf2 = MLPClassifier(solver='lbfgs')
clf2.fit(x_train,y_train)
y_pred2 = clf2.predict(x_test)
print(accuracy(y_test,y_pred2))

