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

svc_clf = SVC(kernel='linear')
svc_clf.fit(x_train,y_train)
y_pred = svc_clf.predict(x_test)
print("正确率为：%f"% accuracy(y_test,y_pred))

svc_clf2 = SVC(kernel='poly', degree=4)
svc_clf2.fit(x_train,y_train)
y_pred2 = svc_clf2.predict(x_test)
print("正确率为：%f"%accuracy(y_test,y_pred2))

svc_clf3 = SVC(kernel='rbf')
svc_clf3.fit(x_train,y_train)
y_pred3 = svc_clf3.predict(x_test)
print("正确率为：%f"%accuracy(y_test,y_pred3))

svc_clf4 = SVC(kernel='sigmoid')
svc_clf4.fit(x_train,y_train)
y_pred4 = svc_clf4.predict(x_test)
print("正确率为：%f"%accuracy(y_test,y_pred4))