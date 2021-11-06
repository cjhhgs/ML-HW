from sklearn import linear_model
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
import sklearn.linear_model
from sklearn.metrics import mean_squared_error, mean_absolute_error

def LinearRegressionPred(x_train,x_test,y_train,y_test):

    model = linear_model.LinearRegression()      #选择模型
    model.fit(x_train,y_train)      #拟合
    y_pred = model.predict(x_test)  #预测
    
    loss1 = mean_squared_error(y_test, y_pred)       #计算损失函数
    loss2 = mean_absolute_error(y_test, y_pred)
    
    return loss1,loss2
 
def RidgeRegressionPred(x_train,x_test,y_train,y_test):
    model = linear_model.Ridge()    #选择模型
    model.fit(x_train,y_train)      #拟合
    y_pred = model.predict(x_test)  #预测
    
    loss1 = mean_squared_error(y_test, y_pred)       #计算损失函数
    loss2 = mean_absolute_error(y_test, y_pred)
    
    return loss1,loss2

def LassoRegressionPred(x_train,x_test,y_train,y_test):
    model = linear_model.Lasso()    #选择模型
    model.fit(x_train,y_train)      #拟合
    y_pred = model.predict(x_test)  #预测
    
    loss1 = mean_squared_error(y_test, y_pred)       #计算损失函数
    loss2 = mean_absolute_error(y_test, y_pred)
    
    return loss1,loss2

if __name__ == '__main__':
    boston = load_boston()
    x,y = boston.data,boston.target

    testNum=10

    mseLinear=0
    maeLinear=0
    mseRidge=0
    maeRidge=0
    mseLasso=0
    maeLasso=0

    for i in range(testNum):
        x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2)
        t1,t2 = LinearRegressionPred(x_train, x_test, y_train, y_test)
        mseLinear+=t1
        maeLinear+=t2

        t1,t2 = RidgeRegressionPred(x_train, x_test, y_train, y_test)
        mseRidge+=t1
        maeRidge+=t2

        t1,t2 = LassoRegressionPred(x_train, x_test, y_train, y_test)
        mseLasso+=t1
        maeLasso+=t2
    

    print("结果展示：")
    print("     Linear      Ridge       lasso")
    print("mse: %.2f       %.2f       %.2f "%   (mseLinear/testNum,mseRidge/testNum,mseLasso/testNum))
    print("mae: %.2f        %.2f        %.2f "% (maeLinear/testNum,maeRidge/testNum,maeLasso/testNum))



