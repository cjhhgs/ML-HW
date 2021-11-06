from sklearn.datasets import load_boston
if __name__ == '__main__':
    boston = load_boston()
    x,y = boston.data,boston.target
    print(x.shape)
    print(y.shape)
    for i in range(7):
        print(x[i],'   ',y[i])