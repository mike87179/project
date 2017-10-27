import numpy as np
import pandas as pd
import sys


def normalize(X):
    mean = np.min(X,axis=0)
    std = np.max(X,axis=0)
    res = (X - mean) / (std - mean)
    return res 

def sigmoid(x):
    res = 1 / (1.0 + np.exp(-x))
    return np.clip(res, 1e-8, 1-(1e-8))


def read_data(filename):
    return pd.read_csv(filename).as_matrix().astype('int')

def acc_count(y_pred,y):
    
    y_pred[y_pred>=0.5] = 1
    y_pred[y_pred<0.5] = 0
    return np.mean(1-np.abs(y-y_pred))

def add_bias(X):
    return np.concatenate((np.ones((X.shape[0],1)),X),axis=1)


def sel_feature(X):
    index = [0, 1, 3, 4, 5]
    
    
    X = np.concatenate((
                X, 
                X[:,index]** 2,
                X[:,index]** 3, 
                X[:,index] ** 4, 
                np.log(X[:,index] + 1e-10),
                
            ), axis=1)
    
    return X


def cross_validation(X, y, n):
    inds = np.random.permutation(X.shape[0])
    X_train = X[inds[:int(n*X.shape[0])]]
    y_train = y[inds[:int(n*X.shape[0])]]
    X_valid = X[inds[int(n*X.shape[0])]:]
    y_valid = y[inds[int(n*X.shape[0])]:]
    
    return X_train, y_train, X_valid, y_valid



def Logistic_regression(X,y,epoch=5000,lr=1,mom=1,lamb=1):         
    
    
    w = np.random.randn(X.shape[1],1).reshape(X.shape[1],1) / X.shape[1] / X.shape[0]
    G = np.random.randn(X.shape[1],1).reshape(X.shape[1],1) / X.shape[1] / X.shape[0]
    
    acc_record = []
    
    for i in range(1,epoch+1):
        
        last_step = 0
            
        y_pred = sigmoid(X.dot(w)) 
        diff = y_pred - y
        cost = -np.mean(y*np.log(y_pred+1e-20) + (1-y)*np.log(1-y_pred+1e-20))  
        grad = X.T.dot(diff) + mom*last_step + lamb*w
        G += grad**2
        w -= lr*grad / np.sqrt(G)
        last_step = lr*grad / np.sqrt(G)
        acc = acc_count(y_pred,y)
            
        if i % 200 == 0:
            print('epoch : %d | cost : %f | acc : %f' %(i,cost,acc))
            acc_record.append(acc)
        
    return w , acc

def test_predict(X, w):

    y = sigmoid(X.dot(w))
    y[y>=0.5] = 1
    y[y< 0.5] = 0    
    
    finalString = "id,label\n"
    with open(sys.argv[6], "w") as f:
        for i in range(len(y)) :
            finalString = finalString + str(i+1) + "," + str(int(y[i][0])) + "\n"
        f.write(finalString)
    
    
    return y


np.random.seed(0)    
X = read_data(sys.argv[3])
y = read_data(sys.argv[4])
X_test  = read_data(sys.argv[5])


temp = np.concatenate((X,X_test), axis=0)
temp = sel_feature(temp)

temp = normalize(temp)
temp = add_bias(temp)

X = temp[:X.shape[0],:]
X_test = temp[X.shape[0]:,:]

X, y, X_valid, y_valid = cross_validation(X, y, 0.9)


w , acc= Logistic_regression(X,y,epoch=5000,lr=1,mom=0,lamb=0)
 
acc_valid = acc_count(sigmoid(X_valid.dot(w)),y_valid)

print('train_acc : %f | valid_acc : %f' %(acc,acc_valid))


ans = test_predict(X_test, w)

