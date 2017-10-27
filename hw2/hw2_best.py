import numpy as np
import pandas as pd
import sys
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import VotingClassifier

def normalize(X):
    mean = np.mean(X,axis=0)
    std = np.std(X,axis=0)
    res = (X - mean) / std
    return res 


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

def read_data(filename):
    return pd.read_csv(filename).as_matrix().astype('float')

def add_bias(X):
    return np.concatenate((np.ones((X.shape[0],1)),X),axis=1)



def cross_validation(X, y, n):
    inds = np.random.permutation(X.shape[0])
    X_train = X[inds[:int(n*X.shape[0])]]
    y_train = y[inds[:int(n*X.shape[0])]]
    X_valid = X[inds[int(n*X.shape[0])]:]
    y_valid = y[inds[int(n*X.shape[0])]:]
    
    return X_train, y_train, X_valid, y_valid



def test_predict(X):

    y = eclf.predict(X)
    y[y>0.5] = 1
    y[y<= 0.5] = 0    
    
    finalString = "id,label\n"
    with open(sys.argv[6], "w") as f:
        for i in range(len(y)) :
            finalString = finalString + str(i+1) + "," + str(int(y[i])) + "\n"
        f.write(finalString)
    
    
    return y


np.random.seed(0)    
X = read_data(sys.argv[3])
y = read_data(sys.argv[4])
X_test  = read_data(sys.argv[5])
temp = np.concatenate((X,X_test) , axis=0)
temp = normalize(temp)
X = temp[:X.shape[0]]
X_test = temp[X.shape[0]:]
    

clf1 = GradientBoostingClassifier(learning_rate=0.1,n_estimators=600,verbose=True)
clf2 = GradientBoostingClassifier(learning_rate=0.1,n_estimators=700,verbose=True)
   
eclf = VotingClassifier(estimators=[('ada', clf1), ('aaa', clf2)], voting='hard')
eclf.fit(X, y.reshape(-1))
    
mse_train = mean_squared_error(y, eclf.predict(X))
print("train_acc: ",(1-mse_train))

y_pre = test_predict(X_test)

