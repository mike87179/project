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
    return  pd.read_csv(filename).as_matrix().astype('int')


def train(X_all, Y_all, X_test):
    # Split a 10%-validation set from the training set
    X_train, Y_train = X_all, Y_all
    
    # Gaussian distribution parameters
    train_data_size = X_train.shape[0]
    cnt1 = 0
    cnt2 = 0

    mu1 = np.zeros((106,))
    mu2 = np.zeros((106,))
    for i in range(train_data_size):
        if Y_train[i] == 1:
            mu1 += X_train[i]
            cnt1 += 1
        else:
            mu2 += X_train[i]
            cnt2 += 1
    mu1 /= cnt1
    mu2 /= cnt2

    sigma1 = np.zeros((106,106))
    sigma2 = np.zeros((106,106))
    for i in range(train_data_size):
        if Y_train[i] == 1:
            sigma1 += np.dot(np.transpose([X_train[i] - mu1]), [(X_train[i] - mu1)])
        else:
            sigma2 += np.dot(np.transpose([X_train[i] - mu2]), [(X_train[i] - mu2)])
    sigma1 /= cnt1
    sigma2 /= cnt2
    shared_sigma = (float(cnt1) / train_data_size) * sigma1 + (float(cnt2) / train_data_size) * sigma2
    N1 = cnt1
    N2 = cnt2

    sigma_inverse = np.linalg.inv(shared_sigma)
    w = np.dot( (mu1-mu2), sigma_inverse)
    x = X_test.T
    b = (-0.5) * np.dot(np.dot([mu1], sigma_inverse), mu1) + (0.5) * np.dot(np.dot([mu2], sigma_inverse), mu2) + np.log(float(N1)/N2)
    a = np.dot(w, x) + b
    y = sigmoid(a)
    y = np.around(y)

    finalString = "id,label\n"
    with open(sys.argv[6], "w") as f:
        for i in range(len(y)) :
            finalString = finalString + str(i+1) + "," + str(int(y[i])) + "\n"
        f.write(finalString)
    
    return y

def load_data(filename1,filename2,filename3):
    X = read_data(filename1)
    y = read_data(filename2)
    X_test  = read_data(filename3)    
    return (X, y, X_test)

np.random.seed(0)    

X, y, X_test = load_data(sys.argv[3],sys.argv[4],sys.argv[5])

np.random.seed(1)    
temp = np.concatenate((X,X_test) , axis=0)
temp = normalize(temp)
X = temp[:X.shape[0]]
X_test = temp[X.shape[0]:]
y_pred = train(X, y, X_test)
print('finished!!!')
