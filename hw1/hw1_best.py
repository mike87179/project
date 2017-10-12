import numpy as np
import sys
import csv
from numpy.linalg import inv




def process_data_in(filename, hour):
    
    data = []
    for i in range(18):
        data.append([])
        
    n_row = 0
    text = open(filename, 'r',encoding='big5') 
    row = csv.reader(text , delimiter=",")
    for r in row:
        if n_row != 0:
            for i in range(3,27):
                if r[i] != "NR":
                    data[(n_row-1)%18].append( float( r[i] ) )
                else:
                    data[(n_row-1)%18].append( float( 0 ) )	
        n_row =n_row+1
 
    text.close()
 
    train_x = []
    train_y = []
    
    for i in range(12):
        for j in range(480-hour):
            train_x.append( [] )
            for t in range(18):
                for s in range(hour):
                    train_x[(480-hour)*i+j].append( data[t][480*i+j+s] )
            train_y.append( data[9][480*i+j+hour] )
   

            
    
    train_x = np.array(train_x)
    train_y = np.array(train_y)

    train_y = np.reshape(train_y,(len(train_y),1))

    amb = np.copy(train_x[:,0:hour])  
    ch4 = np.copy(train_x[:,hour:2*hour]) 
    co = np.copy(train_x[:,2*hour:3*hour]) 
    nmhc = np.copy(train_x[:,3*hour:4*hour]) 
    no = np.copy(train_x[:,4*hour:5*hour]) 
    no2 = np.copy(train_x[:,5*hour:6*hour]) 
    nox = np.copy(train_x[:,6*hour:7*hour]) 
    o3 = np.copy(train_x[:,7*hour:8*hour])
    pm10 = np.copy(train_x[:,8*hour:9*hour])
    pm25 = np.copy(train_x[:,9*hour:10*hour])
    rf = np.copy(train_x[:,10*hour:11*hour])
    rh = np.copy(train_x[:,11*hour:12*hour]) 
    so2 = np.copy(train_x[:,12*hour:13*hour]) 
    thc = np.copy(train_x[:,13*hour:14*hour]) 
    wd_hr = np.copy(train_x[:,14*hour:15*hour]) 
    wind_dir = np.copy(train_x[:,15*hour:16*hour]) 
    wind_speed = np.copy(train_x[:,16*hour:17*hour]) 
    ws_hr = np.copy(train_x[:,17*hour:18*hour]) 

    #train_x = np.column_stack((amb,co,nmhc,no,no2,nox,o3,pm10,pm25,rh,so2,thc,wd_hr,ws_hr))
    #train_x = np.column_stack((no2, o3, pm10, pm25, so2, pm25**2,pm10**2, no))
    #train_x = np.column_stack((pm10,pm25,o3,wind_dir,wind_speed,wd_hr,ws_hr,rf,pm10**2,pm25**2))    
    train_x = np.column_stack((pm10,pm25,o3,wind_dir,wind_speed,wd_hr,ws_hr,rf,co,pm10**3,pm25**3,pm25*o3))
    #train_x = np.column_stack((pm10,pm25,o3,wind_dir,wind_speed,wd_hr,ws_hr,rf,pm10**2,pm25**2,pm25*o3))
    for i in range(471):
        train_x = np.delete(train_x,2826,0)
        train_y = np.delete(train_y,2826,0)

    rem = []
    for i in range(len(train_x[:,0])):
        if -1 in pm25[i,:] or -1 in train_y[i]:
            rem.append(i)
    
    
    train_x = np.delete(train_x,rem,0)
    train_y = np.delete(train_y,rem,0)
    
    mean =  np.mean(train_x,axis=0)
    std =   np.std(train_x,axis=0) 
    
    
    train_x = (train_x - mean ) / std        
    
    train_x = np.concatenate((train_x, np.ones((len(train_x[:,0]), 1))), axis=1)
    

    #train_x = pm25
    
    return train_x, train_y, mean, std

def LinReg_fit(X, y, X_test=None, y_test=None, lr=1e-7, lamb=0, epoch=10000, print_every=100, lamb1=0, momentum=0):
    """train Linear Regression by adagrad"""
    # initialize
   # W = np.random.randn(X.shape[1]).reshape(X.shape[1],1) / X.shape[1] / X.shape[0]
 #   W = np.load('weight_ini.npy').T
    W = inv(X.T.dot(X)).dot(X.T).dot(y)
    
    train_loss = []
    train_RMSE = []
    test_loss = []
    test_RMSE = []


    G = np.zeros(W.shape)

    for i in range(epoch):
   #     inds = []
        last_step = 0

   #     inds.append(np.random.permutation(X.shape[0]))
   #     diff = X[inds].dot(W) - y[inds]
        diff = X.dot(W) - y
        # calculate gradients
        w = np.array(W)
        w[w > 0] = 0.5
        w[w < 0] = -0.5
        grad_X = X.T.dot(diff)
        grad_regulariz = lamb * W 
        grad_first_order_reg = lamb1 * w 
        grad = grad_X + grad_regulariz + grad_first_order_reg

                # calculate update step
        G += grad**2
        delta_W = (grad + momentum * last_step) / np.sqrt(G)
        W -= lr * delta_W

                # reset variables
        last_step = delta_W
  

        objective = (((X.dot(W) - y)**2).sum() + lamb * (W**2).sum())
        RMSE = cal_RMSE(X, W, y)

        if X_test is not None and y_test is not None:
            # losses
            loss_X = ((X_test.dot(W) - y_test)**2).sum() 
            loss_reg = lamb * (W**2).sum() 
            loss_first_reg = lamb1 * (abs(W).sum())

            obj_t = loss_X + loss_reg + loss_first_reg
            RMSE_t = cal_RMSE(X_test, W, y_test)

            test_loss.append(obj_t)
            test_RMSE.append(RMSE_t)

        # print out the progress
        if i % print_every == 0:
            if X_test is not None and y_test is not None:
                print('\tepoch: [%d]; loss: [%.4f]; RMSE: [%.4f]; RMSE_test: [%.4f]' %
                      (i, objective, RMSE, RMSE_t))
            else:
                print('\tepoch: [%d]; loss: [%.4f]; RMSE: [%.4f]' %
                      (i, objective, RMSE))

        train_loss.append(objective)
        train_RMSE.append(RMSE)

    print('final obj: %.4f' % train_loss[-1])

    return W, train_loss, train_RMSE, test_loss, test_RMSE


def cal_RMSE(X, W, y):
    """Calculate the RMSE"""
    return np.sqrt(((X.dot(W) - y) ** 2).sum() / len(y))


def process_data_out(filename, W, hour, mean, std):

    test = []


    for i in range(240):
        test.append([])


    n_row = 0
    text = open('test.csv', 'r') 
    row = csv.reader(text , delimiter=",")
    for r in row:
        for i in range(11-hour,11):
            if r[i] != "NR":
                test[n_row//18].append( float( r[i] ) )
            else:
                test[n_row//18].append( float( 0 ) )	
        n_row += 1
    
    
    text.close()
    test = np.array(test)


    amb = np.copy(test[:,0:hour])  
    ch4 = np.copy(test[:,hour:2*hour]) 
    co = np.copy(test[:,2*hour:3*hour]) 
    nmhc = np.copy(test[:,3*hour:4*hour]) 
    no = np.copy(test[:,4*hour:5*hour]) 
    no2 = np.copy(test[:,5*hour:6*hour]) 
    nox = np.copy(test[:,6*hour:7*hour]) 
    o3 = np.copy(test[:,7*hour:8*hour])
    pm10 = np.copy(test[:,8*hour:9*hour])
    pm25 = np.copy(test[:,9*hour:10*hour])
    rf = np.copy(test[:,10*hour:11*hour])
    rh = np.copy(test[:,11*hour:12*hour]) 
    so2 = np.copy(test[:,12*hour:13*hour]) 
    thc = np.copy(test[:,13*hour:14*hour]) 
    wd_hr = np.copy(test[:,14*hour:15*hour]) 
    wind_dir = np.copy(test[:,15*hour:16*hour]) 
    wind_speed = np.copy(test[:,16*hour:17*hour]) 
    ws_hr = np.copy(test[:,17*hour:18*hour]) 


    #test = np.column_stack((no2, o3, pm10, pm25, so2, pm25**2,pm10**2, no))
    #test = np.column_stack((pm10,pm25,o3,wind_dir,wind_speed,wd_hr,ws_hr,rf,pm10**2,pm25**2))
    test = np.column_stack((pm10,pm25,o3,wind_dir,wind_speed,wd_hr,ws_hr,rf,co,pm10**3,pm25**3,pm25*o3))
    
    test = (test - mean) / std
    
    
    test = np.concatenate((test, np.ones((len(test[:,0]), 1))), axis=1)
    
    y_hat = np.dot(test,W)


    finalString = "id,value\n"
    for i in range(240) :
        finalString = finalString + "id_" + str(i) + "," + str(y_hat[i][0]) + "\n"
    f = open(sys.argv[2], "w")
    
    f.write(finalString)
    f.close()
    
    return y_hat

    
#--------------------main function------------------------------------------- 


    
hr = 9
X_train, y_train,mean ,std = process_data_in(sys.argv[3], hr)  
#W, train_loss, train_RMSE, valid_loss, valid_RMSE = LinReg_fit(X_train, y_train,lr=0.1,epoch=20000,print_every=200)

#np.save('weight_best',W)

W = np.load('weight_best.npy')

loss = ((X_train.dot(W) - y_train)**2).sum() 
RMSE = cal_RMSE(X_train, W, y_train)


y_hat = process_data_out(sys.argv[1], W, hr,mean,std)

