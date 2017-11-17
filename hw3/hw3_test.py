import pandas as pd
import numpy as np
import sys
from keras.utils import np_utils
from keras.models import load_model

def read_data(filename,test=False):

    data = pd.read_csv(filename)
    index = list(data)
    y = data[index[0]]
        
    X = data[index[1]].str.split(expand=True).astype('float32').values
    
    if test == True:
        return X.reshape(-1,48,48,1) / 255
    else:
        return X.reshape(-1,48,48,1) / 255, np_utils.to_categorical(y, 7)


def out_data(X,model1,model2,model3,model4):

    y1 = model1.predict(X)
    y2 = model2.predict(X)
    y3 = model3.predict(X)
    y4 = model4.predict(X)

    y = y1 + y2 +y3 + y4

    pred = np.argmax(y, axis = 1)

    
    with open(sys.argv[2],'w') as f:
        f.write('id,label\n')
        for i in range(y.shape[0]):
            f.write( str(i) + ',' + str(pred[i]) + '\n')
    print('output finished!!!')

    return y

model1 = load_model('./model/2_model-00092-0.84587.h5')
model2 = load_model('./model/1_model-00167-0.70132.h5')
model3 = load_model('./model/model-01023-0.70724.h5')
model4 = load_model('./model/semi_best.h5')

X_test = read_data(sys.argv[1],test=True)

y = out_data(X_test,model1,model2,model3,model4)

