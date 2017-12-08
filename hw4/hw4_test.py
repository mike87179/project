import numpy as np
import pandas as pd
from gensim.models import word2vec
import re
from keras.models import load_model
import sys



test = pd.read_csv(sys.argv[1], sep="\n", skiprows=1, engine='python', header=None, names=['text'])
X_test = test['text'].str.split(',', 1 , expand=True)

test['text'] = X_test[1].apply(lambda x: x.lower())

max_length = 40
data_size = test.shape[0]
word_dim = 100


model = word2vec.Word2Vec.load(sys.argv[3])

tmp = []    

    
i = 0

for row in test.text.str.split():
    try:
        tmp.append(np.pad(model[row],((max_length-len(row),0), (0,0)), mode='constant'))
    except:
        string = []
        for ele in row:
            if ele in model:
                string.append(ele)
        if len(string) == 0 :
            tmp.append(np.zeros([max_length,word_dim]).astype('float32'))
        else:
            tmp.append(np.pad(model[string],((max_length-len(string),0), (0,0)), mode='constant'))
    i += 1
    print("\rtesting data : " + repr(i), end="", flush=True)
    
    
x_train = np.array(tmp)


model = load_model(sys.argv[4])
 
pre = model.predict_classes(x_train,batch_size=2048)

index = [i for i in range(pre.shape[0])]

answer = pd.DataFrame({'id':np.array(index), 'label':np.squeeze(pre)})
answer.to_csv(sys.argv[2], index=False)

print('finish')





