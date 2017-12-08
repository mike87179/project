import numpy as np
import pandas as pd
from gensim.models import word2vec
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, BatchNormalization, Masking, GRU, Bidirectional
from keras.callbacks import ReduceLROnPlateau, CSVLogger, EarlyStopping, ModelCheckpoint
import re
from keras.models import load_model
from keras.layers.advanced_activations import LeakyReLU
import sys



data = pd.read_csv(sys.argv[1], sep="\+\+\+\$\+\+\+", engine='python', header=None, names=['label', 'text'])

data['text'] = data['text'].apply(lambda x: x.lower())
y_train = data['label'].values


max_length = 40
data_size = data.shape[0]
word_dim = 100

model = word2vec.Word2Vec.load(sys.argv[3])


tmp = []    

i = 0    


for row in data.text.str.split():
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
    print("\rtraining data : " + repr(i), end="", flush=True)


x_train = np.array(tmp)




model = Sequential()
model.add(Masking(input_shape=x_train.shape[1:]))
model.add(Bidirectional(GRU(128, activation='tanh', dropout=0.3)))
model.add(BatchNormalization())
model.add(Dense(64))
model.add(LeakyReLU(1./20))
model.add(BatchNormalization())
model.add(Dropout(0.3))
model.add(Dense(1,activation='sigmoid'))

model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

lr_reducer = ReduceLROnPlateau(factor=np.sqrt(0.1), cooldown=0, patience=5, min_lr=0.5e-6)
earlyStopping = EarlyStopping(monitor='val_acc', patience=5, verbose=0, mode='auto')
csv_logger = CSVLogger('./model.csv')
checkpointer = ModelCheckpoint(filepath='./model.h5', save_best_only=True,period=1,monitor='val_acc')


history = model.fit(x_train, y_train, batch_size=512, epochs=8000, verbose=2,validation_split=0.1,
          callbacks=[lr_reducer, earlyStopping, csv_logger, checkpointer])




