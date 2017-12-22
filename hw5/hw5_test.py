import sys
import os
import numpy as np
import pandas as pd
import keras.backend as K
from keras.layers import Input, Embedding, Flatten, Dense, Dropout
from keras.layers.merge import Dot, Add, Concatenate
from keras.models import load_model
from keras.callbacks import ReduceLROnPlateau, CSVLogger, EarlyStopping, ModelCheckpoint
from define import *


# <test.csv path> <prediction file path> <movies.csv path> <users.csv path>


movies, all_genres = read_movie(sys.argv[3])
genders, ages, occupations = read_user(sys.argv[4])

#train = pd.read_csv(DATA_DIR + '/train.csv').values


#userID, movieID, userGender, userAge, userOccu, movieGenre, Y = \
#        preprocess(train, genders, ages, occupations, movies)

n_users = len(occupations)
n_movies = len(movies)

'''

model = build_model(EMB_DIM=128,n_users=n_users,n_movies=n_movies)

cb = []
cb.append( ReduceLROnPlateau(factor=np.sqrt(0.1), cooldown=0, patience=5, min_lr=0.5e-6) )
cb.append( CSVLogger('./model/model.csv') )
cb.append( EarlyStopping(monitor='val_rmse', patience=30, verbose=1, mode='auto') )
cb.append( ModelCheckpoint(monitor='val_rmse', save_best_only=True, save_weights_only=False, \
                           mode='auto', filepath='./model/best.h5') )
history = model.fit([userID, movieID, userGender, userAge, userOccu, movieGenre], Y, \
                        epochs=5000, verbose=2, batch_size=10000, callbacks=cb, \
                        validation_split=0.05)
H = history.history
best_val = str( round(np.min(H['val_rmse']), 6) )
print('Best Val:', best_val)
'''
model = load_model(sys.argv[5], custom_objects={'rmse': rmse})
test = pd.read_csv(sys.argv[1]).values
  
userID, movieID, userGender, userAge, userOccu, movieGenre, _Y = \
        preprocess(test, genders, ages, occupations, movies)

result = model.predict([userID, movieID, userGender, userAge, userOccu, movieGenre])
   
ID =[i+1 for i in range(len(test))]
   
ans = pd.DataFrame({'TestDataID':ID,'rating':np.squeeze(result)})
ans.to_csv(sys.argv[2],index=False)
print('finished')
