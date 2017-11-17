import pandas as pd
import numpy as np
import sys
from keras.models import Sequential
from keras.layers import Convolution2D, MaxPooling2D, Flatten, Dense, Activation, Conv2D, Dropout, AveragePooling2D
from keras.utils import np_utils
from keras.layers.normalization import BatchNormalization
from keras.callbacks import ModelCheckpoint
from keras.layers.advanced_activations import LeakyReLU
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ReduceLROnPlateau, EarlyStopping


def read_data(filename,test=False):

    data = pd.read_csv(filename)
    index = list(data)
    y = data[index[0]]
        
    X = data[index[1]].str.split(expand=True).astype('float32').values
    
    if test == True:
        return X.reshape(-1,48,48,1) / 255
    else:
        return X.reshape(-1,48,48,1) / 255, np_utils.to_categorical(y, 7)

def cro_val(X, y, n):
    inds = np.random.permutation(X.shape[0])
    return X[inds[:int(n*X.shape[0])]], y[inds[:int(n*X.shape[0])]], X[inds[int(n*X.shape[0]):]], y[inds[int(n*X.shape[0]):]]



np.random.seed(0)

print('Reading data!')

X, y= read_data(sys.argv[1])

print('Cross validation!')


X, y, X_valid, y_valid = cro_val(X, y, 0.8)

datagen = ImageDataGenerator(rotation_range=30, width_shift_range=0.2, height_shift_range=0.2, zoom_range=0.2 , shear_range=0.2, horizontal_flip=True)

print('Model construction!')


model = Sequential()

model.add(Conv2D(64, kernel_size=(5, 5), input_shape=X.shape[1:4], padding='valid',kernel_initializer='glorot_normal'))
model.add(LeakyReLU(1./20))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
model.add(Dropout(0.25))

model.add(Conv2D(128, kernel_size=(3, 3), padding='same', kernel_initializer='glorot_normal'))
model.add(LeakyReLU(1./20))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
model.add(Dropout(0.3))

model.add(Conv2D(512, kernel_size=(5, 5), padding='same',kernel_initializer='glorot_normal'))
model.add(LeakyReLU(1./20))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
model.add(Dropout(0.35))

model.add(Conv2D(512, kernel_size=(3, 3), padding='same',kernel_initializer='glorot_normal'))
model.add(LeakyReLU(1./20))
model.add(BatchNormalization())
model.add(AveragePooling2D(pool_size=(2, 2), padding='same'))
model.add(Dropout(0.4))

model.add(Flatten())


model.add(Dense(512, activation='relu', kernel_initializer='glorot_normal'))
model.add(LeakyReLU(1./20))
model.add(BatchNormalization())
model.add(Dropout(0.5))


model.add(Dense(512, activation='relu', kernel_initializer='glorot_normal'))
model.add(BatchNormalization())
model.add(Dropout(0.5))

model.add(Dense(7, activation='softmax', kernel_initializer='glorot_normal'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])


lr_reducer = ReduceLROnPlateau(factor=np.sqrt(0.1), cooldown=0, patience=5, min_lr=0.5e-6)
earlyStopping = EarlyStopping(monitor='val_acc', patience=10, verbose=0, mode='auto')
checkpointer = ModelCheckpoint(filepath='./model/model_{epoch:05d}_{val_acc:.5f}.h5', save_best_only=True,period=1,monitor='val_acc')

print('Model fitting!')

history = model.fit_generator(datagen.flow(X, y, batch_size=128), steps_per_epoch=1000, epochs=8000, validation_data=(X_valid, y_valid), max_queue_size=100, callbacks=[lr_reducer, earlyStopping, checkpointer])


