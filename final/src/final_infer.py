import os
import sys
import pickle as pk
import numpy as np
from keras.preprocessing.sequence import pad_sequences
from keras.models import load_model
import pandas as pd


with open('tokenizer.pk', 'rb') as f:
    tokenizer = pk.load(f)
TEST_DATA = sys.argv[1]
TEST_CAPTION = sys.argv[2]

with open('info.pk','rb') as f:
    info = pk.load(f) 
max_caption_len = info['max_caption_length']
max_test_length = info['max_train_x_length']

test_y = pd.read_csv(TEST_CAPTION,header=None)
test_y = '\t ' + test_y.astype(str) + ' \n'
test_y = test_y.stack().values
test_y = pd.DataFrame(test_y,columns=['text'])
print('========================================')
print('Fitting tokenizer', end=' ', flush=True)
test_y = tokenizer.texts_to_sequences(test_y['text'])
test_y = pad_sequences(test_y, maxlen=max_caption_len+1, padding='post', truncating='post')
decoder_input_data = test_y[:, :-1]
decoder_target = np.expand_dims(test_y[:, 1:], axis=-1)
print('[finished]', flush=True)
print('========================================')
print('Reading mfcc', end=' ', flush=True)
test = np.array(np.load(TEST_DATA)) 

with open('mean.pk','rb') as f:
    mean = pk.load(f)
with open('std.pk','rb') as f:
    std = pk.load(f)    
   
test_ = np.zeros((test.shape[0], max_test_length, 39), dtype='float32')
for i in range(len(test)):
    for j in range(len(test[i])):
        for k in range(39):
            test_[i, j, k] = (test[i][j][k] - mean[k]) / std[k]
test = test_
print('[finished]', flush=True)
print('========================================')
print('Loading model', end=' ', flush=True)
model = load_model('model.h5')
print('[finished]', flush=True)
print('========================================')
print('Predicting result', end=' ', flush=True)

f = open(sys.argv[3], 'w')
f.write('id,answer\n')
for j in range(2000):
    pre = [0] * 4
    pre[0] = model.evaluate([test[j:j+1],decoder_input_data[4*j:4*j+1]],decoder_target[4*j:4*j+1],verbose=2)[0]
    pre[1] = model.evaluate([test[j:j+1],decoder_input_data[4*j+1:4*j+2]],decoder_target[4*j+1:4*j+2],verbose=2)[0] 
    pre[2] = model.evaluate([test[j:j+1],decoder_input_data[4*j+2:4*j+3]],decoder_target[4*j+2:4*j+3],verbose=2)[0] 
    pre[3] = model.evaluate([test[j:j+1],decoder_input_data[4*j+3:4*j+4]],decoder_target[4*j+3:4*j+4],verbose=2)[0]
    ans = pre.index(min(pre))
    f.write('%d,%d\n' % (j+1, ans))
    print("\rtesting data : " + str(j) + '/' + str(test.shape[0]), end="", flush=True)
f.close()
print("\rtesting data : " + str(test.shape[0]) + '/' + str(test.shape[0]), end="", flush=True)
print(' [finished]', flush=True)



