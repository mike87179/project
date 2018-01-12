import numpy as np
import pandas as pd
import sys
from keras.models import Model
from keras.layers import Dense, Input
from sklearn.decomposition import PCA
from keras.models import load_model
from sklearn.cluster import KMeans


def build_model(data, encoding_dim=32):
    input_img = Input(shape=(784,))

    encoded = Dense(256, activation='tanh')(input_img)
    encoded = Dense(128, activation='tanh')(encoded)
    encoded = Dense(64, activation='tanh')(encoded)
    encoder_output = Dense(encoding_dim)(encoded)

    decoded = Dense(64, activation='tanh')(encoder_output)
    decoded = Dense(128, activation='tanh')(decoded)
    decoded = Dense(256, activation='tanh')(decoded)
    decoded = Dense(784, activation='linear')(decoded)

    autoencoder = Model(input=input_img, output=decoded)   
    autoencoder.compile(optimizer='adam', loss='mse')
    autoencoder.fit(data, data, epochs=30, batch_size=1024, shuffle=False)
   
    encoder = Model(input=input_img, output=encoder_output)	
        
    return encoder

def ans_out(test_data, labels, filename):
    
    test_data['y1'] = test_data['image1_index'].apply(lambda x:labels[x])
    test_data['y2'] = test_data['image2_index'].apply(lambda x:labels[x])
    test_data['y1+y2'] = test_data['y1'] + test_data['y2']
    test_data['Ans'] = test_data['y1+y2'].apply(lambda x: 1 if x!=1 else 0)

    ID = test_data['ID']
    Ans = test_data['Ans']
    
    out = pd.concat((ID,Ans),axis=1)

    out.to_csv(filename,index=False)
    print('finish!!!')
    

def main(args):
    train = np.load(args[1]) / 255. - 0.5
    test = pd.read_csv(args[2])
    model = build_model(train)	
    reduce = model.predict(train)
    data = PCA(n_components=2).fit(reduce)
    reduced_data = data.transform(reduce)
    kmeans_model = KMeans(n_clusters=2, random_state=1).fit(reduced_data)    
    labels = kmeans_model.labels_
    ans_out(test,labels,args[3])

if __name__ == '__main__':
    main(sys.argv)
