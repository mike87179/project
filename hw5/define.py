import numpy as np
import keras.backend as K
from keras.layers import Input, Embedding, Flatten, Dense, Dropout
from keras.layers.merge import Dot, Add, Concatenate
from keras.regularizers import l2
from keras.models import Model

def build_model(EMB_DIM,n_users,n_movies):
    in_userID = Input(shape=(1,))      
    in_movieID = Input(shape=(1,),name='in_movieID')     
    in_userGender = Input(shape=(1,))  
    in_userAge = Input(shape=(1,))     
    in_userOccu = Input(shape=(21,))   
    in_movieGenre = Input(shape=(18,))  

    emb_userID = Embedding(n_users, EMB_DIM, embeddings_regularizer=l2(0.00001))(in_userID)
    emb_movieID = Embedding(n_movies, EMB_DIM, embeddings_regularizer=l2(0.00001))(in_movieID)
    vec_userID = Dropout(0.5)( Flatten()(emb_userID) )
    vec_movieID = Dropout(0.5)( Flatten(name='vec_movieID')(emb_movieID) )
    vec_userOccu = Dropout(0.5)( Dense(EMB_DIM, activation='linear')(in_userOccu) )
    vec_movieGenre = Dropout(0.5)( Dense(EMB_DIM, activation='linear')(in_movieGenre) )

    dot1 = Dot(axes=1)([vec_userID, vec_movieID])
    dot3 = Dot(axes=1)([vec_userID, vec_movieGenre])
    dot4 = Dot(axes=1)([vec_movieID, vec_userOccu])
    dot5 = Dot(axes=1)([vec_movieID, vec_movieGenre])
    dot6 = Dot(axes=1)([vec_userOccu, vec_movieGenre])

    con_dot = Concatenate()([dot1,       dot3, dot4, dot5, dot6, in_userGender, in_userAge])
    dense_out = Dense(1, activation='linear')(con_dot)

    emb2_userID = Embedding(n_users, 1, embeddings_initializer='zeros', embeddings_regularizer=l2(0.00001))(in_userID)
    emb2_movieID = Embedding(n_movies, 1, embeddings_initializer='zeros', embeddings_regularizer=l2(0.00001))(in_movieID)
    bias_userID = Flatten()(emb2_userID)
    bias_movieID = Flatten()(emb2_movieID)

    out = Add()([bias_userID, bias_movieID, dense_out])
    
    model = Model(inputs=[in_userID, in_movieID, in_userGender, in_userAge, in_userOccu, in_movieGenre], outputs=out)
    model.summary()

    model.compile(optimizer='adam', loss='mse', metrics=[rmse])
    return model

def to_categorical(index, categories):
    categorical = np.zeros(categories, dtype=int)
    categorical[index] = 1
    return list(categorical)


def read_movie(filename):

    def genre_to_number(genres, all_genres):
        result = []
        for g in genres.split('|'):
            if g not in all_genres:
                all_genres.append(g)
            result.append( all_genres.index(g) )
        return result, all_genres

    movies, all_genres = [[]] * 3953, []
    with open(filename, 'r', encoding='latin-1') as f:
        f.readline()
        for line in f:
            movieID, title, genre = line[:-1].split('::')
            genre_numbers, all_genres = genre_to_number(genre, all_genres)
            movies[int(movieID)] = genre_numbers
    
    categories = len(all_genres)
    for i, m in enumerate(movies):
        movies[i] = to_categorical(m, categories)


    return movies, all_genres


def read_user(filename):

    genders, ages, occupations = [[]]*6041, [[]]*6041, [ [0]*21 ]*6041
    categories = 21
    with open(filename, 'r', encoding='latin-1') as f:
        f.readline()
        for line in f:
            userID, gender, age, occu, zipcode = line[:-1].split('::')
            genders[int(userID)] = 0 if gender is 'F' else 1
            ages[int(userID)] = int(age)
            occupations[int(userID)] = to_categorical(int(occu), categories)

    return genders, ages, occupations



def preprocess(data, genders, ages, occupations, movies):

    if data.shape[1] == 4:

        np.random.seed(1019)
        index = np.random.permutation(len(data))
        data = data[index]

    userID = np.array(data[:, 1], dtype=int)
    movieID = np.array(data[:, 2], dtype=int)

    userGender = np.array(genders)[userID]
    userAge = np.array(ages)[userID]
    userOccu = np.array(occupations)[userID]
    movieGenre = np.array(movies)[movieID]

    std = np.std(userAge)
    userAge = userAge / std

    Rating = []
    if data.shape[1] == 4:

        Rating = data[:, 3].reshape(-1, 1)

    return userID, movieID, userGender, userAge, userOccu, movieGenre, Rating

def rmse(y_true, y_pred):
    y_pred = K.clip(y_pred, 1.0, 5.0)
    return K.sqrt(K.mean(K.pow(y_true - y_pred, 2)))


