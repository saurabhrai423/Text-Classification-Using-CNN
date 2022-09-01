### Model 2
from numpy import array
from tensorflow.keras.preprocessing.text import one_hot
from tensorflow.keras.preprocessing.sequence import pad_sequences


from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense, Input, GlobalMaxPooling1D , Dropout
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Embedding , Concatenate
from tensorflow.keras.models import Model
import pickle
from tensorflow.keras.utils import to_categorical
import tensorflow as tf

import pandas


token = tf.keras.preprocessing.text.Tokenizer(filters='!"#$%&()*+,-./:;<=>?@[\\]^`{|}~\t\n' ,char_level=True, oov_token = True )

token.fit_on_texts(X_train)
vocab_size = len(token.word_index) + 1  # No. of unique words in X_train

X_tr_tok = token.texts_to_sequences(X_train)  # Transforming train and test !
X_test_tok = token.texts_to_sequences(X_test)

max_length = 3000                   # Max lenth of text in Data , to pad all files upto this length ! Too large Value was Exploding my VRAM !

X_tr_pad = pad_sequences(X_tr_tok  ,maxlen=max_length, padding='post')
X_te_pad = pad_sequences(X_test_tok  ,maxlen=max_length, padding='post')

path = '/content/300d-char.txt'  # Loding Char Embedding to Ram!
embedding_vectors = {}
with open(path, 'r') as f:
    for line in f:
        line_split = line.strip().split(" ")
        vec = np.array(line_split[1:], dtype=float)
        char = line_split[0]
        embedding_vectors[char] = vec

vocab_size = len(token.word_index) + 1  
max_length = 3000

embedding_matrix = np.zeros((vocab_size, 300))   # WE used 300d Glove Vector , so 300d embedding to match the dimensions !
for char, i in token.word_index.items():
	embedding_vector = embedding_vectors.get(char)
	if embedding_vector is not None:
		embedding_matrix[i] = embedding_vector

embedding_matrix.shape

import random  as rn
tf.keras.backend.clear_session()

np.random.seed(0)
rn.seed(0)

input = Input(shape=(max_length,), dtype='int32')

embedded_layer =  Embedding(vocab_size , 300 , weights=[embedding_matrix] , input_length= max_length,  trainable=False)(input)

x = Conv1D(128 , 3, activation='relu'  )(embedded_layer)
y = Conv1D(128 , 4, activation='relu' )(x)

pool1 = MaxPooling1D()(y)

i =  Conv1D(64 , 5, activation='relu')(pool1)
j =  Conv1D(64 , 6, activation='relu' )(i)

pool2 = MaxPooling1D()(j)

flat = Flatten()(pool2)

drop = Dropout(0.5)(flat)
FC = Dense(128, activation='relu')(drop)
FC = Dense(64, activation='relu')(drop)


out = Dense(20, activation='softmax')(FC)

model2 = Model(input, out)

tf.keras.utils.plot_model(model2 , show_shapes = True , to_file= 'm2.png' )

!rm -rf ./logs1/ 
from tensorflow.keras.callbacks import ReduceLROnPlateau
earlystop = EarlyStopping(monitor='val_acc', min_delta=0.001, patience=2)
micro_f1 = m_F1(X_te_pad , y_test)
log_dir="logs1"
tensorboard = tf.keras.callbacks.TensorBoard(log_dir=log_dir,histogram_freq=1, write_graph=True)

lr2 = ReduceLROnPlateau( monitor='val_acc', factor= 0.9 , patience=1 , verbose=1, mode='auto',
    min_delta=0.0001, cooldown=0, min_lr=0 )

cb = [earlystop ,micro_f1 , tensorboard] 

opt = tf.keras.optimizers.RMSprop(learning_rate=0.001)


model2.compile(loss='categorical_crossentropy',  
              optimizer=opt,
              metrics=['acc'] )

model2.fit(X_tr_pad, y_train,
          batch_size= 16,
          epochs= 4,
          validation_data=(X_te_pad, y_test) , callbacks = cb)