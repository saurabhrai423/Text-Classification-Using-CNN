
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

data = pandas.read_csv('clean2.csv')

y = data['class']

from sklearn import preprocessing
le = preprocessing.LabelEncoder()

y = le.fit_transform(y)
y = tf.keras.utils.to_categorical(y)

y.shape

data['final_text'] = data[['preprocessed_email' , 'preprocessed_subject' , 'preprocessed_text']].astype(str).apply(' '.join , axis = 1)

X = data['final_text']

import numpy as np
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y , test_size=0.25, stratify = y ,random_state=42)

token = tf.keras.preprocessing.text.Tokenizer(filters='!"#$%&()*+,-./:;<=>?@[\\]^`{|}~\t\n' , oov_token = True )   # Tokenizing text which can handle unseen data on test

token.fit_on_texts(X_train)
vocab_size = len(token.word_index) + 1  # No. of unique words in X_train

X_tr_tok = token.texts_to_sequences(X_train)  # Transforming train and test !
X_test_tok = token.texts_to_sequences(X_test)

max_length = 3000                   # Max lenth of text in Data , to pad all files upto this length ! Too large Value was Exploding my VRAM !

X_tr_pad = pad_sequences(X_tr_tok  ,maxlen=max_length, padding='post')
X_te_pad = pad_sequences(X_test_tok  ,maxlen=max_length, padding='post')

embeddings_index = dict()     # LOADING GLOVE TO MEMORY !
f = open('glove.6B.100d.txt')
for line in f:
	values = line.split()
	word = values[0]
	coefs = np.asarray(values[1:], dtype='float32')
	embeddings_index[word] = coefs
f.close()
print('Loaded %s word vectors.' % len(embeddings_index))

#Creating Embedding Layer
embedding_matrix = np.zeros((vocab_size, 100))   # WE used 100d Glove Vector , so 100d embedding to match the dimensions !
for word, i in token.word_index.items():
	embedding_vector = embeddings_index.get(word)
	if embedding_vector is not None:
		embedding_matrix[i] = embedding_vector