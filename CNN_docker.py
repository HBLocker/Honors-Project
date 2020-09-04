from keras.callbacks import History 
history = History()
from keras.models import Sequential
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Embedding, LSTM, Dropout, Dense
from keras.utils import to_categorical
from keras.callbacks import ModelCheckpoint, TensorBoard
from keras.callbacks import EarlyStopping
from keras.models import Model
from keras.optimizers import RMSprop
from keras.preprocessing.text import Tokenizer
from keras.layers import Dense, Dropout, Activation,Bidirectional
from keras.layers import Embedding
from keras.layers import Conv1D, GlobalMaxPooling1D
from keras.preprocessing.text import Tokenizer
from keras.preprocessing import sequence
from sklearn.preprocessing import LabelEncoder

from keras import backend
from keras.layers import Dense
from keras import Sequential
from keras.utils import  plot_model
import numpy as np
import tqdm
import time
import sklearn.metrics as metrics
from matplotlib.pyplot import *
from matplotlib import pyplot as plt

"""   !pip install keras-metrics this shoudl run within a dokcer container now """ 

from keras.callbacks import History 
history = History()
from keras.models import Sequential
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Embedding, LSTM, Dropout, Dense
from keras.utils import to_categorical
from keras.callbacks import ModelCheckpoint, TensorBoard
from keras.callbacks import EarlyStopping
from keras.models import Model
from keras.optimizers import RMSprop
from keras.preprocessing.text import Tokenizer
from keras.layers import Dense, Dropout, Activation,Bidirectional
from keras.layers import Embedding
from keras.layers import Conv1D, GlobalMaxPooling1D
from keras.preprocessing.text import Tokenizer
from keras.preprocessing import sequence
from sklearn.preprocessing import LabelEncoder
from keras import backend
from keras.layers import Dense
from keras import Sequential
from keras.utils import  plot_model
import numpy as np
import tqdm
import time
import sklearn.metrics as metrics
from matplotlib.pyplot import *
from matplotlib import pyplot as plt
#import keras_metrics
import keras_metrics
from tensorflow.keras.optimizers import SGD
import matplotlib.pyplot as plt
from keras.utils.vis_utils import plot_model
from keras.models import model_from_json
from matplotlib.pyplot import *
from matplotlib import pyplot as plt
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Dropout
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
from keras.utils import to_categorical


SEQUENCE_LENGTH = 100
EMBEDDING_SIZE = 300 #dimensions for embedding file 
TEST_SIZE = 0.5 #size of data_set
FILTERS = 70
BATCH_SIZE = 200

EPOCHS =100  # number of epochs
# give lables a numeric value
label2int = {"ham": 0, "spam": 1}
int2label = {0: "ham", 1: "spam"}

max_features = 5000
maxlen = 100
embedding_dims = 50
filters = 250
kernel_size = 7
hidden_dims = 250
pool_size = 15
lstm_output_size = 1028

def Load_data():
  text, labels = [], []
  with open("datatest2",encoding='utf-8-sig') as f: #loads dataset python 3, encodinf stig stops dataset breaking 
    for line in f:
      split = line.split()
      labels.append(split[0].strip())
      text.append(''.join(split[1:]).strip())
  return text,labels

X, y = Load_data() # loads the x and Y data

tokenizer = Tokenizer() #converts the utf-8 into tokinized characters 
tokenizer.fit_on_texts(X)
# tokinize into ints 
X = tokenizer.texts_to_sequences(X)

X = np.array(X)
y = np.array(y)
X = pad_sequences(X, maxlen=SEQUENCE_LENGTH)

y = [ label2int[label] for label in y ] #loads lables 
y = to_categorical(y)

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, random_state=43) #splits the data into test and train sets

"""Loads glve embedding file"""

def get_embedding_vectors(tokenizer, dim=EMBEDDING_SIZE): #dim is dimensions EMBEDDINGSIZE is size for the netwokr and the file
    embedding_index = {}
    with open("glove.6B.300d.txt", 'r',encoding='utf8',errors = 'ignore') as f: #ignore erros is due to sometimes when reading it fails to convert values 
        for line in tqdm.tqdm(f, "Reading GloVe"):
            values = line.split()
            word = values[0]
            vectors = np.asarray(values[1:], dtype='float32') 
            embedding_index[word] = vectors

    word_index = tokenizer.word_index
    embedding_matrix = np.zeros((len(word_index)+1, dim))
    for word, i in word_index.items():
        embedding_vector = embedding_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector
            
    return embedding_matrix
    """ end of the method"""
    
    """ cnn model imp""" 

    # load embedding as a dict
def load_embedding(filename):
	# load embedding into memory, skip first line
	file = open(filename,'r',errors='Ignore')
	lines = file.readlines()
	file.close()
	embedding = dict()
	for line in lines:
		parts = line.split()
		embedding[parts[0]] = asarray(parts[1:], dtype='float32')
	return embedding

raw_embedding = get_embedding_vectors(tokenizer)

#custom embedidng layer for Glove 
embedding_layer = Embedding(BATCH_SIZE,[embedding_dims])

def cnn_boi(tokenizer): #loads the function and creates a BI_rnn network 
  embedding_matrix = get_embedding_vectors(tokenizer)
  model = Sequential()
  model.add(Embedding(max_features,
                    embedding_dims,
                    input_length=maxlen))
  model.add(Dropout(0.2))

  model.add(Conv1D(filters,
                 kernel_size,
                 padding='valid',
                 activation='relu',
                 strides=1))
  model.add(GlobalMaxPooling1D())

  model.add(Dense(hidden_dims))
  model.add(Dropout(0.2))
  model.add(Activation('relu'))

#thinns model to sigmoid large data ---> small Box (squashed) ^_^  
  model.add(Dense(2))
  model.add(Activation('relu'))
  plot_model(model, to_file='1d_cnn.png', show_shapes=True, show_layer_names=True)
  model.compile(loss='binary_crossentropy',
                optimizer='adam',
                metrics=['accuracy'])


  return model
cnn_model = cnn_boi(tokenizer)
import matplotlib.pyplot as plt
from keras.callbacks import History 
model_checkpoint = ModelCheckpoint("phish{val_loss:.2f}", save_best_only=True,verbose=1) #creates model checkpoints was removed for someof te netwkrs as overalp happened 
# 
history = History()
tensorboard = TensorBoard(f"phish{time.time()}")
# print our data shapes
print("X_train.shape:", X_train.shape)
print("X_test.shape:", X_test.shape)
print("y_train.shape:", y_train.shape)
print("y_test.shape:", y_test.shape)
# traiaining in progress 
cnn_model.fit(X_train, y_train, validation_data=(X_test, y_test),
          batch_size=BATCH_SIZE, epochs=EPOCHS,
          callbacks=[tensorboard, model_checkpoint,history],
          verbose=1)
def cnn_mod(text):
    sequence = tokenizer.texts_to_sequences([text])
    # pad the sequence
    sequence = pad_sequences(sequence, maxlen=SEQUENCE_LENGTH)
    # get the prediction
    prediction = cnn_model.predict(sequence)[0]
    prob =  cnn_model.predict_proba(sequence)/100*10
    print (prob)
    return int2label[np.argmax(prediction)] 
"""need a socker now created to here ""
""socker from docker container to here to recive the email"" 
""work in progress""


