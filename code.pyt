import os
import pandas as pd
import librosa
import glob 
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.optimizers import Adam
from keras.utils import np_utils
from sklearn import metrics


train = pd.read_csv('../data/train.csv')

#now we will construct functions that are going to extract files(since there are files inside of file) and features from files that can be used in algo. There are a number of features that we coud extract, we are going to with help of librosa extract mfcc feature from every file
#os.path module for manipulating files and directories

def parser(row):
   # function to load files and extract features
   file_name = os.path.join(os.path.abspath(data_dir), 'Train', str(row.ID) + '.wav')

   # handle exception to check if there isn't a file which is corrupted
   try:
      # here kaiser_fast is a technique used for faster extraction
      X, sample_rate = librosa.load(file_name, res_type='kaiser_fast') 
      # we extract mfcc feature from data
      mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T,axis=0) 
   except Exception as e:
      print("Error encountered while parsing file: ", file)
      return None, None
 
   feature = mfccs
   label = row.Class
 
   return [feature, label]

temp = train.apply(parser, axis=1)
temp.columns = ['feature', 'label']


#now let us transform the data in a usable format (Deep learnin can only handle numerical variables, hence we use one hot encoding (Labelencoder))


from sklearn.preprocessing import LabelEncoder

X = np.array(temp.feature.tolist())
y = np.array(temp.label.tolist())

lb = LabelEncoder()

y = np_utils.to_categorical(lb.fit_transform(y))



#now we are ready to plug it in the model, we are going to build sequential(linear) neural network with mulitple layers adding them gradually (with add))
#variant of gradient descent is adam, hidden-layer activation function is simply max(0,x), dropout-to avoid overfitting is set at 50%, i.e. every second neuron is dropped

num_labels = y.shape[1]
filter_size = 2

# build model
model = Sequential()

model.add(Dense(256, input_shape=(40,)))
model.add(Activation('relu'))
model.add(Dropout(0.5))

model.add(Dense(256))
model.add(Activation('relu'))
model.add(Dropout(0.5))

model.add(Dense(num_labels))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer='adam')


#training our model, specific epoch and batch size depend on size of our data to avoid overfitting an make use of iterations
model.fit(X, y, batch_size=32, epochs=5, validation_data=(val_x, val_y))


#after one forward and one backward pass of all training examples, i.e. one epoch we get

Train on 5435 samples, validate on 1359 samples
Epoch 1/10
5435/5435 [==============================] - 2s - loss: 12.0145 - acc: 0.1799 - val_loss: 8.3553 - val_acc: 0.2958
Epoch 2/10
5435/5435 [==============================] - 0s - loss: 7.6847 - acc: 0.2925 - val_loss: 2.1265 - val_acc: 0.5026
Epoch 3/10
5435/5435 [==============================] - 0s - loss: 2.5338 - acc: 0.3553 - val_loss: 1.7296 - val_acc: 0.5033
Epoch 4/10
5435/5435 [==============================] - 0s - loss: 1.8101 - acc: 0.4039 - val_loss: 1.4127 - val_acc: 0.6144
Epoch 5/10
5435/5435 [==============================] - 0s - loss: 1.5522 - acc: 0.4822 - val_loss: 1.2489 - val_acc: 0.6637





