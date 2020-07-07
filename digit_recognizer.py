#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 31 09:10:39 2019

@author: Parika
"""

import pandas as pd
from keras.utils.np_utils import to_categorical
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense 
from keras.layers import Dropout
from keras.layers import Flatten 

from keras.layers.convolutional import Convolution2D
from keras.layers.convolutional import MaxPooling2D

data=pd.read_csv("/Users/Parika/Desktop/Python/MAchine Learning/digit-recognizer/digit_recognizer_train.csv")
data.head(20)
#just taking a peek at the data so that it can be preprocessed 
# not performing Label encoding 
Y=dataset['label'] #will return a series but we want a list hence type conversion
y=list(Y)
X=dataset.drop('label',axis=1,inplace=True)
dataset=data.sample(frac=0.5, random_state=0)
#taking half the data available
#test_dataset=pd.read_csv("/Users/Parika/Desktop/Python/MAchine Learning/digit-recognizer/digit_recognizer_test.csv")

 # as we want our X_train we will drop the labels column from the dataset
X = (dataset.iloc[:,:-1])/255
X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)


#now reshape the data
X_train=X_train.values.reshape(-1, 28, 28, 1)
X_test=X_test.values.reshape(-1, 28, 28, 1)

Y_train= to_categorical(Y_train, num_classes=10)
Y_test= to_categorical(Y_test,num_classes=10)
#one hot encoding to avoid confusing the model


#test=test_dataset.values.reshape(-1,28,28,1)


def baseline_model():
    model=Sequential()
    #model.add(Convolution2D(32, 5, 5, border_mode='valid', input_shape=(1, 28, 28),activation='relu'))
    model.add(Convolution2D(32, (5, 5), activation='relu', input_shape=(1,32,32), data_format='channels_first'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))
    # Compile model
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model
    
#build the model
model=baseline_model
#fit the modelmodel.fit(X_train, y_train, validation_data=(X_test, y_test), nb_epoch=20, batch_size=200,verbose=2)
model.fit(X_train,Y_train,validation_data=(X_test, Y_test),nb_epochs=12,batch_size=200,verbose=2)
#final evaluation of the model
scores=model.evaluate(X_test, Y_test, verbose=0)
print("Baseline Error: %.2f%%"% (100-scores[1]*100))

    








