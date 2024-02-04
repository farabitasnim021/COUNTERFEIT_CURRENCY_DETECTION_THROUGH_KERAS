import random
import ssl
from encodings.utf_8 import decode

import numpy as np
from numpy import genfromtxt
import tensorflow as tf
import keras
from keras import models
from keras.models import Sequential
from keras.layers import Dense

data = genfromtxt('/Users/farabitasnimahmed/DATA/bank_note_data.txt',delimiter=',')
#separate the labels
labels = data[:,4]
features = data[:,0:4]
X = features
y = labels

#split the dataset into training and test data
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.33, random_state=42)
#scaling the data
from sklearn.preprocessing import MinMaxScaler
scaler_object = MinMaxScaler()
scaler_object.fit(X_train)
scaled_X_train = scaler_object.transform(X_train)
scaled_X_test = scaler_object.transform(X_test)
model = Sequential()

model.add(Dense(4, input_dim=4,activation='relu'))
model.add(Dense(8,activation='relu'))
model.add(Dense(1,activation='sigmoid'))
model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])

model.fit(scaled_X_train,y_train,epochs=50,verbose=2)
# model.predict(scaled_X_test)

# evaluation of the model
from sklearn.metrics import confusion_matrix,classification_report
predictions = model.predict(scaled_X_test).round()
confusion_matrix(y_test,predictions)
print(classification_report(y_test,predictions))

#Save the model
model.save('mysupermodel.h5')

#Load the model
# from keras.models import load_model
# new_model = load_model('mysupermodel.h5')
# new_model.predict_classes(scaled_X_test)