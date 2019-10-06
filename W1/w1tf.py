import numpy as np 
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt 
import h5py
import scipy
from PIL import Image
from scipy import ndimage
from lr_utils import load_dataset

train_set_x_orig,train_set_y,test_set_x_orig,test_set_y, classes = load_dataset()

#flatten the matrix
train_set_x_flatten = train_set_x_orig.reshape(train_set_x_orig.shape[0],-1).T 
test_set_x_orig_flatten = test_set_x_orig.reshape(test_set_x_orig.shape[0],-1).T 

#normalize the input
train_set_x = train_set_x_flatten/255
test_set_x = test_set_x_orig_flatten/255

#model with flatten input

model = keras.Sequential([
    keras.layers.Dense(12288, activation='relu'),
    keras.layers.Dense(1, activation='sigmoid')
])
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])


model.fit(train_set_x.T,train_set_y.T, epochs = 50)
test_loss, test_acc = model.evaluate(test_set_x.T,  test_set_y.T, verbose=2)

print('\nTest accuracy:', test_acc)