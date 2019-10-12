import time
import numpy as np
import h5py
import matplotlib.pyplot as plt
import scipy
from PIL import Image
from scipy import ndimage
import skimage
from dnn_app_utils_v2 import *
from tensorflow import keras

train_x_orig, train_y, test_x_orig, test_y, classes = load_data()
train_x_flatten = train_x_orig.reshape(train_x_orig.shape[0], -1).T   # The "-1" makes reshape flatten the remaining dimensions
test_x_flatten = test_x_orig.reshape(test_x_orig.shape[0], -1).T

# Standardize data to have feature values between 0 and 1.
train_x = train_x_flatten/255.
test_x = test_x_flatten/255.

model = keras.Sequential([
    keras.layers.Dense(12288,kernel_initialzer = 'random_uniform',bias_initializer = 'zeros',activation = 'relu'),
    keras.layers.Dense(1,kernel_initialzer = 'random_uniform',bias_initializer = 'zeros',activation = 'sigmoid')
])

model.compile(optimizer = 'adam',loss = 'binary_crossentropy',metrics=['accuracy'])

model.fit(train_x.T,train_y.T,epochs=2500)

loss,acc = model.evaluate(test_x,test_y,verbose=2)
print('\nFinal loss :',loss)
print('\nFinal accuracy :',acc)