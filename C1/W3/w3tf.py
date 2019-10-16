from planar_utils import plot_decision_boundary, sigmoid, load_planar_dataset, load_extra_datasets
import numpy as np
import tensorflow as tf 
import matplotlib.pyplot as plt 
from tensorflow import keras

X,Y = load_planar_dataset()

def layer_size(X,Y): 
    n_x = X.shape[0]
    n_y = Y.shape[0]
    return n_x,n_y

###test code 
n_x, n_y = layer_size(X,Y)

print("The size of input layer is :"+ str(n_x))
print("The size of output layer is :"+ str(n_y))
### test codes ends

model = keras.Sequential([
    keras.layers.Dense(100,kernel_initializer = 'random_uniform',bias_initializer = 'zeros',activation='tanh'),
    keras.layers.Dense(1,kernel_initializer = 'random_uniform',bias_initializer = 'zeros',activation='sigmoid')
])

model.compile(optimizer='adam',
            loss= 'binary_crossentropy',
            metrics=['accuracy']
)

model.fit(X.T,Y.T,epochs=10000)

loss,acc = model.evaluate(X.T,Y.T,verbose=2)
print('\nFinal loss :',loss)
print('\nFinal accuracy :',acc)





