import numpy as np
import pydot
import tensorflow as tf
from tensorflow import keras
import h5py
import pandas as pd
import os
import matplotlib.pyplot as plt


def load_dataset(path_to_train, path_to_test):
    train_dataset = h5py.File(path_to_train)
    train_x = np.array(train_dataset['train_set_x'][:])
    train_y = np.array(train_dataset['train_set_y'][:])

    test_dataset = h5py.File(path_to_test)
    test_x = np.array(test_dataset['test_set_x'][:])
    test_y = np.array(test_dataset['test_set_y'][:])
    classes = np.array(test_dataset["list_classes"][:])  # the list of classes

    # y reshaped
    train_y = train_y.reshape((1, train_x.shape[0]))
    test_y = test_y.reshape((1, test_y.shape[0]))

    return train_x, train_y, test_x, test_y, classes

def plot_learning_curve(history):
    pd.DataFrame(history.history).plot(figsize=(8,5))
    plt.grid(True)
    plt.gca().set_ylim(0,1)
    plt.savefig("happyhouse_lc.png")

## Load and normalize data
X_train_orig, Y_train_orig, X_test_orig, Y_test_orig, classes = load_dataset("train_happy.h5", "test_happy.h5")

# Normalize image vectors
X_train = X_train_orig / 255.
X_test = X_test_orig / 255.

# Reshape
Y_train = Y_train_orig.T
Y_test = Y_test_orig.T

print("number of training examples = " + str(X_train.shape[0]))
print("number of test examples = " + str(X_test.shape[0]))
print("X_train shape: " + str(X_train.shape))
print("Y_train shape: " + str(Y_train.shape))
print("X_test shape: " + str(X_test.shape))
print("Y_test shape: " + str(Y_test.shape))
print("Number of classes" + str(classes.shape))

model = keras.models.Sequential()
model.add(keras.layers.Conv2D(32, (7, 7), padding="same", activation="relu", input_shape=(64, 64, 3), use_bias=True,
                              kernel_initializer=keras.initializers.glorot_normal, bias_initializer="zeros"))
model.add(keras.layers.BatchNormalization(axis=3))
model.add(keras.layers.MaxPooling2D((2,2),padding="same"))
model.add(keras.layers.Flatten())
model.add(keras.layers.Dense(1,activation='sigmoid'))

model.compile(loss='binary_crossentropy',
             optimizer = "Adam",metrics=["accuracy"])

## Create call back function to store weight
checkpoint_path = "training_1/cp.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)
cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                 save_weights_only=True,
                                                 verbose=1)

history = model.fit(X_train,Y_train,epochs=40,batch_size=50,validation_data=(X_test,Y_test),callbacks=[cp_callback])

plot_learning_curve(history)