import numpy as np
import tensorflow as tf
from tensorflow import keras
from settings import *

# import data
from get_data import images, image_labels

# settings
CONV_LAYERS = 0
CONV_FILTERS = 8
CONV_KERNEL = 3

DENSE1_LAYERS = 2
DENSE1_SIZE = 512

DENSE2_LAYERS = 12
DENSE2_SIZE = 86

EPOCHS = 24

# setup model
model = keras.Sequential()
model.add(keras.Input(shape=(ROWS, COLS, 1)))
for i in range(CONV_LAYERS):
    model.add(keras.layers.Conv2D(
        CONV_FILTERS, CONV_KERNEL, activation=tf.nn.relu, input_shape=(None, None, 1)))
model.add(keras.layers.Flatten())
for _ in range(DENSE1_LAYERS):
    model.add(keras.layers.Dense(
            DENSE1_SIZE, activation=tf.nn.relu))
for _ in range(DENSE2_LAYERS):
    model.add(keras.layers.Dense(
            DENSE2_SIZE, activation=tf.nn.relu))  
model.add(keras.layers.Dense(
        48, activation=tf.nn.softmax))
model.compile(
    optimizer=tf.optimizers.Adam(),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy'])

# train model
model.fit(images, image_labels, epochs=EPOCHS)

# save model
model.save('model')
