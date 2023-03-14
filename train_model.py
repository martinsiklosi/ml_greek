import numpy as np
import tensorflow as tf
from tensorflow import keras
from settings import *

# import data
from get_data import images, image_labels

# settings
CONV_LAYERS = 8
CONV_FILTERS = 8
CONV_KERNEL = 4
CONV_STRIDE = 1

DENSE_LAYERS = 2
DENSE_SIZE = 196

EPOCHS = 20

# setup model
model = keras.Sequential()

model.add(keras.Input(shape=(ROWS, COLS, 1)))

for i in range(CONV_LAYERS):
    model.add(keras.layers.Conv2D(
        CONV_FILTERS, CONV_KERNEL, padding="same", strides=CONV_STRIDE, 
        activation=tf.nn.relu, input_shape=(None, None, 1)))
        
model.add(keras.layers.Flatten())

for _ in range(DENSE_LAYERS):
    model.add(keras.layers.Dense(
            DENSE_SIZE, activation=tf.nn.relu))
    
model.add(keras.layers.Dense(
        48, activation=tf.nn.softmax))

model.compile(
    optimizer=tf.optimizers.Adam(),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy'])

# train model
model.fit(images, image_labels, epochs=EPOCHS)

# save model
model.save('model1')
