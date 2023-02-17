import numpy as np
import tensorflow as tf
from tensorflow import keras
from settings import *

# # get data
# # kika på https://www.tensorflow.org/tutorials/load_data/images
# (train_images, train_labels), (test_images, test_labels) = \
#     keras.datasets.mnist.load_data()

# def posterise(images):
#     images[images >= 200] == 255
#     images[images < 200] == 0
#     return images

# train_images = np.concatenate((train_images, test_images), axis=0)
# train_labels = np.concatenate((train_labels, test_labels), axis=0)
# train_images = posterise(train_images)

# setup model
model = keras.Sequential()
for i in range(6):
    model.add(keras.layers.Conv2D(8, 3, activation=tf.nn.relu, input_shape=(COLS-2*i, ROWS-2*i, 1)))
model.add(keras.layers.Flatten())
for _ in range(12):
    model.add(keras.layers.Dense(64, activation=tf.nn.relu))
model.add(keras.layers.Dense(10, activation=tf.nn.softmax))

model.compile(
    optimizer=tf.optimizers.Adam(),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# train model
model.fit(train_images, train_labels, epochs=5)

# save model
model.save('model')
