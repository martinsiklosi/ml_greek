import numpy as np
import tensorflow as tf
from tensorflow import keras
from settings import *
import json

with open("data.json", "r") as file:
    images, image_labels = json.load(file)
    
images = np.array(images, dtype=np.uint8)
image_labels = np.array(image_labels, dtype=np.uint8)

# setup model
model = keras.Sequential()

for i in range(6):
    model.add(
        keras.layers.Conv2D(
            8, 
            3, 
            activation=tf.nn.relu, 
            input_shape=(None, None, 1)
        )
    )
    
model.add(keras.layers.Reshape((45, 45, 1)))
model.add(keras.layers.Flatten())

for _ in range(12):
    model.add(
        keras.layers.Dense(
            64, 
            activation=tf.nn.relu
        )
    )
    
model.add(
    keras.layers.Dense(
        48, 
        activation=tf.nn.softmax
    )
)

model.compile(
    optimizer=tf.optimizers.Adam(),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# train model
model.fit(images, image_labels, epochs=5)

# save model
model.save('model')
