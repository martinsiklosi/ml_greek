from tensorflow import keras
import numpy as np
from settings import *

# load model
model = keras.models.load_model('model1')

def grid_to_picture_data(grid):
    picture_data = np.array(grid, dtype=np.uint8)
    picture_data = np.clip(picture_data, 0, 1)
    picture_data = 1 - picture_data[:,:,:1]
    picture_data = np.expand_dims(picture_data, axis=0)
    return picture_data

def predict(grid):
    picture_data = grid_to_picture_data(grid)
    prediction = model.predict(picture_data)
    index = np.argmax(prediction, axis=-1)[0]
    prediction_name = LABEL_STRINGS[index]
    prediction_name = prediction_name.strip("_")
    probability = prediction[0, index]
    return prediction_name, probability
