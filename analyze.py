from tensorflow import keras
import numpy as np
from settings import *

# load model
model = keras.models.load_model('model')

def grid_to_picture_data(grid):
    picture_data = np.zeros((ROWS, COLS), dtype=int)
    for i in range(ROWS):
        for j in range(COLS):
            pixel = grid[i][j]
            picture_data[i, j] = 255 - pixel[0]
    picture_data = np.expand_dims(picture_data, axis=0)
    return picture_data
            
def predict(grid):
    picture_data = grid_to_picture_data(grid)
    prediction = model.predict(picture_data)
    index = np.argmax(prediction, axis=-1)[0]
    prediction_name = LABEL_STRINGS[index]
    probability = prediction[0, index]
    return prediction_name, probability
