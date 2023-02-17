from tensorflow import keras
import numpy as np
from settings import *

# load model
model = keras.models.load_model('model')

def grid_to_picture_data(grid):
    picture_data = np.zeros((GRID_SIZE, GRID_SIZE), dtype=int)
    for i in range(GRID_SIZE):
        for j in range(GRID_SIZE):
            pixel = grid[i][j]
            picture_data[i, j] = 255 - pixel[0]
    # picture_data = blur_picture_data(picture_data)
    picture_data = np.expand_dims(picture_data, axis=0)
    return picture_data

# def blur_picture_data(picture_data):
#     blurred = np.zeros((GRID_SIZE, GRID_SIZE), dtype=int)
#     for i in range(1, GRID_SIZE-1):
#         for j in range(1, GRID_SIZE-1):
#             if picture_data[i, j] == 255:
#                 blurred[i, j] = 255
#             else:
#                 touching_sum = \
#                     picture_data[i-1, j] +\
#                     picture_data[i+1, j] +\
#                     picture_data[i, j-1] +\
#                     picture_data[i, j+1]
#                 new_pixel = touching_sum // 16
#                 blurred[i, j] = new_pixel
#     return blurred
            
def predict(grid):
    picture_data = grid_to_picture_data(grid)
    prediction = model.predict(picture_data)
    prediction_index = np.argmax(prediction, axis=-1)[0]
    probability = prediction[0, prediction_index]
    return prediction_index, probability
