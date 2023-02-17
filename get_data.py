import json
import cv2
from pathlib import Path
import numpy as np
from settings import *


'''ALLA BILDER ÄR OLIKA STORA :/'''

BASE_FOLDER = "C:\\proj\\ml_data\\greek_alphabet\\NORM\\"
BASE_FOLDER2 = "C:\\proj\\ml_data\\greek_alphabet\\SUFF\\"


images = []
labels = []

def invert(image):
    for i in range(ROWS):
        for j in range(COLS):
            image[i, j] = 255 - image[i, j]
    return image

print("reading images...")
for i, path in enumerate(Path(BASE_FOLDER).rglob("*.bmp")):
    if i >= MAX_TRAINING_FILES:
        break
    image = cv2.imread(f"{path}")[:,:,0]
    image = invert(image)
    images.append(image)
    labels.append(path.parent.name)
print(f"read {len(images)} files")


# convert to native lists
for i, element in enumerate(images):
    images[i] = element[1].tolist()


print("writing to file...")
data = (images, labels)
with open("data.json", "w") as outfile:
    json.dump(data, outfile)
print("done")
