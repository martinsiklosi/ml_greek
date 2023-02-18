import json
import cv2
from pathlib import Path
import numpy as np
from settings import *


BASE_FOLDERS = (
    "C:\\proj\\ml_data\\greek_alphabet\\SUFF\\", 
    "C:\\proj\\ml_data\\greek_alphabet\\NORM\\"
)

images = []
image_labels = []

def invert(image):
    for i in range(len(image)):
        for j in range(len(image[0])):
            image[i, j] = 255 - image[i, j]
    return image

print("reading images...")
for base_folder in BASE_FOLDERS:
    for i, path in enumerate(Path(base_folder).rglob("*.bmp")):
        image = cv2.imread(f"{path}")[:,:,0]
        image = invert(image)
        images.append(image)
        image_labels.append(LABEL_CONVERSIONS[path.parent.name])
print(f"read {len(images)} files")


# convert to native lists
for i, element in enumerate(images):
    images[i] = element[1].tolist()


print("writing to file...")
data = (images, image_labels)
with open("data.json", "w") as outfile:
    json.dump(data, outfile)
print("done")
