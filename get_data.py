import cv2
from pathlib import Path
import numpy as np
from settings import *
from random import randint, shuffle


def resize(image):
    diff_cols = np.size(image, axis=1) - COLS
    diff_rows = np.size(image, axis=0) - ROWS
    if diff_cols > 0:
        a = diff_cols // 2
        b = diff_cols - a
        image = image[:,a:-b,:]
    if diff_rows > 0:
        a = diff_rows // 2
        b = diff_rows - a
        image = image[a:-b,:,:]
    if diff_cols < 0:
        a = randint(0, -diff_cols)
        b = -diff_cols - a
        image = np.pad(image, ((0, 0), (a, b), (0, 0)))
    if diff_rows < 0:
        a = randint(0, -diff_rows)
        b = -diff_rows - a
        image = np.pad(image, ((a, b), (0, 0), (0, 0)))
    return image


paths = []
for base_folder in BASE_FOLDERS:
    paths.extend(Path(base_folder).rglob("*.bmp"))

print()

images = []
image_labels = []
for i, path in enumerate(paths):
    if i % 143 == 0: # just to make it look nice :)
        print(f"\r{len(images)} images collected", end="")
    image = cv2.imread(f"{path}")[:,:,:1]
    image = np.clip(image, 0, 1)
    image = 1 - image
    image = resize(image)
    images.append(image)
    image_labels.append(LABEL_CONVERSIONS[path.parent.name])

temp = list(zip(images, image_labels))
shuffle(temp)
images, image_labels = zip(*temp)

images = np.array(images, dtype=np.uint8)
image_labels = np.array(image_labels, dtype=np.uint8)

print("\n")