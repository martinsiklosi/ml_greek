import cv2
from pathlib import Path
import numpy as np
from settings import *
from random import randint, shuffle


def image_permutations(image):
    perms = (
        image,
        image[3:-3,3:-3],
        image[4:-1,2:-2],
        image[5:-5,6:-5],
        cv2.resize(image, (ROWS-7, COLS-7)),
        cv2.resize(image, (ROWS-2, COLS-5)),
        cv2.resize(image, (ROWS-1, COLS+4)),
        cv2.resize(image, (ROWS+4, COLS+2))
    )
    return perms

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
    image = cv2.imread(f"{path}")
    image = 255 - image # invert
    # image = np.clip(image, 0, 1) # rescale
    perms = image_permutations(image)
    for perm in perms:
        perm = perm[:,:,:1]
        perm[perm < 127] = 0
        perm[perm >= 127] = 1
        perm = resize(perm)
        images.append(perm)
    image_labels.extend(len(perms)*[LABEL_CONVERSIONS[path.parent.name]])

temp = list(zip(images, image_labels))
shuffle(temp)
images, image_labels = zip(*temp)

images = np.array(images, dtype=np.uint8)
image_labels = np.array(image_labels, dtype=np.uint8)

print("\n")