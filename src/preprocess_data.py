import os

import cv2
import numpy as np
from sklearn.utils import shuffle
from tensorflow.keras.utils import to_categorical

from image_preprocessing import preprocess_img

# https://www.kaggle.com/datasets/grassknoted/asl-alphabet
BASE_DIR = os.path.dirname(__file__)
ROOT = os.path.dirname(BASE_DIR)
DATA_PATH = os.path.join(ROOT, 'data', 'raw')
PROCESSED_DATA = os.path.join(ROOT, 'data', 'preprocessed')

imgs = []
labels = []

for categories in os.listdir(DATA_PATH):
    category = os.path.join(DATA_PATH, categories)
    for img_paths in os.listdir(category):
        img_path = os.path.join(category, img_paths)
        img = cv2.imread(img_path)
        img = preprocess_img(img, -1)
        imgs.append(img)
        labels.append(categories)
    print(f"Finished Folder {category}")  # Debug or more like a sanity check.

class_names = sorted(set(labels))
class_to_index = {name: idx for idx, name in enumerate(class_names)}
int_labels = [class_to_index[label] for label in labels]

x = np.array(imgs, dtype=np.float32)
y = to_categorical(np.array(int_labels), num_classes=len(class_names))
x, y = shuffle(x, y, random_state=42)

np.savez_compressed(
    os.path.join(PROCESSED_DATA, 'preprocessed.npz'),
    x=x,
    y=y,
    class_names=class_names
)
