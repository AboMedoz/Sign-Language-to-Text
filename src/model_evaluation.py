import os

import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import to_categorical

from image_preprocessing import preprocess_img

BASE_DIR = os.path.dirname(__file__)
ROOT = os.path.dirname(BASE_DIR)
TEST_DATA_PATH = os.path.join(ROOT, 'data', 'test')
MODEL_PATH = os.path.join(ROOT, 'models')

imgs = []
labels = []

for img_str in os.listdir(TEST_DATA_PATH):
    img_path = os.path.join(TEST_DATA_PATH, img_str)
    img = cv2.imread(img_path)
    img = preprocess_img(img, -1)

    label = img_str.split('_')[0]

    imgs.append(img)
    labels.append(label)

class_names = sorted(set(labels))
class_names_to_index = {name: idx for idx, name in enumerate(class_names)}
int_labels = [class_names_to_index[label] for label in labels]

x = np.array(imgs)
y = to_categorical(int_labels)

model = load_model(os.path.join(MODEL_PATH, 'sign_language_model.keras'))

_, accuracy = model.evaluate(x, y)
print(f"Accuracy: {accuracy * 100:.2f}")


