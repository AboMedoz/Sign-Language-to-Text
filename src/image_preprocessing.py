import cv2
import numpy as np


def preprocess_img(img, axis, color_mode='BGR'):
    if img is None or img.size == 0:
        pass

    if color_mode == 'BGR':
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    img = cv2.resize(img, (128, 128))
    img = img / 255.0
    img = np.expand_dims(img, axis)
    return img
