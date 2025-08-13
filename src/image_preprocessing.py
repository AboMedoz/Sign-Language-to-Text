import cv2
import numpy as np


def preprocess_img(img, axis, color_mode='BGR'):
    # color mode: 'RGB' or 'BGR' or 'None'
    # Mediapipe & MobilenetV2 expects RGB
    if img is None or img.size == 0:
        pass

    if color_mode:
        if color_mode == 'BGR':
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        if color_mode == 'RGB':
            img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        img = cv2.GaussianBlur(img, (5, 5), 2)
        thresh = cv2.adaptiveThreshold(
            img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2
        )
        _, img = cv2.threshold(thresh, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_BINARY_INV)
    img = cv2.resize(img, (96, 96))
    img = img.astype(np.float32) / 255.0
    img = np.expand_dims(img, axis)
    return img
