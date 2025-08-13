import json
import os
from collections import deque, Counter

import cv2
import mediapipe as mp
import numpy as np
from tensorflow.keras.models import load_model

from image_preprocessing import preprocess_img

BASE_DIR = os.path.dirname(__file__)
ROOT = os.path.dirname(BASE_DIR)
MODEL_PATH = os.path.join(ROOT, 'models')

model = load_model(os.path.join(MODEL_PATH, 'mobilenetv2_sign_language_model.h5'))
with open(os.path.join(MODEL_PATH, 'class_names.json'), 'r') as f:
    class_names = json.load(f)

cap = cv2.VideoCapture(0)
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

QUEUE_SIZE = 15
prediction_queue = deque(maxlen=QUEUE_SIZE)
sentence = ''
last_letter = ''
cooldown_counter = 0
cooldown_threshold = 30

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    h, w, _ = frame.shape
    bg_sub = cv2.createBackgroundSubtractorMOG2()
    fg_mask = bg_sub.apply(frame)
    frame = cv2.bitwise_and(frame, frame, mask=fg_mask)

    results = hands.process(rgb)

    prediction = ''
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            x_min, y_min = w, h
            x_max = y_max = 0
            for lm in hand_landmarks.landmark:
                x, y = int(lm.x * w), int(lm.y * h)
                x_min = min(x_min, x)
                y_min = min(y_min, y)
                x_max = max(x_max, x)
                y_max = max(y_max, y)

            padding = 20
            x_min = max(x_min - padding, 0)
            y_min = max(y_min - padding, 0)
            x_max = min(x_max + padding, w)
            y_max = min(y_max + padding, h)

            hand_img = rgb[y_min:y_max, x_min:x_max]
            # hand_img = preprocess_img(hand_img, 0, 'RGB')  CNN
            hand_img = preprocess_img(hand_img, 0, None)  # MobilenetV2

            prediction_proba = model.predict(hand_img, verbose=0)
            prediction = class_names[np.argmax(prediction_proba)]
            prediction_queue.append(prediction)

            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (255, 0, 0), 2)

    if len(prediction_queue) == QUEUE_SIZE:
        most_common, count = Counter(prediction_queue).most_common(1)[0]
        if count > QUEUE_SIZE // 2:
            if most_common != last_letter and cooldown_counter == 0:
                if most_common == 'space':
                    sentence += ' '
                elif most_common == 'del':
                    sentence = sentence[:-1]
                else:
                    sentence += most_common
                last_letter = most_common
                cooldown_counter = cooldown_threshold
        else:
            last_letter = ''

    if cooldown_counter > 0:
        cooldown_counter -= 1

    cv2.putText(frame, sentence, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)
    if prediction:
        cv2.putText(frame, f"Detecting: {prediction}", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.2,
                    (255, 255, 255), 2)

    cv2.imshow("Sign Language Recognition", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
