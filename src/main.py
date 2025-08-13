from collections import deque, Counter
import json
import os

import cv2
import mediapipe as mp
import numpy as np
from tensorflow.keras.models import load_model

from image_preprocessing import preprocess_img

BASE_DIR = os.path.dirname(__file__)
ROOT = os.path.dirname(BASE_DIR)
MODEL_PATH = os.path.join(ROOT, 'models', 'sign_language_model.keras')
CLASS_NAMES_PATH = os.path.join(ROOT, 'models', 'class_names.json')

model = load_model(MODEL_PATH)
with open(CLASS_NAMES_PATH, 'r') as f:
    class_names = json.load(f)

cap = cv2.VideoCapture(0)
hands = mp.solutions.hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

queue = deque(maxlen=15)
last_letter = ''
cooldown = 0
cooldown_threshold = 30
sentence = ''


def predict_sign():
    global last_letter, cooldown, sentence
    while True:
        ret, frame = cap.read()
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, _ = frame.shape

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
                # roi = preprocess_img(hand_img, 0, 'RGB')  CNN
                roi = preprocess_img(hand_img, 0, None)  # MobilenetV2

                prediction_proba = model.predict(roi, verbose=0)
                prediction = class_names[np.argmax(prediction_proba)]
                queue.append(prediction)

                # Majority vote
                if len(queue) == queue.maxlen:
                    most_common, count = Counter(queue).most_common(1)[0]
                    if count > queue.maxlen // 2:
                        if most_common != last_letter and cooldown == 0:
                            if most_common == 'space':
                                sentence += ' '
                            elif most_common == 'del':
                                sentence = sentence[:-1]
                            else:
                                sentence += most_common
                            last_letter = most_common
                            cooldown = cooldown_threshold
                    else:
                        last_letter = ''

                if cooldown > 0:
                    cooldown -= 1

                cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (255, 0, 0), 2)
                cv2.putText(frame, f"Detecting: {prediction}", (10, 100),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                cv2.putText(frame, sentence, (10, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)

        _, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
