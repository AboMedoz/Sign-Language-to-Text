import json
import os

import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Conv2D, Dense, Dropout, Flatten, MaxPooling2D
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.image import ImageDataGenerator

BASE_DIR = os.path.dirname(__file__)
ROOT = os.path.dirname(BASE_DIR)
PROCESSED_DATA_PATH = os.path.join(ROOT, 'data', 'preprocessed')
MODEL_PATH = os.path.join(ROOT, 'models')

data = np.load(os.path.join(PROCESSED_DATA_PATH, 'preprocessed.npz'), allow_pickle=True)
x = data['x']
y = data['y']
class_names = data['class_names'].tolist()

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 1)))
model.add(MaxPooling2D(2, 2))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(2, 2))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(len(class_names), activation='softmax'))
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()
datagen = ImageDataGenerator(
    rotation_range=5,
    width_shift_range=0.05,
    height_shift_range=0.05,
    zoom_range=0.05,
    horizontal_flip=True
)  # Can confirm brightness_range fucks the training
datagen.fit(x_train)
model.fit(datagen.flow(x_train, y_train, batch_size=32), epochs=10, validation_data=(x_test, y_test))

_, accuray = model.evaluate(x_test, y_test)
print(f"Accuracy: {accuray * 100:.2f}")

model.save(os.path.join(MODEL_PATH, 'sign_language_model.keras'))
with open(os.path.join(MODEL_PATH, 'class_names.json'), 'w') as f:
    json.dump(class_names, f)
