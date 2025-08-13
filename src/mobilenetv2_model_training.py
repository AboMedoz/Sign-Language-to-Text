import os

import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

BASE_DIR = os.path.dirname(__file__)
ROOT = os.path.dirname(BASE_DIR)
DATA_PATH = os.path.join(ROOT, 'data', 'preprocessed')
MODELS_PATH = os.path.join(ROOT, 'models')

data = np.load(os.path.join(DATA_PATH, 'preprocessed.npz'), allow_pickle=True)

x = data['x']
y = data['y']
class_names = data['class_names'].tolist()


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(96, 96, 3))

xs = base_model.output
xs = GlobalAveragePooling2D()(xs)
xs = Dropout(0.3)(xs)
predictions = Dense(len(class_names), activation='softmax')(xs)
model = Model(inputs=base_model.input, outputs=predictions)
for layer in base_model.layers:
    layer.trainable = False
model.compile(optimizer=Adam(learning_rate=1e-3), loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(x_train, y_train, batch_size=32, epochs=5, validation_data=(x_test, y_test))
model.save(os.path.join(MODELS_PATH, 'mobilenetv2_sign_language_model.h5'))
