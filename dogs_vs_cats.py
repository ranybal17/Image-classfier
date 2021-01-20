import numpy as np
import os
import glob

import matplotlib.pyplot as plt

from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from preprocess_function import preprocess

train_dir = 'data/train'
train_path = os.path.join(train_dir, '*g')
train_files = glob.glob(train_path)

test_dir = 'data/test'
test_path = os.path.join(test_dir, '*g')
test_files = glob.glob(test_path)

train_data = []
train_labels = []

test_data = []
test_labels = []

train_data, train_labels = preprocess(train_files)
test_data, test_labels = preprocess(test_files)

# train_data, train_labels = train_data[:1000], train_labels[:1000]

# model = Sequential()

# model.add(Conv2D(32, (3,3), activation='relu', input_shape=(128,128,3)))
# model.add(MaxPooling2D(pool_size=(2,2)))
# model.add(Dropout(0.2))

# model.add(Conv2D(64, (3,3), activation='relu', input_shape=(128,128,3)))
# model.add(MaxPooling2D(pool_size=(2,2)))
# model.add(Dropout(0.2))

# model.add(Conv2D(64, (3,3), activation='relu'))
# model.add(MaxPooling2D(pool_size=(2,2)))
# model.add(Dropout(0.2))

# model.add(Conv2D(128, (3,3), activation='relu'))
# model.add(MaxPooling2D(pool_size=(2,2)))
# model.add(Dropout(0.2))

# model.add(Flatten())
# model.add(Dense(512, activation='relu'))
# model.add(Dense(2, activation='softmax'))

# model.summary()

# 95.65%

model = Sequential()

model.add(Conv2D(32, (3,3), activation='relu', input_shape=(128, 128, 3)))
model.add(MaxPooling2D((2,2)))
model.add(Dropout(0.2))

model.add(Conv2D(64, (3,3), activation='relu'))
model.add(MaxPooling2D((2,2)))
model.add(Dropout(0.2))

model.add(Conv2D(64, (3,3), activation='relu'))
model.add(MaxPooling2D((2,2)))
model.add(Dropout(0.2))

model.add(Conv2D(128, (3,3), activation='relu'))
model.add(MaxPooling2D((2,2)))
model.add(Dropout(0.2))

model.add(Conv2D(128, (3,3), activation='relu'))

model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(2, activation='softmax'))

model.summary()

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

history = model.fit(train_data, train_labels, epochs=18, validation_data=(test_data, test_labels))
model.save("model6_catsVSdogs_10epoch.h5")














