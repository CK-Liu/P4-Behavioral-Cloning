import csv
import cv2
import numpy as np

lines = []
with open('./data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    i = 0
    for line in reader:
         if i > 0:
            lines.append(line)
         i =1
    print(lines[0])
    print(len(lines))
    
images = []
measurements = []
for line in lines:
#     for i in range(3):
#         local_path = './data/' + line[i]
#         img = cv2.imread(local_path)
#         images.append(img)
    local_path = './data/' + line[0]
    img = cv2.imread(local_path)
    image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    images.append(image)
    correction = 0.2
    measurement = float(line[3])
    measurements.append(measurement)
#     measurements.append(measurement+correction)
#     measurements.append(measurement-correction)

print(images[0].shape)
print(len(images))
print(len(measurements))
print(type(measurements[0]))
print(type(images[0]))
print(type(images))


augmented_images = []
augmented_measurements = []
for image, measurement in zip(images, measurements):
    augmented_images.append(image)
    augmented_measurements.append(measurement)
    flipped_image = cv2.flip(image, 1)
    flipped_measurement = -1.0 * measurement
    augmented_images.append(flipped_image)
    augmented_measurements.append(flipped_measurement)

shape = images[0].shape
# X_train = np.array(images)
# y_train = np.array(measurements)
X_train = np.array(augmented_images)
y_train = np.array(augmented_measurements)
print(X_train.shape)
print(type(X_train))
print(X_train[0].shape)
print(y_train.shape)
print(X_train.ndim)

import keras
from keras.models import Sequential, Model
from keras.layers.core import Flatten, Dense, Lambda
from keras.layers.convolutional import Conv2D
from keras.layers import Cropping2D
from keras.layers.pooling import MaxPooling2D

model = Sequential()
#model.add(Cropping2D(cropping=((50,20), (0,0)), input_shape=shape))
# model.add(Cropping2D(cropping=((50,20), (0,0)), input_shape=shape))
model.add(Cropping2D(cropping=((70,25), (0,0)), input_shape=shape))
model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=shape))


# model.add(Convolution2D(6,5,5,activation='relu'))
# model.add(MaxPooling2D())
# model.add(Convolution2D(16,5,5,activation='relu'))
# model.add(MaxPooling2D())
model.add(Conv2D(24, (5,5),strides=(2,2),activation='relu'))
model.add(Conv2D(36, (5,5),strides=(2,2),activation='relu'))
model.add(Conv2D(48, (5,5),strides=(2,2),activation='relu'))

# model.add(Convolution2D(36,5,5,subsample=(2,2),activation='relu'))
# model.add(Convolution2D(48,5,5,subsample=(2,2),activation='relu'))
# model.add(Convolution2D(64,5,5,activation='relu'))
# model.add(Convolution2D(64,5,5,activation='relu'))
model.add(Conv2D(64, (3,3),activation='relu'))
model.add(Conv2D(64, (3,3),activation='relu'))

model.add(Flatten())
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))

model.compile(optimizer='adam', loss='mse')
model.fit(X_train, y_train, validation_split=0.2, shuffle=True, epochs=5)

model.save('model.h5')
    
    