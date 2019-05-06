from keras.utils import to_categorical
from PIL import Image
import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPool2D, Dropout, Flatten
from keras import backend as K
from sklearn.model_selection import train_test_split
import cv2


import pickle
import os


with open('mylabel.pkl', 'rb') as f:
    y_train = pickle.load(f)


def getData(path):
    faces = []
    labels = []
    i = 0
    for label in ['male', 'female']:
        for f in os.listdir(path+label):
            # print(f)
            image = cv2.imread(path+label+'/'+f)
            image = np.array(image)
            faces.append(image)

        i += 1

    return faces


x_train = getData('data/')

y_train = to_categorical(y_train)

print(y_train)


x_train = np.array(x_train)
print(x_train.shape)

x_train, x_test, y_train, y_test = train_test_split(
    x_train, y_train, test_size=0.1, random_state=20)


with open('mydata.pkl', 'wb') as f:
    pickle.dump(x_train, f)

if K.image_data_format() == 'channels_first':
    x_train = x_train.reshape(x_train.shape[0], 3, 64, 64)
    input_shape = (3, 64, 64)
else:
    x_train = x_train.reshape(x_train.shape[0], 64, 64, 3)
    input_shape = (64, 64, 3)

model = Sequential()
model.add(Conv2D(64, kernel_size=(3, 3), strides=(1, 1),
                 activation='relu', input_shape=input_shape))
model.add(Conv2D(32, kernel_size=(3, 3), strides=(2, 2)))
model.add(MaxPool2D(pool_size=(2, 2), strides=(3, 3)))
model.add(Flatten())
model.add(Dense(50, activation='relu'))
model.add(Dense(2, activation='softmax'))

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.SGD(lr=0.005),
              metrics=['accuracy'])

his = model.fit(x_train, y_train,
                batch_size=200,
                epochs=20,
                verbose=2,
                validation_data=(x_test, y_test))

model.save("CNNmodel.h5")
