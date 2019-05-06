import numpy as np
from keras.models import load_model
import cv2

import face_recognition

image = face_recognition.load_image_file('wonder.jpeg')
location = face_recognition.face_locations(image)[0]

print(location)

image = np.array(image)

test = image[location[0]:location[2], location[3]:location[1]]

test = cv2.resize(test, (64, 64))

model = load_model('CNNmodel.h5')

nbr = model.predict([[test]])
print(nbr)
