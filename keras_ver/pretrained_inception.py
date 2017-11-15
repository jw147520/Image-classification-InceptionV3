from keras.applications.inception_v3 import *
from keras.preprocessing import image
import numpy as np

model = InceptionV3(weights='imagenet')
model.compile(optimizer='rmsprop', loss='categorical_crossentropy')
model.summary()

image_path = '../data/2classes/benign/SOB_B_A-14-22549AB-40-002.jpg'
img = image.load_img(image_path, target_size=(224, 224))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)

prediction = model.predict(x)
print(prediction)

