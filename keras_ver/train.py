# -*- coding: utf-8 -*-
import keras
from keras.applications.inception_v3 import InceptionV3
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D
from keras import backend as K
import numpy as np
from time import time
from load_data import *


nb_classes = 2  # number of classes
data_path = '../data/2classes/'  # input data path
img_width = 139
img_height = 139
img_channel = 3
img_shape = (img_width, img_height, img_channel)
batch_size = 32

# create the base pre-trained model
base_model = InceptionV3(weights='imagenet', include_top=False, input_shape=img_shape)
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu')(x)
predictions = Dense(nb_classes, activation='softmax')(x)

# this is the model to train
model = Model(inputs=base_model.input, outputs=predictions)

# train only the top layers (which were randomly initialized)
# i.e. freeze all convolutional InceptionV2 layers
for layer in base_model.layers:
	layer.trainable = False

# compile the model (should be done "after" setting layers to non-trainable)
model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

loss_training_file = open('loss_training.txt', 'w')
loss_val_file = open('loss_val.txt', 'w')
acc_training_file = open('acc_training.txt', 'w')
acc_val_file = open('acc_val.txt', 'w')


print("Start Training...")
tbCallBack = keras.callbacks.TensorBoard(log_dir='./log'.format(time()), histogram_freq=0, write_graph=True,
                                         batch_size=16, write_images=True)
batch_count = 0
try:
	for i in range(0, 20):
		print('--------------- On epoch: ' + str(i) + ' ---------------')
		for x_data, y_label, classlist in load_batch(data_path, batch_size):
			x_data = np.array(x_data)
			print(x_data.shape)

			history = model.fit(x_data, y_label, verbose=1, epochs=1, validation_split=.2)
			batch_count += 1

			if(batch_count % 20) == 0:
				print("Saving checkpoint " + str(batch_count))
				model.save('model_chkp_' + str(batch_count) + '.h5')
				print("Checkpoint saved. Continuing...")

except Exception as e:
	print("Excepted with " + str(e))
	print("Saving model...")
	model.save('excepted_model.h5')
	print("Model saved.")

# plot_model(model, to_file='model.png')


"""
# at this point, the top layers are well trained and we can start fine-tuning
# convolutional layers from inception V3. we will freeze the bottom N layers
# and train the remaining top layers.

# visualize layer names and layer indices to see how many layers
# we should freeze:
for i, layer in enumerate(base_model.layers):
	print(i, layer.name)

# chose to train the top 2 inception blocks, i.e. we will freeze
# the first 249 layers and unfreeze the rest:
for layer in model.layers[:249]:
	layer.trainable = False
for layer in model.layers[249:]:
	layer.trainable = True

# we need to recompile the model for these modifications to take effect
# we use SGD with a low learning rate
from keras.optimizers import SGD
model.compile(optimizer=SGD(lr=0.0001, momentum=0.9), loss='categorical_crossentropy')

# train the model again (this time fine-tuning the top 2 inception blocks
# alongside the top Dense layers
model.fit()

"""

