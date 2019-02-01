from random import shuffle
import glob
import os
import cv2
import numpy as np
import tensorflow as tf
import pydot
import sys
import matplotlib.pyplot as plt
import keras
import scipy.ndimage as nd
import graphviz
from keras.engine.training_generator import fit_generator
import Kanjibot_img2tfrecord as kb
from keras.models import Sequential
from keras.layers.core import Flatten, Dense, Dropout, Activation
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D, Conv2D
from keras.preprocessing.sequence import pad_sequences
from keras.utils.vis_utils import plot_model
from sklearn.utils import compute_class_weight, class_weight
from sklearn.model_selection import train_test_split
from sklearn import tree
from ann_visualizer.visualize import ann_viz
from IPython.display import SVG
from keras import activations, optimizers

k = kb.KanjibotImg2TFrecord()

data_path = 'TFRecord\\kanji_train.tfrecord'

# create a list of filenames and pass it to a FIFO queue
filename_queue = tf.train.string_input_producer([data_path], num_epochs=1)

# define a reader and read the next record
reader = tf.TFRecordReader()
_, serialized_example, = reader.read(filename_queue)

with tf.Session() as sess:
	feature = {'train/image': tf.FixedLenFeature([], tf.string),
			   'train/label': tf.FixedLenFeature([], tf.int64)}

	# decode the record
	features = tf.parse_single_example(serialized_example, features=feature)

	# convert the image data from a string back to numbers
	image = tf.decode_raw(features['train/image'], tf.float32)

	# cast label data into int32
	label = tf.cast(features['train/label'], tf.int32)

	# reshape the image to its original shape
	image = tf.reshape(image, [64, 64, 3])
	print(image.shape)

	# preprocessing here
	# train_datagen = keras.preprocessing.image.ImageDataGenerator()

	# creates batches by randomly shuffling tensors
	images, labels = tf.train.shuffle_batch([image, label], batch_size=10, capacity=50, num_threads=1, min_after_dequeue=0)

	# initialize all global and local variables
	init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
	sess.run(init_op)

	# create a coordinator and run all Queuerunner objects
	coord = tf.train.Coordinator()
	threads = tf.train.start_queue_runners(coord=coord)

	for batch_index in range(5):
		img, lbl = sess.run([images, labels])
		# img = img.astype(np.uint8)

		for j in range(6):
			plt.subplot(2, 3, j+1)
			plt.imshow(img[j, ...])
			plt.title(lbl)

		plt.show()

	# stop the threads
	coord.request_stop()

	# wait for threads to stop
	coord.join(threads)
	sess.close()

# set up the correct weights as not to deal with uneven amounts of images in a class
class_weight_list = compute_class_weight('balanced', np.unique(k.data_split()[2]), k.data_split()[2])
class_weight_dict = dict(zip(np.unique(k.data_split()[2]), class_weight_list))

# set up the keras model, using input of 2d images of 64 * 64
# set up the model's layers. The first Dense layer has 4096 nodes (for the amount of pixels per image),
# and the output layer has 50 nodes, one for each image class
model = Sequential()

model.add(Conv2D(32, (3, 3), input_shape=(64, 64, 3), activation='relu'))

model.add(Conv2D(32, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(4096, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(50, activation='softmax'))
model.add(Dropout(0.25))

# Define the learning rate (lower = less weight shifts)
# adam = optimizers.Adam(lr=0.001)

# compile the model prior to training
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(k.data_split()[0], k.data_split()[2], batch_size=128, epochs=10, class_weight=class_weight_dict)

# model.fit_generator(train_datagen.flow(k.data_split()[0], k.data_split()[2], batch_size=64),
# 					steps_per_epoch=len(k.data_split()[0]) / 64, epochs=10, class_weight=class_weight_dict)

scores = model.evaluate(k.data_split()[0], k.data_split()[2])
print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
print(model.summary())
plot_model(model, to_file='Warmind_Nobunaga.png', show_shapes=True, show_layer_names=True)
top_layer = model.layers[0]
plt.imshow(top_layer.get_weights()[0][:, :, :, 0].squeeze(), cmap='gray')
