from random import shuffle
import glob
import os, sys
import cv2
import numpy as np
import tensorflow as tf
import pickle
import pydot
import time
import matplotlib.pyplot as plt
import keras
import scipy.ndimage as nd
import graphviz
from keras.engine.training_generator import fit_generator
import Kanjibot_img2tfrecord as kb
from keras.models import Sequential
from keras.layers import BatchNormalization
from keras.layers.core import Flatten, Dense, Dropout, Activation
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D, Conv2D
from keras.preprocessing.sequence import pad_sequences
from keras.utils.vis_utils import plot_model
from keras.callbacks import CSVLogger
from sklearn.utils import compute_class_weight, class_weight
from sklearn.model_selection import train_test_split
from sklearn import tree
from ann_visualizer.visualize import ann_viz
from IPython.display import SVG
from keras import activations, optimizers

k = kb.KanjibotImg2TFrecord()

config = tf.ConfigProto(allow_soft_placement=True)
config.gpu_options.allocator_type = 'BFC'
config.gpu_options.allow_growth = True

# def _parse(serialized_example):
# 	feature = {'train/image': tf.FixedLenFeature([], tf.string),
# 			   'train/label': tf.FixedLenFeature([], tf.int64)}
# 	# decode the record
# 	features = tf.parse_single_example(serialized_example, features=feature)
# 	# convert the image data from a string back to numbers
# 	image = tf.decode_raw(features['train/image'], tf.int64)
# 	# cast label data into int32
# 	label = tf.cast(features['train/label'], tf.int32)
# 	return (dict({'image':image}), label)


data_path_train = os.path.abspath('./TFRecord/kanji_train.tfrecord')
data_path_test = os.path.abspath('./TFRecord/kanji_test.tfrecord')
data_path_val = os.path.abspath('./TFRecord/kanji_val.tfrecord')

batch_size = 50

# data_paths = [data_path_train, data_path_test, data_path_val]
# dataset_train = tf.data.TFRecordDataset(data_paths)
# dataset_train = dataset_train.map(shuffle(True).batch(16))
# iterator = dataset_train.make_one_shot_iterator()
# iterator_final = iterator.get_next()

# def tfrecord_train_input_fn(batch_size=32):
# 	dataset = tf.data.TFRecordDataset(data_path_train)
# 	dataset = dataset.map(lambda x: _parse(x)).shuffle(True).batch(batch_size)
# 	iterator = dataset.make_one_shot_iterator()
# 	return iterator.get_next()


# create a list of filenames and pass it to a FIFO queue
filename_queue = tf.train.string_input_producer([data_path_train], num_epochs=1)

# define a reader and read the next record
reader = tf.TFRecordReader()
_, serialized_example, = reader.read(filename_queue)

with tf.Session(config=config) as sess:
	feature = {'train/image': tf.FixedLenFeature([], tf.string),
			   'train/label': tf.FixedLenFeature([], tf.int64)}
	# decode the record
	features = tf.parse_single_example(serialized_example, features=feature)
	# convert the image data from a string back to numbers
	image = tf.decode_raw(features['train/image'], tf.int32)
	# cast label data into int32
	label = tf.cast(features['train/label'], tf.int32)

	# reshape the image to its original shape
	image = tf.reshape(image, [64, 64, 3])
	print(image.shape)

	# preprocessing here
	# train_datagen = keras.preprocessing.image.ImageDataGenerator()

	# creates batches by randomly shuffling tensors
	images, labels = tf.train.shuffle_batch([image, label], batch_size=batch_size, capacity=50, num_threads=1, min_after_dequeue=10)
	# images, labels = tf.train.batch([image, label], batch_size=16, capacity=50, num_threads=1)
#

	# initialize all global and local variables
	init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
	sess.run(init_op)

	# create a coordinator and run all Queuerunner objects
	coord = tf.train.Coordinator()
	threads = tf.train.start_queue_runners(coord=coord)

	for batch_index in range(6):
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

model.add(Conv2D(64, kernel_size=(3, 3), input_shape=(64, 64, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(BatchNormalization())

model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(BatchNormalization())

model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(BatchNormalization())

model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(BatchNormalization())
model.add(Dropout(0.2))

model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(50, activation='softmax'))

# Define the learning rate (lower = less weight shifts)
adam = optimizers.Adam(lr=0.001)

# Define a logger to save progress mad eduring training
csv_logger = CSVLogger('Warmind_Nobunaga_log.csv', append=True, separator=',')
# compile the model prior to training
model.compile(optimizer=adam, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
history = model.fit(k.data_split()[0], k.data_split()[2], batch_size=batch_size, epochs=210, class_weight=class_weight_dict, verbose=1, callbacks=[csv_logger])
# model.fit_generator(train_datagen.flow(k.data_split()[0], k.data_split()[2], batch_size=batch_size), steps_per_epoch=len(k.data_split()[0]) / batch_size, epochs=10, class_weight=class_weight_dict, verbose=1)
# List all the data in history
# print(history.history.keys())
# Summarize history for accuracy
# plt.plot(history.history['acc'])
# plt.title('model accuracy')
# plt.ylabel('accuracy')
# plt.xlabel('epoch')
# plt.legend(['train', 'test'], loc='upper left')
# plt.show()
# # summarize history for loss
# plt.plot(history.history['loss'])
# plt.title('model loss')
# plt.ylabel('loss')
# plt.xlabel('epoch')
# plt.legend(['train', 'test'], loc='upper left')
# plt.show()

scores = model.evaluate(k.data_split()[1], k.data_split()[3])
print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
print(model.summary())
plot_model(model, to_file='Warmind_Nobunaga.png', show_shapes=True, show_layer_names=True)
top_layer = model.layers[0]
plt.imshow(top_layer.get_weights()[0][:, :, :, 0].squeeze(), cmap='gray')
