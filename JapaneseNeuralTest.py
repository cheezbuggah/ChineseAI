from random import shuffle
import glob
import os
import cv2
import numpy as np
import tensorflow as tf
import sys
import matplotlib.pyplot as plt
import keras
import scipy.ndimage as nd
from keras.engine.training_generator import fit_generator
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.preprocessing.sequence import pad_sequences
from keras.utils import np_utils
from sklearn.utils import compute_class_weight, class_weight
from sklearn.model_selection import train_test_split

shuffle_data = True
kanji_path = ".\\Chars\\Kanji_chars\\"
dir_paths = os.listdir(kanji_path)
addrs = glob.glob(".\\Chars\\Kanji_chars\\*\\*.png")
labels = {l: i for i, l in enumerate(dir_paths)}
print(labels)


if shuffle_data:
	c_labels = list(labels)
	c_addrs = list(addrs)
	c = [c_addrs, c_labels]
	shuffle(c)
	addrs, labels = list(c)

label_list = []
index_labels = []
i = 0
for addres in addrs:
	label_list.append(addres.split('\\')[3])
	for item in labels:
		if label_list[i] == item:
			index_labels.append(labels.index(item))
	i += 1

# # Training: 60%
# train_addrs = addrs[0:int(0.6*len(addrs))]
# train_labels = index_labels[0:int(0.6*len(index_labels))]
#
# # Validation: 20%
# val_addrs = addrs[int(0.6*len(addrs)):int(0.8*len(addrs))]
# val_labels = index_labels[int(0.6*len(index_labels)):int(0.8*len(index_labels))]
#
# # Test: 20%
# test_addrs = addrs[int(0.8*len(addrs)):]
# test_labels = index_labels[int(0.8*len(index_labels)):]

(train_addrs, test_addrs, train_labels, test_labels) = train_test_split(addrs, index_labels, test_size=0.4)
(val_addrs, test_addrs, val_labels, test_labels) = train_test_split(test_addrs, test_labels, test_size=0.5)


def load_image(addr):
	addr = str(addr)
	img = cv2.imread(addr)
	img = cv2.resize(img, (64, 64), interpolation=cv2.INTER_CUBIC)
	img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
	img = img.astype(np.float32)
	#print(addr)
	#img = img_to_array(img)
	return img

def _int64_feature(value):
	return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _bytes_feature(value):
	return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


# Training set
train_filename = 'TFRecord\\kanji_train.tfrecord'
writer = tf.python_io.TFRecordWriter(train_filename)

for i in range(len(train_addrs)):
	if not i % 1000:
		print('Train data: {}/{}'.format(i, len(train_addrs)))
		sys.stdout.flush()

	img = load_image(train_addrs[i])
	label = train_labels[i]

	# create a feature
	feature = {'train/label': _int64_feature(label),
			   'train/image': _bytes_feature(tf.compat.as_bytes(img.tostring()))}

	# create an example protocol buffer
	example = tf.train.Example(features=tf.train.Features(feature=feature))

	# serialize to string and write to file
	writer.write(example.SerializeToString())

writer.close()
sys.stdout.flush()

# Validation set
val_filename = 'TFRecord\\kanji_val.tfrecord'
writer = tf.python_io.TFRecordWriter(val_filename)

for i in range(len(val_addrs)):
	if not i % 1000:
		print('Val data: {}/{}'.format(i, len(val_addrs)))
		sys.stdout.flush()

	img = load_image(val_addrs[i])
	label = val_labels[i]

	feature = {'val/label': _int64_feature(label),
			   'val/image': _bytes_feature(tf.compat.as_bytes(img.tostring()))}

	example = tf.train.Example(features=tf.train.Features(feature=feature))
	writer.write(example.SerializeToString())

writer.close()
sys.stdout.flush()

# Test set
test_filename = 'TFRecord\\kanji_test.tfrecord'
writer = tf.python_io.TFRecordWriter(test_filename)

for i in range(len(test_addrs)):
	if not i % 1000:
		print('Test data: {}/{}'.format(i, len(test_addrs)))
		sys.stdout.flush()

	img = load_image(test_addrs[i])
	label = test_labels[i]
	feature = {'test/label': _int64_feature(label),
			   'test/image': _bytes_feature(tf.compat.as_bytes(img.tostring()))}

	example = tf.train.Example(features=tf.train.Features(feature=feature))
	writer.write(example.SerializeToString())

writer.close()
sys.stdout.flush()

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
	# image = image.astype('float32')

	# datagen = keras.preprocessing.image.ImageDataGenerator(
	# 	featurewise_center=False,
	# 	featurewise_std_normalization=False,
	# 	rotation_range=0,
	# 	width_shift_range=0.1,
	# 	height_shift_range=0.1,
	# 	horizontal_flip=False
	# )
	# datagen.fit(image)

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
		#img = img.astype(np.uint8)

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
class_weight_list = compute_class_weight('balanced', np.unique(train_labels), train_labels)
class_weight_dict = dict(zip(np.unique(train_labels), class_weight_list))

# set up the keras model, using input of 2d images of 64 * 64
# set up the model's layers. The first Dense layer has 128 nodes, and the output layer
# has 50 nodes, one for each image class
model = Sequential([
	Flatten(input_shape=(64, 64)),
	Dense(128, activation=tf.nn.relu),
	Dense(50, activation=tf.nn.softmax)
])

# compile the model prior to training
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(train_addrs, train_labels, batch_size=32, epochs=10, class_weight=class_weight_dict)
