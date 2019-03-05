import tensorflow as tf
import os
import matplotlib.pyplot as plt
import numpy as np
import keras
from keras import layers
from keras import models
from sklearn.utils import compute_class_weight
from keras.utils.vis_utils import plot_model

import Kanjibot_img2tfrecord as kb
import LRFinder

k = kb.KanjibotImg2TFrecord()

config = tf.ConfigProto(allow_soft_placement=True)
config.gpu_options.allocator_type = 'BFC'
config.gpu_options.allow_growth = True

seed = 128
rng = np.random.RandomState(seed)

# data retrieval
data_path_train = os.path.abspath('./TFRecord/kanji_train.tfrecord')
train_nimg = 0
for record in tf.python_io.tf_record_iterator(data_path_train):
		train_nimg += 1
data_path_test = os.path.abspath('./TFRecord/kanji_test.tfrecord')
test_nimg = 0
for record in tf.python_io.tf_record_iterator(data_path_test):
		test_nimg += 1
data_path_val = os.path.abspath('./TFRecord/kanji_val.tfrecord')
val_nimg = 0
for record in tf.python_io.tf_record_iterator(data_path_test):
		val_nimg += 1

print("Amount of training images: " + str(train_nimg))
print("Amount of testing images: " + str(test_nimg))
print("Amount of validation images: " + str(val_nimg))

train_image_list = k.data_split()[0]
train_labels = k.data_split()[2]
test_image_list = k.data_split()[1]
test_labels = k.data_split()[3]
val_image_list = k.data_split()[4]
val_labels = k.data_split()[5]
train_images = k.data_split()[10]
test_images = k.data_split()[7]
val_images = k.data_split()[6]

# create a list of filenames and pass it to a FIFO queue
filename_queue = tf.train.string_input_producer([data_path_train], num_epochs=1)

# define a reader and read the next record
reader = tf.TFRecordReader()
_, serialized_example, = reader.read(filename_queue)

# Define a batch size
batch_size = 128

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

	# preprocessing here; create a data generator to provide more training samples
	print("Initializing the data generators...")
	train_datagen = keras.preprocessing.image.ImageDataGenerator(
		rotation_range=1,
		width_shift_range=0.1,
		height_shift_range=0.1,
		rescale=1./255,
		horizontal_flip=False,
		fill_mode='nearest'
	)

	train_generator = train_datagen.flow(train_image_list, train_labels, batch_size=batch_size)
	print("Succesfully initialized the Train data generator!")

	test_datagen = keras.preprocessing.image.ImageDataGenerator(rescale=1./255)

	test_generator = test_datagen.flow(test_image_list, test_labels, batch_size=batch_size)
	print("Succesfully initialized the Test data generator!")

	val_generator = test_datagen.flow(val_image_list, val_labels, batch_size=batch_size)
	print("Succesfully initialized the Validation data generator!")

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

# Image dimensions

img_height = 64
img_width = 64
channels = 3

# Network parameters
cardinality = 32
epochs = 80

def residual_network(x):
	# ResNeXt by default. For ResNet set `cardinality` = 1 above
	def add_common_layers(y):
		y = layers.BatchNormalization()(y)
		y = layers.LeakyReLU()(y)

		return y

	def grouped_conv(y, nb_channels, _strides):
		# when `cardinality` == 1 this is just a standard convolution
		if cardinality == 1:
			return layers.Conv2D(nb_channels, kernel_size=(3, 3), strides=_strides, padding='same')(y)

		assert not nb_channels % cardinality
		_d = nb_channels // cardinality

		# in a grouped convolution layer, input and output channels are divided into `cardinality` groups,
		# and convolutions are separately performed within each group
		groups = []
		for j in range(cardinality):
			group = layers.Lambda(lambda z: z[:, :, :, j * _d:j * _d + _d])(y)
			groups.append(layers.Conv2D(_d, kernel_size=(3, 3), strides=_strides, padding='same',)(group))

		# the grouped convolutional layer concatenates them as the outputs of the layer
		y = layers.concatenate(groups)

		return y

	def residual_block(y, nb_channels_in, nb_channels_out, _strides=(1, 1), _project_shortcut=False):
		"""
		        Our network consists of a stack of residual blocks. These blocks have the same topology,
		        and are subject to two simple rules:
		        - If producing spatial maps of the same size, the blocks share the same hyper-parameters (width and filter sizes).
		        - Each time the spatial map is down-sampled by a factor of 2, the width of the blocks is multiplied by a factor of 2.
		"""
		shortcut = y

		# we modify the residual building block as a bottleneck design to make the network more economical
		y = layers.Conv2D(nb_channels_in, kernel_size=(1, 1), strides=(1, 1), padding='same')(y)
		y = add_common_layers(y)

		# ResNeXt (Identical to ResNet if 'cardinality' == 1
		y = grouped_conv(y, nb_channels_in, _strides=_strides)
		y = add_common_layers(y)

		y = layers.Conv2D(nb_channels_out, kernel_size=(1, 1), strides=(1, 1), padding='same')(y)
		# batch normalization is employed after aggregating the transformations and before adding to the shortcut
		y = layers.BatchNormalization()(y)

		# identity shortcuts used directly when the input and output are of the same dimensions
		if _project_shortcut or _strides != (1, 1):
			# when the dimensions increase projection shortcut is used to match dimensions (done by 1×1 convolutions)
			# when the shortcuts go across feature maps of two sizes, they are performed with a stride of 2
			shortcut = layers.Conv2D(nb_channels_out, kernel_size=(1, 1), strides=_strides, padding='same')(shortcut)
			shortcut = layers.BatchNormalization()(shortcut)

		y = layers.add([shortcut, y])

		# relu is performed right after each batch normalization,
		# expect for the output of the block where relu is performed after the adding to the shortcut
		y = layers.LeakyReLU()(y)

		return y

	# conv layer 1:
	x = layers.Conv2D(64, kernel_size=(7, 7), strides=(2, 2), padding='same')(x)
	x = add_common_layers(x)

	# conv layer 2:
	x = layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same')(x)
	for i in range(3):
		project_shortcut = True if i == 0 else False
		x = residual_block(x, 32, 64, _project_shortcut=project_shortcut)

	# conv layer 3:
	for i in range(4):
		# down-sampling is performed by conv3_1, conv4_1, and conv5_1 with a stride of 2
		strides = (2, 2) if i == 0 else (1, 1)
		x = residual_block(x, 64, 128, _strides=strides)

	# conv layer 4:
	for i in range(6):
		strides = (2, 2) if i == 0 else (1, 1)
		x = residual_block(x, 128, 256, _strides=strides)

	# conv layer 5:
	for i in range(3):
		strides = (2, 2) if i == 0 else (1, 1)
		x = residual_block(x, 256, 512, _strides=strides)

	x = layers.GlobalAveragePooling2D()(x)
	x = layers.Dense(50)(x)

	return x

image_tensor = layers.Input(shape=(img_height, img_width, channels))
network_output = residual_network(image_tensor)

model = models.Model(inputs=[image_tensor], outputs=[network_output])
print(model.summary())

class_weight_list = compute_class_weight('balanced', np.unique(k.data_split()[2]), k.data_split()[2])
class_weight_dict = dict(zip(np.unique(k.data_split()[2]), class_weight_list))

lr_finder = LRFinder.LRFinder(min_lr=1e-5, max_lr=1e-2, steps_per_epoch=(epochs/batch_size), epochs=3)
learning_rate = 0.1
decay_rate = learning_rate / epochs
momentum = 0.8
sgd = keras.optimizers.SGD(lr=learning_rate, momentum=momentum, decay=decay_rate, nesterov=False)
adam = keras.optimizers.Adam(lr=learning_rate, decay=decay_rate)

model.compile(optimizer=adam, loss='sparse_categorical_crossentropy', metrics=['accuracy'])

history = model.fit_generator(train_generator, steps_per_epoch=train_nimg // batch_size, epochs=epochs,
 							  validation_data=val_generator, validation_steps=val_nimg // batch_size,
 							  class_weight=class_weight_dict, verbose=1, callbacks=[lr_finder])
# history = model.fit(train_image_list, train_labels, batch_size=batch_size, epochs=epochs,
# 					class_weight=class_weight_dict, verbose=1, callbacks=[lr_finder])

model.save('kanjibot_network.h5')
# List all the data in history
print(history.history.keys())
lr_finder.plot_loss()
lr_finder.plot_lr()
# Summarize history for accuracy
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

scores = model.evaluate(test_image_list, test_labels)
print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
print(model.summary())
plot_model(model, to_file='Warmind_Nobunaga.png', show_shapes=True, show_layer_names=True)
top_layer = model.layers[0]
plt.imshow(top_layer.get_weights()[0][:, :, :, 0].squeeze(), cmap='gray')