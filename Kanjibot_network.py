import os
import math
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import keras
import Kanjibot_img2tfrecord as kb
import LRFinder
from keras.models import Sequential
from keras.applications import ResNet50, VGG16, VGG19, InceptionResNetV2, InceptionV3
from keras.layers import BatchNormalization, Conv2D
from keras.layers.core import Flatten, Dense, Dropout, Activation
from keras.utils.vis_utils import plot_model
from keras.callbacks import CSVLogger, LearningRateScheduler
from sklearn.utils import compute_class_weight
from keras import activations, optimizers

k = kb.KanjibotImg2TFrecord()

config = tf.ConfigProto(allow_soft_placement=True)
config.gpu_options.allocator_type = 'BFC'
config.gpu_options.allow_growth = True

seed = 128
rng = np.random.RandomState(seed)

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

k49_train_images = np.load("./Dataset/k49-train-imgs.npz")
k49_train_labels = np.load("./Dataset/k49-train-labels.npz")
k49_test_images = np.load("./Dataset/k49-test-imgs.npz")
k49_test_labels = np.load("./Dataset/k49-test-labels.npz")


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

# Define a batch size
batch_size = 128

with tf.Session(config=config) as sess:
	feature = {'image': tf.FixedLenFeature([], tf.string),
			   'label': tf.FixedLenFeature([], tf.int64)}
	# decode the record
	features = tf.parse_single_example(serialized_example, features=feature)
	# convert the image data from a string back to numbers
	image = tf.decode_raw(features['image'], tf.int32)
	# cast label data into int32
	label = tf.cast(features['label'], tf.int32)

	# reshape the image to its original shape
	image = tf.reshape(image, [75, 75, 3])
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

# set up the correct weights as not to deal with uneven amounts of images in a class
class_weight_list = compute_class_weight('balanced', np.unique(k.data_split()[2]), k.data_split()[2])
class_weight_dict = dict(zip(np.unique(k.data_split()[2]), class_weight_list))

# set up the keras model, using input of 2d images of 64 * 64
# The output layer has 50 nodes, one for each image class
# Possible other network: ResNet, LeNet
# Implement Dropouts of different sizes
# google cloud platform
# ImgAug library
input_num_units = 128
hidden_num_units = 500
output_num_units = 50
dropout_ratio = 0.5

conv_base = ResNet50(weights='imagenet', include_top=False, input_shape=(75, 75, 3))

model = Sequential()
model.add(conv_base)
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dense(50, activation='softmax'))
model.summary()

print('Number of trainable weights before freezing the conv base: ', len(model.trainable_weights))
conv_base.trainable = False
print('Number of trainable weights after freezing the conv base: ', len(model.trainable_weights))

# model = Sequential
#
# model.add(Dense(75, input_shape=(75, 75, 3), activation='relu'))
# model.add(MaxPooling2D(pool_size=(2, 2)))
# # model.add(BatchNormalization())
# # model.add(Dropout(dropout_ratio))
#
# model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
# model.add(MaxPooling2D(pool_size=(2, 2)))
# # model.add(BatchNormalization())
# # model.add(Dropout(dropout_ratio))
#
# model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
# model.add(MaxPooling2D(pool_size=(2, 2)))
# # model.add(BatchNormalization())
# # model.add(Dropout(dropout_ratio))
#
# model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
# model.add(MaxPooling2D(pool_size=(2, 2)))
# # model.add(BatchNormalization())
# # model.add(Dropout(dropout_ratio))
#
# model.add(Flatten())
# model.add(Dropout(dropout_ratio))
# model.add(Dense(128, activation='relu'))
# model.add(Dense(output_dim=50, input_dim=128, activation='softmax'))


# Define the learning rate (lower = less weight shifts)
def step_decay_scheduler(initial_lrate=1e-3, decay_factor=0.75, step_size=10):
	# Wrapper function to create a LearningRateScheduler with step decay schedule.
	def schedule(epoch):
		return initial_lrate * (decay_factor ** np.floor(epoch/step_size))
	return LearningRateScheduler(schedule)

def exp_decay(epoch, lr):
	drop = 0.5
	epochs_drop = 2.0
	lrate = lr * math.pow(drop, math.floor((1+epoch)/(epochs_drop)))
	return lrate

epochs = 50
learning_rate = 1e-5

lr_exp = LearningRateScheduler(exp_decay(epochs, learning_rate))

lr_sched = step_decay_scheduler(initial_lrate=1e-4, decay_factor=0.75, step_size=2)
lr_finder = LRFinder.LRFinder(min_lr=1e-5, max_lr=1e-2, steps_per_epoch=(epochs/batch_size), epochs=3)


decay_rate = learning_rate / epochs
momentum = 0.8
sgd = optimizers.SGD(lr=learning_rate, momentum=momentum, decay=decay_rate, nesterov=False)
adam = optimizers.Adam(lr=learning_rate, decay=decay_rate)
rmspop = optimizers.RMSprop(lr=learning_rate)

# Define a logger to save progress made during training
csv_logger = CSVLogger('Warmind_Nobunaga_log.csv', append=True, separator=',')
# compile the model prior to training
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
# history = model.fit(train_image_list, train_labels, batch_size=batch_size, epochs=epochs, class_weight=class_weight_dict, verbose=1, callbacks=[lr_finder])
history = model.fit_generator(train_generator, steps_per_epoch=train_nimg // batch_size, epochs=epochs,
 							  validation_data=val_generator, validation_steps=val_nimg // batch_size,
 							  class_weight=class_weight_dict, verbose=1)
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
model.save('kanjibot_network.h5')
