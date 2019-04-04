import tensorflow as tf
import random
import shutil
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import os
from dataset_utils import _dataset_exists, _get_filenames_and_classes, write_label_file, _convert_dataset
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.preprocessing.image import ImageDataGenerator

flags = tf.flags

#State your dataset directory
flags.DEFINE_string('dataset_dir', 'Chars', 'String: Your dataset directory')

# Proportion of dataset to be used for evaluation
flags.DEFINE_float('validation_size', 0.3, 'Float: The proportion of examples in the dataset to be used for validation')

# The number of shards to split the dataset into.
flags.DEFINE_integer('num_shards', 2, 'Int: Number of shards to split the TFRecord files into')

# Seed for repeatability.
flags.DEFINE_integer('random_seed', 123, 'Int: Random seed to use for repeatability.')

#Output filename for the naming the TFRecord file
flags.DEFINE_string('tfrecord_filename', 'TFRecord', 'String: The output filename to name your TFRecord file')

FLAGS = flags.FLAGS


# converting the raw image data to TFRecord files
def tf_record_constructor():
	# =============CHECKS==============
	# Check if there is a tfrecord_filename entered
	if not FLAGS.tfrecord_filename:
		raise ValueError('tfrecord_filename is empty. Please state a tfrecord_filename argument.')

	# Check if there is a dataset directory entered
	if not FLAGS.dataset_dir:
		raise ValueError('dataset_dir is empty. Please state a dataset_dir argument.')

	# If the TFRecord files already exist in the directory, then exit without creating the files again
	if _dataset_exists(dataset_dir=FLAGS.dataset_dir, _NUM_SHARDS=FLAGS.num_shards,
					   output_filename=FLAGS.tfrecord_filename):
		print('Dataset files already exist. Exiting without re-creating them.')
		return None
	# ==========END OF CHECKS============

	# Get a list of photo_filenames like ['123.png', '456.png'...] and a list of sorted class names from parsing the subdirectories.
	pic_names, class_names = _get_filenames_and_classes(FLAGS.dataset_dir)
	class_name_to_ids = dict(zip(class_names, range(len(class_names))))

	# Find the number of validation examples we need
	num_validation = int(FLAGS.validation_size * len(pic_names))

	# Divide the training dataset into train and test
	random.seed(FLAGS.random_seed)
	random.shuffle(pic_names)
	training_filenames = pic_names[num_validation:]
	validation_filenames = pic_names[:num_validation]

	_convert_dataset('train', training_filenames, class_name_to_ids,
					 dataset_dir= FLAGS.dataset_dir, tfrecord_filename=FLAGS.tfrecord_filename,
					 _NUM_SHARDS=FLAGS.num_shards)

	_convert_dataset('validation', validation_filenames, class_name_to_ids,
					 dataset_dir=FLAGS.dataset_dir, tfrecord_filename=FLAGS.tfrecord_filename,
					 _NUM_SHARDS=FLAGS.num_shards)


	# write the labels file
	lables_to_class_names = dict(zip(range(len(class_names)), class_names))
	write_label_file(lables_to_class_names, FLAGS.dataset_dir)
	print('\nFinished converting the %s dataset!' % FLAGS.tfrecord_filename)


def neural_network():
	filenames = ["/Chars/Chars_train_1.tfrecord", "Chars/Chars_train_2.tfrecord",
				 "/Chars/Chars_train_1.tfrecord", "Chars/Chars_train_2.tfrecord"]
	training_filenames = ["/Chars/Chars_train_1.tfrecord", "Chars/Chars_train_2.tfrecord"]
	validation_filenames = ["/Chars/Chars_train_1.tfrecord", "Chars/Chars_train_2.tfrecord"]
	kanji_training = tf.data.TFRecordDataset(training_filenames)
	kanji_validation = tf.data.TFRecordDataset(validation_filenames)

	dataset = tf.data.TFRecordDataset(filenames)
	dataset = dataset.repeat(10)

	dataset = dataset.map()


	# classifier = Sequential()
	# classifier.add(Conv2D(32, (3, 3), input_shape=(64, 64, 3), activation='relu'))
	# classifier.add(MaxPooling2D(pool_size=(2, 2)))
	# classifier.add(Flatten())
	# classifier.add(Dense(units=128, activation='relu'))
	# classifier.add(Dense(units=50, activation='relu'))
	#
	# classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
	#
	# train_datagen = ImageDataGenerator(rescale=1./255)
	# shear_range = 0.2,
	# zoom_range = 0.2,
	# horizontal_flip = True
	# validation_datagen = ImageDataGenerator(rescale=1./255)
	# training_set = train_datagen.flow_from_directory('Chars/Kanji_chars', target_size=(64, 64), batch_size=32, class_mode='binary')
	# validation_set = validation_datagen.flow_from_directory('Chars/Kanji_chars', target_size=(64, 64), batch_size=32, class_mode='binary')
	#
	# classifier.fit_generator(training_set, steps_per_epoch=5000, epochs=10, validation_data=validation_set, validation_steps=1000)


if __name__ == "__main__":
	tf_record_constructor()
	neural_network()






