from random import shuffle
import glob
import os
import cv2
import numpy as np
import tensorflow as tf
import sys
from retrying import retry
import time
from sklearn.model_selection import train_test_split


class KanjibotImg2TFrecord:

	def __init__(self):
		pass

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
		while True:
			try:
				label_list.append(addres.split('\\')[3])
				for item in labels:
					if label_list[i] == item:
						index_labels.append(labels.index(item))
				i += 1
			except IndexError:
				print(IndexError, "Out of range because of an uneven split.")
				continue
			break


	def image_list(self, addr_list):
		list = []
		for addr in addr_list:
			load = self.load_image(addr)
			list.append(load)
		out = np.array(list)
		return out

	# # Training: 70%
	# # Validation: 15%
	# # Test: 15%
	# (train_addrs, test_addrs, train_labels, test_labels) = train_test_split(addrs, index_labels, test_size=0.3)
	# (val_addrs, test_addrs, val_labels, test_labels) = train_test_split(test_addrs, test_labels, test_size=0.5)
	def data_split(self):
		(train_addrs, test_addrs, train_labels, test_labels) = train_test_split(self.addrs, self.index_labels, test_size=0.3)
		(val_addrs, test_addrs, val_labels, test_labels) = train_test_split(test_addrs, test_labels, test_size=0.5)

		(train_image_list, test_image_list, train_image_labels, test_image_labels) = train_test_split(self.image_list(self.addrs), self.index_labels, test_size=0.3)
		(val_image_list, test_image_list, val_image_labels, test_image_labels) = train_test_split(test_image_list, test_image_labels, test_size=0.5)

		return [train_image_list, test_image_list, train_image_labels, test_image_labels,
				val_image_list, val_image_labels, val_addrs, test_addrs, val_labels, test_labels, train_addrs]

	def load_image(self, addr):
		addr = str(addr)
		kernel = np.ones((2, 2), np.uint8)
		img = cv2.imread(addr)
		img = cv2.resize(img, (64, 64), interpolation=cv2.INTER_AREA)
		img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
		# img = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
		# img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
		# img = cv2.erode(img, kernel, iterations=1)
		img = img.astype(np.int32)
		return img

	def _int64_feature(self, value):
		return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

	def _bytes_feature(self, value):
		return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


# Training set
	def write_files(self):
		train_filename = 'TFRecord\\kanji_train.tfrecord'
		writer = tf.python_io.TFRecordWriter(train_filename)

		for i in range(len(self.train_addrs)):
			if not i % 1000:
				print('Train data: {}/{}'.format(i, len(self.train_addrs)))
				sys.stdout.flush()
			img = self.load_image(self.train_addrs[i])
			label = self.train_labels[i]

			# create a feature
			feature = {'train/label': self._int64_feature(label),
					   'train/image': self._bytes_feature(tf.compat.as_bytes(img.tostring()))}

			# create an example protocol buffer
			example = tf.train.Example(features=tf.train.Features(feature=feature))

			# serialize to string and write to file
			writer.write(example.SerializeToString())

		writer.close()
		sys.stdout.flush()

		# Validation set
		val_filename = 'TFRecord\\kanji_val.tfrecord'
		writer = tf.python_io.TFRecordWriter(val_filename)

		for i in range(len(self.val_addrs)):
			if not i % 1000:
				print('Val data: {}/{}'.format(i, len(self.val_addrs)))
				sys.stdout.flush()

			img = self.load_image(self.val_addrs[i])
			label = self.val_labels[i]

			feature = {'val/label': self._int64_feature(label),
					   'val/image': self._bytes_feature(tf.compat.as_bytes(img.tostring()))}

			example = tf.train.Example(features=tf.train.Features(feature=feature))
			writer.write(example.SerializeToString())

		writer.close()
		sys.stdout.flush()

		# Test set
		test_filename = 'TFRecord\\kanji_test.tfrecord'
		writer = tf.python_io.TFRecordWriter(test_filename)

		for i in range(len(self.test_addrs)):
			if not i % 1000:
				print('Test data: {}/{}'.format(i, len(self.test_addrs)))
				sys.stdout.flush()

			img = self.load_image(self.test_addrs[i])
			label = self.test_labels[i]
			feature = {'test/label': self._int64_feature(label),
					   'test/image': self._bytes_feature(tf.compat.as_bytes(img.tostring()))}

			example = tf.train.Example(features=tf.train.Features(feature=feature))
			writer.write(example.SerializeToString())

		writer.close()
		sys.stdout.flush()


# KanjibotImg2TFrecord().write_files()