import tensorflow as tf
import numpy as np
import pandas as pd
import time
import LRFinder
import glob
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras import optimizers
from keras.applications import ResNet50
from keras.layers import Dense, Dropout, Activation, Flatten, Convolution2D, MaxPooling2D
from keras.utils import np_utils
from keras.utils.vis_utils import plot_model
from keras.callbacks import CSVLogger, LearningRateScheduler, History, TensorBoard
from sklearn.model_selection import KFold

np.random.seed(123)

def get_files(npz):
	for file in npz:
		npz = npz.get(file)
		return npz


x_train = np.load("./Dataset/k49-train-imgs.npz", mmap_mode='r')
x_train = get_files(x_train)

y_train = np.load("./Dataset/k49-train-labels.npz")
y_train = get_files(y_train)

x_test = np.load("./Dataset/k49-test-imgs.npz", mmap_mode='r')
x_test = get_files(x_test)

y_test = np.load("./Dataset/k49-test-labels.npz")
y_test = get_files(y_test)

class_labels_k49 = pd.DataFrame.from_csv("./Dataset/k49_classmap.csv")

print(class_labels_k49)

x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255

print(x_train.shape)
print(x_test.shape)

y_train = np_utils.to_categorical(y_train, 49)
y_test = np_utils.to_categorical(y_test, 49)

print(y_train.shape)
print(y_test.shape)

# Define the model
model = Sequential()
model.add(Convolution2D(32, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 1)))
model.add(Convolution2D(32, 3, 3, activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(49, activation='softmax'))
print(model.output_shape)
model.summary()

batch_size = 32
epochs = 30
learning_rate = 1e-5
decay_rate = learning_rate / epochs
adam = optimizers.Adam(lr=learning_rate, decay=decay_rate)
tensorboard = TensorBoard(log_dir='./logs', histogram_freq=10, batch_size=batch_size,
						  write_graph=True, write_grads=True, write_images=True)

lr_finder = LRFinder.LRFinder(min_lr=1e-5, max_lr=1e-2, steps_per_epoch=(epochs/batch_size), epochs=3)

csv_logger = CSVLogger('Warmind_Nobunaga_log.csv', append=True, separator=',')
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
history = model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, verbose=1,
					validation_split=0.1, shuffle=True, callbacks=[tensorboard])

# Summarize history for accuracy
print(history.history.keys())
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

scores = model.evaluate(x_test, y_test)
print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
print(model.summary())
plot_model(model, to_file='Warmind_Nobunaga.png', show_shapes=True, show_layer_names=True)

model_json = model.to_json()
with open("kanjibot_hiragana_model.json", "w") as json_file:
	json_file.write(model_json)

model.save_weights('kanjibot_hiragana_final.h5')
print("Saved model as .json and weights as .h5")


