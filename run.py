import tensorflow as tf
from flask import Flask, render_template, request, jsonify
from keras.models import model_from_json, load_model
from scipy.misc import imread, imresize
import math
import numpy as np
import decimal
import pandas
import matplotlib
matplotlib.use('Agg')
# import matplotlib.pyplot as plt
import pandas as pd
import time
import keras
import cv2 as cv
import base64
import sys
import re
import os
import random

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"

def init():
	# load the model JSON
	json_file = open('./hiragana_nn/kanjibot_hiragana_model.json')
	loaded_model_json = json_file.read()
	json_file.close()
	loaded_model = model_from_json(loaded_model_json)

	# load the model weights
	loaded_model.load_weights("./hiragana_nn/kanjibot_hiragana_final.h5")
	print("loaded model + weights from disk")
	loaded_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
	graph = tf.get_default_graph()

	return loaded_model, graph

sys.path.append(os.path.abspath("./model"))

UPLOAD = "./upload/"
SAVE = "./saved/"
BLACK = [0, 0, 0]
class_labels_k49 = pd.read_csv("./Dataset/k49_classmap.csv").to_dict()
print(class_labels_k49)
app = Flask(__name__)

global model, graph
model, graph = init()


# def safesave(img, chr):
# 	rst = random.randint(100000000, 999999999)
# 	rst = str(rst)
# 	try:
# 		cv.imread(SAVE+rst+"_"+chr+".png", 0)
# 		safesave(img, chr)
# 		return
# 	except:
# 		cv.imwrite(SAVE+rst+"_"+chr+".png", img)
# 		return

def predict(url):
	img = cv.imread(url, 0)
	img = cv.bitwise_not(img)
	contours,_ = cv.findContours(img,cv.RETR_EXTERNAL,cv.CHAIN_APPROX_NONE)
	mx, my, mw, mh = 300, 300, 0, 0
	for cnt in contours:
		x, y, w, h = cv.boundingRect(cnt)
		if x < mx:
			mx = x
		if w+x > mw:
			mw = w+x
		if y < my:
			my = y
		if h+y > mh:
			mh = h+y
	cut = img[my:mh, mx:mw]
	cut = cv.copyMakeBorder(cut, 10, 10, 10, 10, cv.BORDER_CONSTANT, value=BLACK)
	# plt.subplot(141), plt.imshow(cut, 'gray')
	# plt.subplot(142), plt.imshow(img, 'gray')
	# plt.show()
	cut = imresize(cut, (28, 28))
	cut2 = cut.reshape((1, 28, 28, 1))
	cut2 = cut2.astype('float32')
	cut2 /= 255

	with graph.as_default():
		out = model.predict(cut2)
		print(out)
	return out



@app.route('/process', methods=['POST'])
def process():
	prediction = []
	imgdata = request.form['imgBase64']
	if imgdata != None:
		imgstr = re.search(r'base64,(.*)', str(imgdata)).group(1)
		with open(UPLOAD+'output.png', 'wb') as output:
			output.write(base64.b64decode(imgstr))
		prediction = predict(UPLOAD+'output.png')
	return jsonify({'rarray':prediction.tolist()})

@app.route('/')
def index():
	return render_template('form.html')

if __name__ == '__main__':
	app.run()