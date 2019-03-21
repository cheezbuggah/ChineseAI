import tensorflow as tf
from flask import Flask, render_template, request, jsonify
from keras.models import model_from_json, load_model
from scipy.misc import imread, imresize
import numpy as np
import pandas
import time
import keras
import base64
import sys
import re
import os

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

app = Flask(__name__)
# Define global variables for easy usage
global model, graph
# Initialize them
model, graph = init()




@app.route('/process', methods=['POST'])
def process():
	imgdata = request.form['imgBase64']
	if imgdata != None:
		imgstr = re.search(r'base64,(.*)', str(imgdata)).group(1)
		with open(UPLOAD+'output.png', 'wb') as output:
			output.write(base64.b64decode(imgstr))
		predict(UPLOAD+'output.png')
	return jsonify({})

@app.route('/')
def index():
	return render_template('form.html')


def predict(url):
	img_data = url
	x = imread(img_data, mode='L')
	x = imresize(x, (28, 28))
	x = x.reshape((1, 28, 28, 1))

	with graph.as_default():
		out = model.predict(x)
		print(out)
		response = np.argmax(out, axis=1)
		print(response)
		return str(response[0])


if __name__ == '__main__':
	app.run(debug=True)