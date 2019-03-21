import flask
import pandas as pd
import tensorflow as tf
import keras
from keras.models import load_model

app = flask.Flask(__name__)

def auc(y_true, y_pred):
	auc = tf.metrics.auc(y_true, y_pred)[1]
	keras.backend.get_session().run(tf.local_variables_initializer())
	return auc

# load the model
global graph
graph = tf.get_default_graph()
model = load_model('kanjibot_hiragana_model.h5', custom_objects={'auc': auc})

@app.route("/predict", methods=["GET", "POST"])
def predict():
	data = {"Success":False}

	# get request parameters
	params = flask.request.json
	if (params == None):
		params = flask.request.args

	# if no parameters are found, echo msg parameter
	if (params != None):
		data["response"] = params.get("msg")
		data["succes"] = True

	# return json response
	return flask.jsonify(data)

# start the app
app.run(host='localhost')
