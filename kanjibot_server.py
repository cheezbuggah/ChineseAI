import tensorflow as tf

tf.keras.backend.set_learning_phase(0)
model = tf.keras.models.load_model('./kanjibot_hiragana_model.h5')
save_path = './Classifier/1'

with tf.keras.backend.get_session() as sess:
	tf.saved_model.simple_save(
		sess,
		save_path,
		inputs={'input_image': model.input},
		outputs={t.name: t for t in model.outputs})