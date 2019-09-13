import numpy as np
import tensorflow as tf
from tensorflow.contrib import rnn

import config
from utils import ground_truth_to_word, batch_ground_truth_to_word
from data import DataGenerator


class CRNN(object):

	def __init__(self, lstm_units, num_classes):

		self.lstm_units = lstm_units
		self.num_classes = num_classes

	def feature_extraction(self, x, name_scope):
		
		# CNN feature extraction.
		with tf.variable_scope(name_scope):

			conv_1 = tf.layers.conv2d(inputs=x, filters = 64, kernel_size = (3, 3), 
										padding = "same", activation=tf.nn.relu, name='conv_1')
			pool_1 = tf.layers.max_pooling2d(inputs=conv_1, pool_size=[2, 2], strides=2, name='pool_1')
			conv_2 = tf.layers.conv2d(inputs=pool_1, filters = 128, kernel_size = (3, 3), 
										padding = "same", activation=tf.nn.relu, name='conv_2')
			pool_2 = tf.layers.max_pooling2d(inputs=conv_2, pool_size=[2, 2], strides=2, name='pool_2')
			conv_3 = tf.layers.conv2d(inputs=pool_2, filters = 256, kernel_size = (3, 3), 
										padding = "same", activation=tf.nn.relu, name='conv_3')
			bn_1 = tf.layers.batch_normalization(conv_3, name='bn_1')
			conv_4 = tf.layers.conv2d(inputs=bn_1, filters = 256, kernel_size = (3, 3), 
										padding = "same", activation=tf.nn.relu, name='conv_4')
			pool_3 = tf.layers.max_pooling2d(inputs=conv_4, pool_size=[2, 2], 
										strides=[1, 2], padding="same", name='pool_3')
			conv_5 = tf.layers.conv2d(inputs=pool_3, filters = 512, kernel_size = (3, 3), 
										padding = "same", activation=tf.nn.relu, name='conv_5')
			bn_2 = tf.layers.batch_normalization(conv_5, name='bn_2')
			conv_6 = tf.layers.conv2d(inputs=bn_2, filters = 512, kernel_size = (3, 3), 
										padding = "same", activation=tf.nn.relu, name='conv_6')
			pool_4 = tf.layers.max_pooling2d(inputs=conv_6, pool_size=[2, 2], 
										strides=[1, 2], padding="same", name='pool_4')
			conv_7 = tf.layers.conv2d(inputs=pool_4, filters = 512, kernel_size = (2, 2), 
										padding = "valid", activation=tf.nn.relu, name='conv_7')
		return conv_7

	def map_to_sequence(self, x, name_scope):

		# From feature map to sequence
		with tf.variable_scope(name_scope):

			x_shape = x.get_shape().as_list()
			output = tf.reshape(x, [-1, x_shape[1] * x_shape[2], x_shape[3]], name='reshaped_cnn')

			# Max char count for seq_len
			self.max_char_count = output.get_shape().as_list()[1]

		return output

	def bidirectional_LSTM(self, x, name_scope):

		with tf.variable_scope(name_scope, reuse=tf.AUTO_REUSE):

			# Forward
			lstm_fw_cell = rnn.BasicLSTMCell(self.lstm_units)
			# Backward
			lstm_bw_cell = rnn.BasicLSTMCell(self.lstm_units)

			# Create bidirectional LSTM
			output, _ = tf.nn.bidirectional_dynamic_rnn(lstm_fw_cell, lstm_bw_cell, x, self.seq_len, dtype=tf.float32)
			# Stack forward and backward
			output = tf.concat(output, 2)

		return output

	def sequence_labeling(self, x, name_scope='sequence_labeling'):

		with tf.variable_scope(name_scope):

			# Bidirectional LSTM layers
			bi_lstm_1 = self.bidirectional_LSTM(x, 'bi_lstm_1')
			bi_lstm_2 = self.bidirectional_LSTM(bi_lstm_1, 'bi_lstm_2')

			logits = tf.reshape(bi_lstm_2, [-1, self.lstm_units * 2], name='reshaped_lstm')

			# Fully-connected layer
			w = tf.get_variable('w', shape=[self.lstm_units * 2, self.num_classes],
								initializer=tf.truncated_normal_initializer(stddev=0.1), trainable=True)
			b = tf.get_variable('b', shape=[self.num_classes],
								initializer=tf.zeros_initializer(), trainable=True)
			logits = tf.matmul(logits, w) + b
			logits = tf.reshape(logits, [tf.shape(x)[0], -1, config.num_classes], name='logits')

			# Model output
			output = tf.transpose(logits, (1, 0, 2), name='transpose_logits')

		return output

	def inference(self, name_scope, reuse=False):

		with tf.variable_scope(name_scope, reuse=reuse):

			# Placeholders
			self.inputs = tf.placeholder(tf.float32, [None, config.img_h, config.img_w, 1], name='inputs')
			self.targets = tf.sparse_placeholder(tf.int32, name='targets')
			self.seq_len = tf.placeholder(tf.int32, [None], name='seq_len')

			cnn_output = self.feature_extraction(self.inputs, 'cnn_output')
			sequence = self.map_to_sequence(cnn_output, 'map_to_sequence')
			net_output = self.sequence_labeling(sequence, 'sequence_labeling')

		return net_output

	def loss(self):

		# Compute CTC-loss
		self.net_output = self.inference('inference')
		self.ctc_loss = tf.reduce_mean(tf.nn.ctc_loss(labels=self.targets, inputs=self.net_output, 
			                                 sequence_length=self.seq_len), name='loss')

	def optimize(self):

		adam = tf.train.AdamOptimizer(learning_rate=1e-4)
		self.loss_minimize = adam.minimize(self.ctc_loss)

	def get_distance(self):

		# Compute Levenshtein distance
		decoded, log_prob = tf.nn.ctc_beam_search_decoder(self.net_output, self.seq_len, merge_repeated=False)
		dense_decoded = tf.sparse_tensor_to_dense(decoded[0], default_value=-1, name='dense_decoded')
		self.distance = tf.reduce_mean(tf.edit_distance(tf.cast(decoded[0], tf.int32), self.targets), name='distance')

	def train(self, epochs):

		# Model compile
		self.loss()
		self.optimize()
		self.get_distance()

		with tf.Session() as sess:

			synth_train_gen = iter(DataGenerator(config.batch_size, 'Synth90k', 'train'))
			synth_val_gen = iter(DataGenerator(config.batch_size, 'Synth90k', 'val'))

			train_len = config.synth_len
			train_epoch_len = train_len // config.batch_size
			iterations = train_epoch_len * epochs

			init_g, init_l = tf.global_variables_initializer(), tf.local_variables_initializer()
			sess.run([init_g, init_l])

			for i in np.arange(iterations):

				# Train
				x_batch, y_batch, dt_batch = next(synth_train_gen)
				feed_dict = {self.inputs: x_batch, self.targets: dt_batch, self.seq_len: [self.max_char_count] * config.batch_size}
				train_loss, train_distance, _ = sess.run([self.ctc_loss, self.distance, self.loss_minimize], feed_dict=feed_dict)

				if i % 50 == 0:

					# Validation
					x_batch_val, y_batch_val, dt_batch_val = next(synth_val_gen)
					feed_dict = {self.inputs: x_batch_val, self.targets: dt_batch_val, self.seq_len: [self.max_char_count] * config.batch_size}
					val_loss, val_distance = sess.run([self.ctc_loss, self.distance], feed_dict=feed_dict)
					print('Iteration {}'.format(i))
					print('Train loss: {:3f} Val loss: {:3f}'.format(float(train_loss), float(val_loss)))
					print('Train distance: {:3f} Val distance: {:3f}'.format(float(train_distance), float(val_distance)))

			saver = tf.train.Saver(max_to_keep=10)
			saver.save(sess, 'model')

	def predict(self, x_batch):

		# Load model
		saver = tf.train.import_meta_graph('model.meta')
		saver.restore(sess, tf.train.latest_checkpoint('./'))
		graph = tf.get_default_graph()

		# Get tensors
		inputs = graph.get_tensor_by_name('inference/inputs:0')
		seq_len = graph.get_tensor_by_name('inference/seq_len:0')
		max_char_count = graph.get_tensor_by_name('inference/map_to_sequence/reshaped_cnn:0').get_shape().as_list()[1]
		dense_decoded = graph.get_tensor_by_name('dense_decoded:0')

		feed_dict = {inputs: x_batch, seq_len: [max_char_count] * x_batch.shape[0]}
		decoded_value = sess.run(dense_decoded, feed_dict=feed_dict)

		return batch_ground_truth_to_word(decoded_value, x_batch.shape[0])

	def check_accuracy(self, dataset):

		if dataset not in {'SVT', 'IIIT5K'}:
			raise ValueError('Invalid dataset name "{}"'.format(dataset))

		with tf.Session() as sess:

			# Load model
			saver = tf.train.import_meta_graph('model.meta')
			saver.restore(sess, tf.train.latest_checkpoint('./'))

			graph = tf.get_default_graph()

			# Get tensors
			inputs = graph.get_tensor_by_name('inference/inputs:0')
			targets_shape = graph.get_tensor_by_name('inference/targets/shape:0')
			targets_values = graph.get_tensor_by_name('inference/targets/values:0')
			targets_indices = graph.get_tensor_by_name('inference/targets/indices:0')
			targets = tf.SparseTensor(targets_indices, targets_values, targets_shape)
			seq_len = graph.get_tensor_by_name('inference/seq_len:0')
			max_char_count = graph.get_tensor_by_name('inference/map_to_sequence/reshaped_cnn:0').get_shape().as_list()[1]
			distance = graph.get_tensor_by_name('distance:0')
			dense_decoded = graph.get_tensor_by_name('dense_decoded:0')

			if dataset == 'SVT':
				test_gen = iter(DataGenerator(config.batch_size, 'SVT', 'test'))
				test_len = config.svt_test_len
			elif dataset == 'IIIT5K':
				test_gen = iter(DataGenerator(config.batch_size, 'IIIT5K', 'test'))
				test_len = config.IIIT5K_test_len

			pred_words, true_words, distances = [], [], []
			for i in np.arange(test_len):

				x_batch, y_batch, tg_batch = next(test_gen)
				if x_batch.ndim == 3:
					x_batch = np.expand_dims(x_batch, axis=0)
				feed_dict = {inputs: x_batch, targets: tg_batch, seq_len: [max_char_count] * x_batch.shape[0]}
				distance_value, decoded_value = sess.run([distance, dense_decoded], feed_dict=feed_dict)

				pred_words.append(batch_ground_truth_to_word(decoded_value, x_batch.shape[0])[0])
				true_words.append(y_batch), distances.append(distance_value)

		pred_words, true_words = np.array(pred_words, dtype=np.object), np.array(true_words, dtype=np.object)
		average_distance, average_acc = np.mean(distances), np.mean(pred_words == true_words)
		print('Average Levenshtein distance:\t{}'.format(average_distance))
		print('Average accuracy:\t{}'.format(average_acc))

if __name__ == '__main__':

	# Usage example
	with tf.Graph().as_default():

		model = CRNN(lstm_units=config.lstm_units, num_classes=config.num_classes)
		model.train(epochs=5)
		model.check_accuracy('SVT')
		model.check_accuracy('IIIT5K')
