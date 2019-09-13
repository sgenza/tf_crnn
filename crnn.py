import numpy as np
import tensorflow as tf
from tensorflow.contrib import rnn
from utils import ground_truth_to_word, batch_ground_truth_to_word
from data import synth_word_generator, svt_word_generator, IIIT5K_word_generator
import config

save_path = '.' 
train_len = 7224612
train_epoch_len = train_len // config.batch_size
iteration_count = train_epoch_len * 5

inputs = tf.placeholder(tf.float32, [config.batch_size, config.img_h, config.img_w, 1], name='inputs')
targets = tf.sparse_placeholder(tf.int32, name='targets')
seq_len = tf.placeholder(tf.int32, [None], name='seq_len')

# Feature extraction
conv1 = tf.layers.conv2d(inputs=inputs, filters = 64, kernel_size = (3, 3), padding = "same", activation=tf.nn.relu, name='conv_1')
pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2, name='pool_1')
conv2 = tf.layers.conv2d(inputs=pool1, filters = 128, kernel_size = (3, 3), padding = "same", activation=tf.nn.relu, name='conv_2')
pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2, name='pool_2')
conv3 = tf.layers.conv2d(inputs=pool2, filters = 256, kernel_size = (3, 3), padding = "same", activation=tf.nn.relu, name='conv_3')

bnorm1 = tf.layers.batch_normalization(conv3, name='bn_1')
conv4 = tf.layers.conv2d(inputs=bnorm1, filters = 256, kernel_size = (3, 3), padding = "same", activation=tf.nn.relu, name='conv_4')
pool3 = tf.layers.max_pooling2d(inputs=conv4, pool_size=[2, 2], strides=[1, 2], padding="same", name='pool_3')
conv5 = tf.layers.conv2d(inputs=pool3, filters = 512, kernel_size = (3, 3), padding = "same", activation=tf.nn.relu, name='conv_5')

bnorm2 = tf.layers.batch_normalization(conv5, name='bn_2')
conv6 = tf.layers.conv2d(inputs=bnorm2, filters = 512, kernel_size = (3, 3), padding = "same", activation=tf.nn.relu, name='conv_6')
pool4 = tf.layers.max_pooling2d(inputs=conv6, pool_size=[2, 2], strides=[1, 2], padding="same", name='pool_4')
conv7 = tf.layers.conv2d(inputs=pool4, filters = 512, kernel_size = (2, 2), padding = "valid", activation=tf.nn.relu, name='conv_7')

# Reshape for LSTM and max char count
reshaped_cnn_output = tf.reshape(conv7, [config.batch_size, -1, 512], name='reshaped_cnn')
max_char_count = reshaped_cnn_output.get_shape().as_list()[1]
max_char_count_name = tf.constant(max_char_count, name='max_char_count')
print('='*20, max_char_count)

with tf.variable_scope(None, default_name="bidirectional-rnn-1"):

	# Forward
	lstm_fw_cell_1 = rnn.BasicLSTMCell(256)
	# Backward
	lstm_bw_cell_1 = rnn.BasicLSTMCell(256)
	# First bidirectional LSTM
	inter_output, _ = tf.nn.bidirectional_dynamic_rnn(lstm_fw_cell_1, lstm_bw_cell_1, reshaped_cnn_output, seq_len, dtype=tf.float32)
	inter_output = tf.concat(inter_output, 2)

with tf.variable_scope(None, default_name="bidirectional-rnn-2"):

	# Forward
	lstm_fw_cell_2 = rnn.BasicLSTMCell(256)
	# Backward
	lstm_bw_cell_2 = rnn.BasicLSTMCell(256)
	# Second bidirectional LSTM
	lstm_outputs, _ = tf.nn.bidirectional_dynamic_rnn(lstm_fw_cell_2, lstm_bw_cell_2, inter_output, seq_len, dtype=tf.float32)
	lstm_outputs = tf.concat(lstm_outputs, 2)

logits = tf.reshape(lstm_outputs, [-1, 512], name='reshaped_lstm')

W = tf.Variable(tf.truncated_normal([512, config.num_classes], stddev=0.1), name="W")
b = tf.Variable(tf.constant(0., shape=[config.num_classes]), name="b")

logits = tf.matmul(logits, W) + b
logits = tf.reshape(logits, [config.batch_size, -1, config.num_classes], name='logits')

# Final layer, the output of the BLSTM
logits = tf.transpose(logits, (1, 0, 2), name='transpose_logits')

# Loss and cost calculation
loss = tf.nn.ctc_loss(targets, logits, seq_len)
cost = tf.reduce_mean(loss, name='cost')

optimizer = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(cost)

decoded, log_prob = tf.nn.ctc_beam_search_decoder(logits, seq_len, merge_repeated=False)
dense_decoded = tf.sparse_tensor_to_dense(decoded[0], default_value=-1, name='dense_decoded')

acc = tf.reduce_mean(tf.edit_distance(tf.cast(decoded[0], tf.int32), targets), name='accuracy')

with tf.Session() as sess:

	init = tf.global_variables_initializer()
	sess.run(init)

	synth_train_gen = synth_word_generator('90kDICT32px/', 'train')
	synth_val_gen = synth_word_generator('90kDICT32px/', 'val')
	svt_gen = svt_word_generator(path='SVT/', mode='test')
	IIIT5K_gen = IIIT5K_word_generator(path='IIIT5K/', mode='test')

	train_loss_hist = []
	val_loss_hist = []
	train_acc_hist = []
	val_acc_hist = []
	svt_loss_hist = []
	svt_acc_hist = []
	IIIT5K_loss_hist = []
	IIIT5K_acc_hist = []
	train_wer_hist = []
	val_wer_hist = []
	svt_wer_hist = []
	IIIT5K_wer_hist = []

	# Train
	for i in np.arange(iteration_count):

		x_batch, y_batch, dt_batch = synth_train_gen.__next__()
		op, decoded_value, loss_value, acc_value = sess.run([optimizer, dense_decoded, cost, acc], feed_dict={inputs: x_batch, targets: dt_batch, seq_len: [max_char_count] * config.batch_size})
		
		if i % 50 == 0:

			print('True synth train:\t{}'.format(y_batch[0]))
			print('Pred synth train:\t{}'.format(ground_truth_to_word(decoded_value[0])))
			print('[{}] Synth 90k iteration. Train loss: {}\tTrain accuracy: {}'.format(i, loss_value, acc_value))
			train_wer = np.sum(np.equal(y_batch, batch_ground_truth_to_word(decoded_value)))/config.batch_size
			print('Train WER:\t{}'.format(train_wer))

			train_wer_hist.append(train_wer)
			train_loss_hist.append(loss_value)
			train_acc_hist.append(acc_value)

			x_batch_val, y_batch_val, dt_batch_val = synth_val_gen.__next__()
			val_decoded_value, val_loss_value, val_acc_value = sess.run([dense_decoded, cost, acc], feed_dict={inputs: x_batch_val, targets: dt_batch_val, seq_len: [max_char_count] * config.batch_size})
			
			print('True synth val:\t{}'.format(y_batch_val[0]))
			print('Pred synth val:\t{}'.format(ground_truth_to_word(val_decoded_value[0])))
			print('[{}] Synth 90k iteration. Val loss: {}\tVal accuracy: {}'.format(i, val_loss_value, val_acc_value))
			val_wer = np.sum(np.equal(y_batch_val, batch_ground_truth_to_word(val_decoded_value)))/config.batch_size
			print('Val WER:\t{}'.format(val_wer))

			val_wer_hist.append(val_wer)
			val_loss_hist.append(val_loss_value)
			val_acc_hist.append(val_acc_value)

			x_batch_svt, y_batch_svt, dt_batch_svt = svt_gen.__next__()
			svt_decoded_value, svt_loss_value, svt_acc_value = sess.run([dense_decoded, cost, acc], feed_dict={inputs: x_batch_svt, targets: dt_batch_svt, seq_len: [max_char_count] * config.batch_size})

			print('True svt:\t{}'.format(y_batch_svt[0]))
			print('Pred svt:\t{}'.format(ground_truth_to_word(svt_decoded_value[0])))
			print('[{}] SVT iteration. Test loss: {}\tTest accuracy: {}'.format(i, svt_loss_value, svt_acc_value))
			svt_wer = np.sum(np.equal(y_batch_svt, batch_ground_truth_to_word(svt_decoded_value)))/config.batch_size
			print('svt WER:\t{}'.format(svt_wer))

			svt_wer_hist.append(svt_wer)
			svt_loss_hist.append(svt_loss_value)
			svt_acc_hist.append(svt_acc_value)

			x_batch_IIIT5K, y_batch_IIIT5K, dt_batch_IIIT5K = IIIT5K_gen.__next__()
			IIIT5K_decoded_value, IIIT5K_loss_value, IIIT5K_acc_value = sess.run([dense_decoded, cost, acc], feed_dict={inputs: x_batch_IIIT5K, targets: dt_batch_IIIT5K, seq_len: [max_char_count] * config.batch_size})

			print('True IIIT5K:\t{}'.format(y_batch_IIIT5K[0]))
			print('Pred IIIT5K:\t{}'.format(ground_truth_to_word(IIIT5K_decoded_value[0])))
			print('[{}] IIIT5K iteration. Test loss: {}\tTest accuracy: {}'.format(i, IIIT5K_loss_value, IIIT5K_acc_value))
			IIIT5K_wer = np.sum(np.equal(y_batch_IIIT5K, batch_ground_truth_to_word(IIIT5K_decoded_value)))/config.batch_size
			print('IIIT5K WER:\t{}'.format(IIIT5K_wer))

			IIIT5K_wer_hist.append(IIIT5K_wer)
			IIIT5K_loss_hist.append(IIIT5K_loss_value)
			IIIT5K_acc_hist.append(IIIT5K_acc_value)

		if i % 5000 == 0:

			np.save('train_loss_hist.npy', np.array(train_loss_hist))
			np.save('val_loss_hist.npy', np.array(val_loss_hist))
			np.save('train_acc_hist.npy', np.array(train_acc_hist))
			np.save('val_acc_hist.npy', np.array(val_acc_hist))
			np.save('svt_loss_hist.npy', np.array(svt_loss_hist))
			np.save('svt_acc_hist.npy', np.array(svt_acc_hist))
			np.save('IIIT5K_loss_hist.npy', np.array(IIIT5K_loss_hist))
			np.save('IIIT5K_acc_hist.npy', np.array(IIIT5K_acc_hist))
			np.save('train_wer_hist.npy', np.array(train_wer_hist))
			np.save('val_wer_hist.npy', np.array(val_wer_hist))
			np.save('svt_wer_hist.npy', np.array(svt_wer_hist))
			np.save('IIIT5K_wer_hist.npy', np.array(IIIT5K_wer_hist))

			saver = tf.train.Saver(tf.global_variables(), max_to_keep=10)
			saver.save(sess, save_path)

	# # Save model
	# saver = tf.train.Saver(tf.global_variables(), max_to_keep=10)
	# saver.save(sess, save_path)

	# # Test
	# x_test = None
	# y_test = None
 
	# decoded_value = sess.run(decoded, feed_dict={inputs: x_test, seq_len: max_char_count * config.batch_size})

	# print('True:\t{}'.format(y_batch[i]))
	# print('Output:\t{}'.format(ground_truth_to_word(decoded_value[j])))
