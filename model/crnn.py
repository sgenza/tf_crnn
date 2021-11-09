import cv2
import numpy as np
import tensorflow as tf
from tensorflow.contrib import rnn
from itertools import chain
from packaging import version
from pathlib import Path
from tqdm import tqdm
from typing import List, Tuple, Union, Optional, Sequence, Mapping, Any

from .utils import arrays2labels, seed_everything, read_config
from .dataset import DataLoaderSynth90K, DataLoaderSVT, DataLoaderIIIT5K


__all__ = ['CRNN', 'DEFAULT_CONFIG_PATH']


TF_VERSION = version.parse(tf.__version__)
DEFAULT_CONFIG_PATH = Path(__file__).parent.parent.joinpath('config/default.yml')
DEFAULT_CKPT_DIR = Path(__file__).parent.joinpath('checkpoints')
Tensor = Union[tf.Tensor, tf.SparseTensor]


class CRNN(object):
    '''
    A class used to create Convolutional Recurrent Neural Network (CRNN) model.

    Original paper: 
        https://arxiv.org/pdf/1507.05717.pdf
    '''

    def __init__(self, 
                 cfg: Optional[Mapping[str, Mapping[str, Any]]] = None,
                 pretrained: bool = False):
        '''
        Initializes the CRNN.

        Args:
            cfg: a dict with configuration values.
            pretrained: using pretrained weights for the model or not.
        '''

        if cfg is None:
            cfg = read_config(DEFAULT_CONFIG_PATH)

        self.cfg = cfg
        self.pretrained = pretrained
        self.lstm_units = cfg['common']['lstm_units']
        self.img_w = cfg['common']['img_width']
        self.img_h = cfg['common']['img_height']
        self.alphabet = cfg['common']['alphabet']
        self.epochs = cfg['train']['epochs']
        self.lr = cfg['train']['learning_rate']
        self.train_bs = cfg['train']['batch_size']
        self.val_bs = cfg['eval']['batch_size']
        self.val_steps = cfg['common']['val_steps']
        self.val_interval = cfg['common']['val_interval']
        self.save_interval = cfg['common']['save_interval']
        self.ckpt_dir = cfg['common']['ckpt_dir'] if cfg['common']['ckpt_dir'] else DEFAULT_CKPT_DIR
        self.use_beam_search = cfg['common']['use_beam_search']
        self.seed = cfg['common']['seed']
        self.num_classes = len(self.alphabet) + 1

        if TF_VERSION >= version.parse('1.14.0'):
            tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
        else:
            tf.logging.set_verbosity(tf.logging.ERROR)

        seed_everything(self.seed)
        self._graph = tf.Graph()
        self._sess_cfg = tf.ConfigProto(allow_soft_placement=True)
        self._session = tf.Session(config=self._sess_cfg, graph=self._graph)
        self._load() if self.pretrained else self._build()

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(cfg, pretrained={self.pretrained})'

    def _load(self) -> None:

        with self._graph.as_default():

            saver = tf.train.import_meta_graph(str(self.ckpt_dir.joinpath('model.meta')))
            ckpt_path = tf.train.latest_checkpoint(str(self.ckpt_dir))
            saver.restore(self._session, ckpt_path)

    def _build_inputs(self) -> None:

        with tf.variable_scope('inputs'):

            self._inputs = tf.placeholder(tf.float32, [None, self.img_h, self.img_w, 1], name='inputs')
            self._targets = tf.sparse_placeholder(tf.int32, name='targets')
            self._seq_len = tf.placeholder(tf.int32, [None], name='seq_len')

    @staticmethod
    def _build_cnn(x: Tensor) -> Tensor:
        
        # CNN feature extraction
        with tf.variable_scope('cnn'):

            conv_1 = tf.layers.conv2d(inputs=x, filters=64, kernel_size=[3, 3],
                                      padding='same', activation=tf.nn.relu, name='conv_1')
            pool_1 = tf.layers.max_pooling2d(inputs=conv_1, pool_size=[2, 2], strides=2, name='pool_1')
            conv_2 = tf.layers.conv2d(inputs=pool_1, filters=128, kernel_size=[3, 3],
                                      padding='same', activation=tf.nn.relu, name='conv_2')
            pool_2 = tf.layers.max_pooling2d(inputs=conv_2, pool_size=[2, 2], strides=2, name='pool_2')
            conv_3 = tf.layers.conv2d(inputs=pool_2, filters=256, kernel_size=[3, 3],
                                      padding='same', activation=tf.nn.relu, name='conv_3')
            conv_4 = tf.layers.conv2d(inputs=conv_3, filters=256, kernel_size=[3, 3],
                                      padding='same', activation=tf.nn.relu, name='conv_4')
            pool_3 = tf.layers.max_pooling2d(inputs=conv_4, pool_size=[2, 2],
                                      strides=[1, 2], padding='same', name='pool_3')
            conv_5 = tf.layers.conv2d(inputs=pool_3, filters=512, kernel_size=[3, 3],
                                      padding='same', activation=tf.nn.relu, name='conv_5')
            bn_1 = tf.layers.batch_normalization(conv_5, name='bn_1')
            conv_6 = tf.layers.conv2d(inputs=bn_1, filters=512, kernel_size=[3, 3],
                                      padding='same', activation=tf.nn.relu, name='conv_6')
            bn_2 = tf.layers.batch_normalization(conv_6, name='bn_2')
            pool_4 = tf.layers.max_pooling2d(inputs=bn_2, pool_size=[2, 2],
                                             strides=[1, 2], padding='same', name='pool_4')
            conv_7 = tf.layers.conv2d(inputs=pool_4, filters=512, kernel_size=[2, 2],
                                      padding='valid', activation=tf.nn.relu, name='conv_7')
        
        return conv_7

    def _map_to_sequence(self, x: Tensor) -> Tensor:

        # From a feature map to a sequence
        with tf.variable_scope('map_to_sequence'):

            x_shape = x.get_shape().as_list()
            output = tf.reshape(x, (-1, x_shape[1] * x_shape[2], x_shape[3]), name='feature_sequence')

            # Get a maximum possible length of the sequence
            self.max_char_count = output.get_shape().as_list()[1]

        return output

    def _build_rnn(self, x: Tensor) -> Tensor:

        def build_bi_lstm(x: Tensor, name_scope: str) -> Tensor:

            with tf.variable_scope(name_scope, reuse=tf.AUTO_REUSE):

                # Forward
                lstm_fw_cell = rnn.BasicLSTMCell(self.lstm_units)

                # Backward
                lstm_bw_cell = rnn.BasicLSTMCell(self.lstm_units)

                # Create the bidirectional LSTM and concatenate both directions
                output, _ = tf.nn.bidirectional_dynamic_rnn(lstm_fw_cell, lstm_bw_cell, x, self._seq_len, dtype=tf.float32)
                output = tf.concat(output, 2)

            return output

        with tf.variable_scope('rnn'):

            # Bidirectional LSTM layers
            bi_lstm_1 = build_bi_lstm(x, 'bi-lstm_1')
            bi_lstm_2 = build_bi_lstm(bi_lstm_1, 'bi-lstm_2')

            logits = tf.reshape(bi_lstm_2, [-1, self.lstm_units * 2], name='rnn_logits')

        return logits

    def _build_model(self) -> None:

        # Build the end-to-end model
        with tf.variable_scope('model', reuse=False):

            cnn_features = self._build_cnn(self._inputs)
            feature_sequence = self._map_to_sequence(cnn_features)
            rnn_logits = self._build_rnn(feature_sequence)

            # Fully-connected layer
            w = tf.get_variable('w', shape=[self.lstm_units * 2, self.num_classes],
                                initializer=tf.truncated_normal_initializer(stddev=0.1), trainable=True)
            b = tf.get_variable('b', shape=[self.num_classes],
                                initializer=tf.zeros_initializer(), trainable=True)
            logits = tf.matmul(rnn_logits, w) + b
            logits = tf.reshape(logits, [tf.shape(feature_sequence)[0], -1, self.num_classes], name='logits')

            self._output = tf.transpose(logits, (1, 0, 2), name='output')

    def _build_loss(self) -> None:

        with tf.name_scope('loss'):
            self._ctc_loss = tf.reduce_mean(tf.nn.ctc_loss(labels=self._targets, inputs=self._output,
                                                           sequence_length=self._seq_len), name='loss')

    def _build_optimizer(self) -> None:

        adam = tf.train.AdamOptimizer(learning_rate=self.lr)
        self._loss_minimize = adam.minimize(self._ctc_loss)

    def _decode_output(self) -> Tuple[Tensor, ...]:

        with tf.name_scope('decoder'):

            # Decode predicted chars
            if self.use_beam_search:
                pred_chars, log_prob = tf.nn.ctc_beam_search_decoder(self._output, self._seq_len, merge_repeated=False)
            else:
                pred_chars, log_prob = tf.nn.ctc_greedy_decoder(self._output, self._seq_len, merge_repeated=False)

            pred_chars = tf.cast(pred_chars[0], tf.int32)
            dense_pred_chars = tf.sparse_tensor_to_dense(pred_chars, default_value=-1, name='dense_pred_chars')

        return pred_chars, log_prob, dense_pred_chars

    def _get_cer(self, pred_chars: Tensor) -> None:

        # Compute the Character Error Rate (CER) per batch in the computing graph
        self._cer = tf.reduce_mean(tf.edit_distance(pred_chars, self._targets), name='cer')

    @staticmethod
    def _get_macc(preds: Sequence[str],
                  labels: Sequence[str]) -> float:

        # Compute the Match Accuracy (1 - MER) per batch
        return (np.array(preds, dtype=np.object) == np.array(labels, dtype=np.object)).mean()

    @staticmethod
    def _print_report(epoch: int,
                      iteration: int,
                      train_loss: float,
                      val_loss: float, 
                      train_cer: float, 
                      val_cer: float,
                      train_macc: float,
                      val_macc: float) -> None:

        print('\n')
        print(f'Epoch:\t{epoch}')
        print(f'Iteration:\t{iteration}\n')
        print(f'Train loss:\t{train_loss:.5f}')
        print(f'Train CER:\t{train_cer:.5f}')
        print(f'Train MAcc:\t{train_macc:.5f}\n')
        print(f'Val loss:\t{val_loss:.5f}')
        print(f'Val CER:\t{val_cer:.5f}')
        print(f'Val MAcc:\t{val_macc:.5f}\n')

    def _build(self) -> None:

        # Build the computing graph
        with self._graph.as_default():

            self._build_inputs()
            self._build_model()
            self._build_loss()
            self._build_optimizer()

            pred_chars, _, self._dense_pred_chars = self._decode_output()
            self._get_cer(pred_chars)

    def train(self) -> None:
        '''
        Train the model with specified parameters.
        '''

        # Initialize a saver to save checkpoints
        with self._graph.as_default():
            saver = tf.train.Saver(max_to_keep=10, save_relative_paths=True)

        with self._session as sess:

            # Initialize all variables
            global_init, local_init = tf.global_variables_initializer(), tf.local_variables_initializer()
            sess.run([global_init, local_init])

            # Create lists for tracking loss values and metrics
            train_losses, val_losses = [], []
            train_preds, val_preds = [], []
            train_texts, val_texts = [], []
            train_cers, val_cers = [], []

            for i in range(self.epochs):

                train_dl = DataLoaderSynth90K(self.cfg, 'train')

                for j, (train_data, train_text, train_targets) in enumerate(tqdm(train_dl)):

                    feed_dict = {self._inputs: train_data, self._targets: train_targets, self._seq_len: [self.max_char_count] * self.train_bs}
                    train_loss, train_cer, _, train_pred = sess.run([self._ctc_loss, 
                                                                     self._cer,
                                                                     self._loss_minimize, 
                                                                     self._dense_pred_chars], feed_dict=feed_dict)

                    train_preds.append(arrays2labels(train_pred, self.alphabet))
                    train_losses.append(train_loss), train_cers.append(train_cer), train_texts.append(train_text)

                    # Validation
                    if j and j % self.val_interval == 0:

                        val_dl = DataLoaderSynth90K(self.cfg, 'val', self.val_steps)

                        for k, (val_data, val_text, val_targets) in enumerate(tqdm(val_dl)):



                            feed_dict = {self._inputs: val_data, self._targets: val_targets, self._seq_len: [self.max_char_count] * self.val_bs}
                            val_loss, val_cer, val_pred = sess.run([self._ctc_loss, 
                                                                    self._cer,
                                                                    self._dense_pred_chars], feed_dict=feed_dict)
                            
                            val_preds.append(arrays2labels(val_pred, self.alphabet))
                            val_losses.append(val_loss), val_cers.append(val_cer), val_texts.append(val_text)
                        
                        # Unpack the values into a single list 
                        train_preds, train_texts = list(chain(*train_preds)), list(chain(*train_texts))
                        val_preds, val_texts = list(chain(*val_preds)), list(chain(*val_texts))
                        
                        train_macc = self._get_macc(train_preds, train_texts)
                        val_macc = self._get_macc(val_preds, val_texts)

                        self._print_report(i, j,
                                           np.mean(train_losses),
                                           np.mean(val_losses), 
                                           np.mean(train_cers), 
                                           np.mean(val_cers),
                                           train_macc,
                                           val_macc)

                        train_losses, val_losses = [], []
                        train_preds, val_preds = [], []
                        train_texts, val_texts = [], []
                        train_cers, val_cers = [], []

                    if j and j % self.save_interval == 0:
                        saver.save(sess, str(self.ckpt_dir.joinpath('model')))

    def _preprocess(self, path: Union[Path, str]) -> np.ndarray:

        # Read and preprocess an image before predicting
        img = cv2.imread(str(path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = cv2.resize(img, (self.img_w, self.img_h)).astype(np.float32)
        img = img[None, ..., None] / 255

        return img

    def __call__(self, path: Union[Path, str]) -> Optional[str]:
        '''
        Predict a text by the input path.

        Args:
            path: pathlib.Path or str path to the image.

        Returns:
            The predicted text if the model has pretrained weights.
        '''
        
        if self.pretrained:

            data = self._preprocess(path)

            # Get tensors from the checkpoint
            inputs = self._graph.get_tensor_by_name('inputs/inputs:0')
            seq_len = self._graph.get_tensor_by_name('inputs/seq_len:0')
            max_char_count = self._graph.get_tensor_by_name('model/map_to_sequence/feature_sequence:0').get_shape().as_list()[1]
            dense_pred_chars = self._graph.get_tensor_by_name('decoder/dense_pred_chars:0')

            feed_dict = {inputs: data, seq_len: [max_char_count]}
            decoded_array = self._session.run(dense_pred_chars, feed_dict=feed_dict)

            return arrays2labels([list(decoded_array.squeeze(0))], self.alphabet)[0]

        else:
            raise ValueError('Please specify pretrained=True when initializing the model!')

    def evaluate(self,
                 dataset_name: str,
                 mode: str = 'test') -> Optional[Tuple[List[str], List[str], float, float]]:
        '''
        Evaluate the model by the specified dataset.

        Args:
            dataset_name: the name of the dataset to be evaluated (must be 'synth90k', 'SVT' or 'IIIT5K').
            mode: the part of the dataset to be evaluated (must be 'train', 'val' or 'test').

        Returns:
            A tuple with results of the evaluation (preds, labels, cer, macc) if the model has pretrained weights.
        '''
        
        if self.pretrained:
            
            if dataset_name == 'synth90k':
                eval_dl = DataLoaderSynth90K(self.cfg, mode)
            elif dataset_name == 'SVT':
                eval_dl = DataLoaderSVT(self.cfg, mode)
            elif dataset_name == 'IIIT5K':
                eval_dl = DataLoaderIIIT5K(self.cfg, mode)
            else:
                raise ValueError('The dataset_name must be "synth90k", "SVT" or "IIIT5K"!')

            # Get tensors from the checkpoint
            inputs = self._graph.get_tensor_by_name('inputs/inputs:0')
            targets_shape = self._graph.get_tensor_by_name('inputs/targets/shape:0')
            targets_values = self._graph.get_tensor_by_name('inputs/targets/values:0')
            targets_indices = self._graph.get_tensor_by_name('inputs/targets/indices:0')
            targets_tensor = tf.SparseTensor(targets_indices, targets_values, targets_shape)

            seq_len = self._graph.get_tensor_by_name('inputs/seq_len:0')
            max_char_count = self._graph.get_tensor_by_name('model/map_to_sequence/feature_sequence:0').get_shape().as_list()[1]
            cer_tensor = self._graph.get_tensor_by_name('cer:0')
            dense_pred_chars = self._graph.get_tensor_by_name('decoder/dense_pred_chars:0')

            preds, labels, cers = [], [], []
            for data, texts, targets in tqdm(eval_dl):

                feed_dict = {inputs: data, targets_tensor: targets, seq_len: [max_char_count] * data.shape[0]}
                cer_value, decoded_arrays = self._session.run([cer_tensor, dense_pred_chars], feed_dict=feed_dict)

                if self.val_bs == 1:
                    decoded_arrays = [list(decoded_arrays.squeeze(0))]                    

                preds.append(arrays2labels(decoded_arrays, self.alphabet))
                labels.append(texts), cers.append(cer_value)

            preds, labels = list(chain(*preds)), list(chain(*labels))

            macc = self._get_macc(preds, labels)
            cer = np.mean(cers)

            return preds, labels, cer, macc
        
        else:
            raise ValueError('Please specify pretrained=True when initializing the model!')