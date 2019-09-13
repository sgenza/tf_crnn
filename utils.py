import numpy as np
import config

def label_to_array(label):
	return [config.char_vector.index(x) for x in label]

def ground_truth_to_word(ground_truth):

    # return ''.join([config.char_vector[i] for i in ground_truth if i != -1])

    try:

        ground_truth_int = ground_truth.astype(np.int16)
        word = ''.join([config.char_vector[int(i)] for i in ground_truth_int if i in np.arange(len(config.char_vector))])

    except TypeError:

        print(ground_truth)
        print(ground_truth_int)

    return word

def batch_ground_truth_to_word(batch_ground_truth):

    batch_words = np.empty(config.batch_size, dtype=np.object)

    try:
        for j in np.arange(config.batch_size):

            batch_words[j] = ''.join([config.char_vector[int(i)] for i in batch_ground_truth[j] if i in np.arange(len(config.char_vector))])

    except TypeError:
        print(batch_ground_truth)
        print(batch_ground_truth[0])
        
    return batch_words


def to_max_len(word):
	while len(word) < config.max_word_len:
		word += '-'
	return word

def sparse_tuple_from(sequences, dtype=np.int32):
    """
        Inspired (copied) from https://github.com/igormq/ctc_tensorflow_example/blob/master/utils.py
    """

    indices = []
    values = []

    for n, seq in enumerate(sequences):
        indices.extend(zip([n]*len(seq), [i for i in range(len(seq))]))
        values.extend(seq)

    indices = np.asarray(indices, dtype=np.int64)
    values = np.asarray(values, dtype=dtype)
    shape = np.asarray([len(sequences), np.asarray(indices).max(0)[1]+1], dtype=np.int64)

    return indices, values, shape