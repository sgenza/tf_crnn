import os
import yaml
import random
import numpy as np
import tensorflow as tf
from pathlib import Path
from typing import List, Tuple, Dict, Union, Sequence, Callable, Any


__all__ = ['label2array',
           'seed_everything', 
           'read_config', 
           'arrays2labels', 
           'sparse_tuple_from']


label2array: Callable[[str, str], List[int]] = lambda label, alphabet: [alphabet.index(x) for x in label]

def seed_everything(seed: int) -> None:

    random.seed(seed)
    np.random.seed(seed)
    tf.set_random_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['TF_DETERMINISTIC_OPS'] = '1'

def read_config(path: Union[Path, str]) -> Dict[str, Dict[str, Any]]:
    '''
    Read a YAML configuration file into a python dict.

    Args:
        path: pathlib.Path or str path to the configuration file.

    Returns:
        A dict of configuration values.
    '''
    
    with open(path, 'r') as f:
        data = yaml.safe_load(f)

    return data

def arrays2labels(arrays: Sequence[Sequence[int]],
                  alphabet: str) -> List[str]:
    '''
    Convert each item of a batch from an array of integers to a string.

    Args:
        arrays: the batch of arrays of integers where each integer is an index of a character in the alphabet.
        alphabet: a string of all unique symbols that can be predicted by the model.

    Returns:
        A list of strings where each string is a sequence of characters.
    '''

    return [''.join([alphabet[int(idx)] for idx in array if idx in list(range(len(alphabet)))]) \
            for array in arrays]

def sparse_tuple_from(sequences: Sequence[Sequence[int]]) -> Tuple:
    '''
    Create a sparse representation of x.

    Args:
        sequences: a list of lists of type dtype where each element is a sequence.

    Returns:
        A tuple with (indices, values, shape).

    Based on:
        https://github.com/igormq/ctc_tensorflow_example/blob/master/utils.py
    '''

    indices, values = [], []
    for n, seq in enumerate(sequences):
        indices.extend(zip([n] * len(seq), [i for i in range(len(seq))]))
        values.extend(seq)

    indices = np.asarray(indices, dtype=np.int64)
    values = np.asarray(values, dtype=np.int32)
    shape = np.asarray([len(sequences), np.asarray(indices).max(0)[1] + 1], dtype=np.int64)

    return indices, values, shape