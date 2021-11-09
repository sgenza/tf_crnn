import argparse
import tensorflow as tf
from pathlib import Path

from model import CRNN, DEFAULT_CONFIG_PATH, read_config


DEVICE = '/gpu:0'


def _get_args():
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, default=DEFAULT_CONFIG_PATH,
                                    help='The config file path.')
    parser.add_argument('--device', type=str, default=DEVICE,
                                    help='The training device.')

    return parser.parse_args()

def main():

    params = _get_args()
    cfg = read_config(params.config)

    with tf.device(params.device):

        model = CRNN(cfg)
        model.train()


if __name__ == '__main__':
    main()