import cv2
import xmltodict
import numpy as np
import pandas as pd
from mat4py import loadmat
from pathlib import Path
from typing import Optional, Mapping, Dict, Any

from .utils import label2array, sparse_tuple_from


__all__ = ['DataLoaderSynth90K, DataLoaderSVT', 'DataLoaderIIIT5K']


class DataLoaderSynth90K(object):

    '''
    A class for loading data from the Synth90K dataset.
    To download the dataset please follow the link: https://www.robots.ox.ac.uk/~vgg/data/text
    '''

    def __init__(self,
                 cfg: Mapping[str, Mapping[str, Any]], 
                 mode: str, 
                 steps: Optional[int] = None):
        '''
        Initializes the data loader.

        Args:
            cfg: a dict with configuration values.
            mode: the part of the dataset to be used (must be 'train', 'val' or 'test').
            steps: a number of steps (optional, if not specified then will be computed by the all data)
        '''
        
        self.n = 0
        self.mode = mode
        self.cfg = cfg
        self.img_w = cfg['common']['img_width']
        self.img_h = cfg['common']['img_height']
        self.alphabet = cfg['common']['alphabet']
        self.dataset_dir = Path(self.cfg['common']['mjsynth_dir'])

        if self.mode == 'train':
            annotation_path = self.dataset_dir.joinpath('annotation_train.txt')
            self.batch_size = self.cfg['train']['batch_size']
        elif self.mode == 'val':
            annotation_path = self.dataset_dir.joinpath('annotation_val.txt')
            self.batch_size = self.cfg['eval']['batch_size']
        elif self.mode == 'test':
            annotation_path = self.dataset_dir.joinpath('annotation_test.txt')
            self.batch_size = self.cfg['eval']['batch_size']
        else:
            raise ValueError('The mode must be "train", "val" or "test"!')

        with open(annotation_path, 'r') as f:
            self._file_names = f.readlines()

        dataset_steps = len(self._file_names) // self.batch_size
        self.steps = steps if steps and steps <= dataset_steps else dataset_steps

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(cfg, mode={self.mode}, steps={self.steps})'
    
    def __iter__(self):
        return self

    def __len__(self) -> int:
        return self.steps

    def __next__(self):
        
        if self.n < len(self):
            
            if self.mode == 'train':
                # Random sampler with repeats
                file_names = np.random.choice(self._file_names, self.batch_size)
            else:
                # Sequential sampler
                file_names = self._file_names[self.n * self.batch_size:(self.n + 1) * self.batch_size]

            self.n += 1
            imgs, texts, targets = [], [], []

            for i, file_name in enumerate(file_names):

                path = self.dataset_dir.joinpath(file_name.split(' ')[0])
                img = cv2.imread(str(path))

                while img is None:

                    # The case when the data is corrupted 
                    file_name = np.random.choice(self._file_names, 1).item()
                    path = self.dataset_dir.joinpath(file_name.split(' ')[0])
                    img = cv2.imread(str(path))

                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                img = cv2.resize(img, (self.img_w, self.img_h)).astype(np.float32)

                label = file_name.split('_')[1].upper()
                target = label2array(label, self.alphabet)

                imgs.append(img[..., None]), texts.append(label), targets.append(target)

            imgs = np.array(imgs, dtype=np.float32)
            texts = np.array(texts, dtype=np.object)

            imgs /= 255
            targets = sparse_tuple_from(targets)

            return imgs, texts, targets

        else:
            raise StopIteration


class DataLoaderSVT(object):
    '''
    A class for loading data from the Street View Text (SVT) dataset.
    To download the dataset please follow the link: http://vision.ucsd.edu/~kai/svt/
    '''

    def __init__(self, 
                 cfg: Dict, 
                 mode: str):
        '''
        Initializes the data loader.

        Args:
            cfg: a dict with configuration values.
            mode: the part of the dataset to be used (must be 'train', 'val' or 'test').
        '''
        
        self.n = 0
        self.mode = mode
        self.cfg = cfg
        self.img_w = cfg['common']['img_width']
        self.img_h = cfg['common']['img_height']
        self.alphabet = cfg['common']['alphabet']
        self.dataset_dir = Path(self.cfg['common']['svt_dir'])

        if self.mode == 'train':
            xml_path = self.dataset_dir.joinpath('train.xml')
            self.batch_size = self.cfg['train']['batch_size']
        elif self.mode == 'test':
            xml_path = self.dataset_dir.joinpath('test.xml')
            self.batch_size = self.cfg['eval']['batch_size']
        else:
            raise ValueError('The mode must be "train", "val" or "test"!')

        with open(xml_path, 'rb') as f:
            xml_dict = xmltodict.parse(f)
            raw_df = pd.DataFrame.from_dict(xml_dict['tagset'])

        self._df = self._prepare_data(raw_df)

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(cfg, mode={self.mode})'

    def __iter__(self):
        return self

    def __len__(self) -> int:
        return len(self._df) // self.batch_size

    @staticmethod
    def _prepare_data(raw_df: pd.DataFrame) -> pd.DataFrame:

        data = []
        for i in range(len(raw_df)):

            df_item = raw_df.iloc[i]['image']
            file_name = df_item['imageName']
            
            if isinstance(df_item['taggedRectangles']['taggedRectangle'], list):
                
                # Several bounding boxes on the scene
                for word in df_item['taggedRectangles']['taggedRectangle']:

                    label = word['tag']
                    height = int(word['@height'])
                    width = int(word['@width'])
                    x = int(word['@x'])
                    y = int(word['@y'])
                    
                    coords = [x, y, width, height]
                    if (np.array(coords) < 0).any():
                        continue

                    data.append([file_name, label, coords])
            else:
                
                # One bounding box on the scene
                word = df_item['taggedRectangles']['taggedRectangle']
                label = word['tag']
                height = int(word['@height'])
                width = int(word['@width'])
                x = int(word['@x'])
                y = int(word['@y'])
                
                coords = [x, y, width, height]
                if (np.array(coords) < 0).any():
                    continue
                
                data.append([file_name, label, coords])

        df = pd.DataFrame(data, columns=['file_name', 'label', 'coords'])

        return df

    def __next__(self):
        
        if self.n < len(self):

            if self.mode == 'train':
                # Random sampler with repeats
                idxs = np.random.choice(range(len(self._df)), self.batch_size)
            else:
                # Sequential sampler
                idxs = np.arange(len(self._df))[self.n * self.batch_size:(self.n + 1) * self.batch_size]

            self.n += 1
            imgs, texts, targets = [], [], []
            for idx in idxs:

                df_item = self._df.iloc[idx]
                path = self.dataset_dir.joinpath(df_item['file_name'])
                
                img = cv2.imread(str(path))
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                x, y, width, height = df_item['coords']
                label = df_item['label']
                target = label2array(label, self.alphabet)

                word_img = img[y:y + height, x:x + width]
                word_img = cv2.resize(word_img, (self.img_w, self.img_h)).astype(np.float32)

                imgs.append(word_img[..., None]), texts.append(label), targets.append(target)

            imgs = np.array(imgs, dtype=np.float32)
            texts = np.array(texts, dtype=np.object)

            imgs /= 255
            targets = sparse_tuple_from(targets)

            return imgs, texts, targets

        else:
            raise StopIteration

class DataLoaderIIIT5K(object):
    '''
    A class for loading data from the IIIT 5K-word dataset.
    To download the dataset please follow the link: https://cvit.iiit.ac.in/research/projects/cvit-projects/the-iiit-5k-word-dataset
    '''

    def __init__(self,
                 cfg: Dict,
                 mode: str):
        '''
        Initializes the data loader.

        Args:
            cfg: a dict with configuration values.
            mode: the part of the dataset to be used (must be 'train', 'val' or 'test').
        '''
        
        self.n = 0
        self.mode = mode
        self.cfg = cfg
        self.img_w = cfg['common']['img_width']
        self.img_h = cfg['common']['img_height']
        self.alphabet = cfg['common']['alphabet']
        self.dataset_dir = Path(self.cfg['common']['iiit5k_dir'])

        if self.mode == 'train':
            mat_path = self.dataset_dir.joinpath('traindata.mat')
            key = 'traindata'
            self.batch_size = self.cfg['train']['batch_size']
        elif self.mode == 'test':
            mat_path = self.dataset_dir.joinpath('testdata.mat')
            key = 'testdata'
            self.batch_size = self.cfg['eval']['batch_size']
        else:
            raise ValueError('The mode must be "train", "val" or "test"!')

        self._df = pd.DataFrame(loadmat(str(mat_path))[key])

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(cfg, mode={self.mode})'

    def __iter__(self):
        return self

    def __len__(self) -> int:
        return len(self._df) // self.batch_size

    def __next__(self):
        
        if self.n < len(self):

            if self.mode == 'train':
                # Random sampler with repeats
                idxs = np.random.choice(range(len(self._df)), self.batch_size)
            else:
                # Sequential sampler
                idxs = np.arange(len(self._df))[self.n * self.batch_size:(self.n + 1) * self.batch_size]

            self.n += 1
            imgs, texts, targets = [], [], []
            for idx in idxs:

                df_item = self._df.iloc[idx]
                path = self.dataset_dir.joinpath(df_item['ImgName'])
                label = df_item['GroundTruth']
                
                img = cv2.imread(str(path))
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                target = label2array(label, self.alphabet)

                img = cv2.resize(img, (self.img_w, self.img_h)).astype(np.float32)

                imgs.append(img[..., None]), texts.append(label), targets.append(target)

            imgs = np.array(imgs, dtype=np.float32)
            texts = np.array(texts, dtype=np.object)

            imgs /= 255
            targets = sparse_tuple_from(targets)

            return imgs, texts, targets

        else:
            raise StopIteration