import numpy as np
import tensorflow as tf
import pandas as pd
import random
import xmltodict
from imageio import imread
from scipy.misc import imresize
from tensorflow.keras.utils import to_categorical
from mat4py import loadmat

import config
from utils import label_to_array, sparse_tuple_from


class DataGenerator(object):

	def __init__(self, batch_size, dataset, mode):
		
		self.batch_size = batch_size
		self.mode = mode
		self.dataset = dataset

		if self.mode not in {'train', 'val', 'test'}:
			raise ValueError('Invalid mode name "{}"'.format(self.mode))
		if self.dataset not in {'SVT', 'IIIT5K', 'Synth90k'}:
			raise ValueError('Invalid dataset name "{}"'.format(self.dataset))

	def __iter__(self):

		if self.dataset == 'Synth90k':

			dir_path = '90kDICT32px/'

			# File reading
			if self.mode == 'train':
				file_path = 'annotation_train.txt'
			elif self.mode == 'val':
				file_path = 'annotation_val.txt'
			elif self.mode == 'test':
				file_path = 'annotation_test.txt'

			with open(dir_path + file_path, 'r') as file:
				file_rows = file.readlines()

			self.dataset_len = len(file_rows)

			while True:

				# Random files for this batch
				batch_rows = [random.choice(file_rows) for _ in np.arange(self.batch_size)]

				x_batch = np.empty([self.batch_size, config.img_h, config.img_w, 1])
				y_batch = np.empty(self.batch_size, dtype=np.object)
				tg_batch = np.empty(self.batch_size, dtype=np.object)

				for i, row in enumerate(batch_rows):

					try:
						img_path = row.split(' ')[0]
						label = img_path.split('_')[1].upper()
						target = label_to_array(label)
						img = imread(dir_path + img_path)[:, :, 0]

					except (IndexError, ValueError):
						print('Error loading data, attempt to load another data')
						row = random.choice(file_rows)
						img_path = row.split(' ')[0]
						label = img_path.split('_')[1].upper()
						target = label_to_array(label)
						img = imread(dir_path + img_path)[:, :, 0]

					img = imresize(img, (config.img_h, config.img_w))[:, :, None]

					x_batch[i, :, :, :] = img
					y_batch[i] = label
					tg_batch[i] = target

				# Normalization
				x_batch /= 255
				tg_batch = sparse_tuple_from(np.reshape(np.array(tg_batch), (-1)))

				yield x_batch, y_batch, tg_batch

		if self.dataset == 'SVT':

			dir_path = 'SVT/'
			xml_path = 'test.xml' if self.mode == 'test' else 'train.xml'
			self.batch_size = 1 if self.mode == 'test' else self.batch_size

			with open(dir_path + xml_path, 'rb') as file:
				xmlDict = xmltodict.parse(file)
				df = pd.DataFrame.from_dict(xmlDict)

			# Get full dataset
			x_dataset, y_dataset, tg_dataset = [], [], []
			for image in df['tagset']['image']:

				img_path = image['imageName']
				full_img = imread(dir_path + img_path)

				if isinstance(image['taggedRectangles']['taggedRectangle'], list):

					# On scene few bounding boxes with text
					for word in image['taggedRectangles']['taggedRectangle']:

						label = word['tag']
						target = label_to_array(label)
						
						# Get bounding box and cut text
						height = int(word['@height'])
						width = int(word['@width'])
						x = int(word['@x'])
						y = int(word['@y'])
						word = full_img[y:y + height, x:x + width, :]
						if (word.shape[0] == 0) or (word.shape[1] == 0):
							continue

						word = imresize(word, (config.img_h, config.img_w))
						x_dataset.append(word), y_dataset.append(label), tg_dataset.append(target)
				else:
					# On scene one bounding box with text
					word = image['taggedRectangles']['taggedRectangle']
					label = word['tag']
					target = label_to_array(label)

					# Get bounding box and cut text
					height = int(word['@height'])
					width = int(word['@width'])
					x = int(word['@x'])
					y = int(word['@y'])
					word = full_img[y:y + height, x:x + width, :]

					word = imresize(word, (config.img_h, config.img_w))
					x_dataset.append(word), y_dataset.append(label), tg_dataset.append(target)

			# From RGB to grayscale and data normalization
			x_dataset = np.mean(np.array(x_dataset, dtype=np.float32), axis=-1)[:, :, :, None]
			y_dataset = np.array(y_dataset, dtype=np.object)
			tg_dataset = np.array(tg_dataset, dtype=np.object)
			x_dataset /= 255
			self.dataset_len = x_dataset.shape[0]

			if self.mode == ('train' or 'val'):

				while True:

					idx = np.random.choice(self.dataset_len, self.batch_size)
					x_batch, y_batch, tg_batch = x_dataset[idx], y_dataset[idx], tg_dataset[idx]
					tg_batch = sparse_tuple_from(np.reshape(np.array(tg_batch), (-1)))

					yield x_batch, y_batch, tg_batch

			elif self.mode == 'test':

				for idx in np.arange(self.dataset_len):

					x_batch, y_batch, tg_batch = x_dataset[idx], y_dataset[idx], tg_dataset[idx]
					tg_batch = sparse_tuple_from(np.reshape(np.array(tg_batch), (-1)))

					yield x_batch, y_batch, tg_batch

		if self.dataset == 'IIIT5K':

			dir_path = 'IIIT5K/'
			self.batch_size = 1 if self.mode == 'test' else self.batch_size

			if self.mode == ('train' or 'val'):
				file_path = 'traindata.mat'
				key = 'traindata'
			elif self.mode == 'test':
				file_path = 'testdata.mat'
				key = 'testdata'

			file = loadmat(dir_path + file_path)
			self.dataset_len = len(file[key]['ImgName'])

			# Get full dataset
			x_dataset, y_dataset, tg_dataset = [], [], []
			for i in np.arange(self.dataset_len):

				img_path = file[key]['ImgName'][i]
				label = file[key]['GroundTruth'][i]
				target = label_to_array(label)

				img = imread(dir_path + img_path)
				if img.ndim == 2:
					img = np.stack([img, img, img], axis=2)
				img = imresize(img, (config.img_h, config.img_w))

				x_dataset.append(img), y_dataset.append(label), tg_dataset.append(target)

			# From RGB to grayscale and data normalization
			x_dataset = np.mean(np.array(x_dataset, dtype=np.float32), axis=-1)[:, :, :, np.newaxis]
			x_dataset /= 255
			y_dataset = np.array(y_dataset, dtype=np.object)
			tg_dataset = np.array(tg_dataset, dtype=np.object)

			if self.mode == ('train' or 'val'):

				while True:

					idx = np.random.choice(self.dataset_len, self.batch_size)
					x_batch, y_batch, tg_batch = x_dataset[idx], y_dataset[idx], tg_dataset[idx]
					tg_batch = sparse_tuple_from(np.reshape(np.array(tg_batch), (-1)))

					yield x_batch, y_batch, tg_batch

			elif self.mode == 'test':

				for idx in np.arange(self.dataset_len):

					x_batch, y_batch, tg_batch = x_dataset[idx], y_dataset[idx], tg_dataset[idx]
					tg_batch = sparse_tuple_from(np.reshape(np.array(tg_batch), (-1)))

					yield x_batch, y_batch, tg_batch

if __name__ == '__main__':

	import time
	synth_gen = iter(DataGenerator(64, 'Synth90k', 'train'))
	start = time.time()
	for i in range(200):
		a = next(synth_gen)
		print(i)
	end = time.time()
	print('{} seconds'.format(end - start))