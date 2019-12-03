import gc
import os
import sys

import numpy as np
import tensorflow as tf
import cv2

def data_augmentation(image, mode):
	if mode == 0:
		# original
		return image
	elif mode == 1:
		# flip up and down
		return np.flipud(image)
	elif mode == 2:
		# rotate counterwise 90 degree
		return np.rot90(image)
	elif mode == 3:
		# rotate 90 degree and flip up and down
		image = np.rot90(image)
		return np.flipud(image)
	elif mode == 4:
		# rotate 180 degree
		return np.rot90(image, k=2)
	elif mode == 5:
		# rotate 180 degree and flip
		image = np.rot90(image, k=2)
		return np.flipud(image)
	elif mode == 6:
		# rotate 270 degree
		return np.rot90(image, k=3)
	elif mode == 7:
		# rotate 270 degree and flip
		image = np.rot90(image, k=3)
		return np.flipud(image)


class train_data():
	def __init__(self, filepath=''):
		self.filepath = filepath
		assert '.npy' in filepath
		os.path.isfile(filepath)
		if not os.path.isfile(filepath):
			print("[!] Data file not exists")
			sys.exit(1)

	def __enter__(self):
		print("[*] Loading data...")
		self.data = np.load(self.filepath)
		#np.random.shuffle(self.data)
		print("[*] Load successfully...")
		return self.data

	def __exit__(self, type, value, trace):
		del self.data
		gc.collect()

def load_data(filepath=''):
	return train_data(filepath=filepath)

def load_images(filelist):
	# pixel value range 0-255
	data = np.array( [cv2.imread(img) for img in filelist] )
	return data


def save_images(filepath, ground_truth, noisy_image=None, clean_image=None):
	# assert the pixel value range is 0-255
	ground_truth = np.squeeze(ground_truth)
	noisy_image = np.squeeze(noisy_image)
	clean_image = np.squeeze(clean_image)
	if not clean_image.any():
		cat_image = ground_truth
	else:
		cat_image = np.concatenate([ground_truth, noisy_image, clean_image], axis=1)
	cv2.imwrite(filepath, cat_image)


def cal_psnr(im1, im2):
	# assert pixel value range is 0-255 and type is uint8
	mse = float(((im1.astype(np.float32) - im2.astype(np.float32)) ** 2).mean())
	psnr = 10 * np.log10(255 ** 2 / mse)
	return psnr


def tf_psnr(im1, im2):
	# assert pixel value range is 0-1
	mse = tf.losses.mean_squared_error(labels=im2 * 255.0, predictions=im1 * 255.0)
	return 10.0 * (tf.log(255.0 ** 2 / mse) / tf.log(10.0))