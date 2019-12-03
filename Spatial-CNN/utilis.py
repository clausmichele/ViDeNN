# -*- coding: utf-8 -*-
"""
Created on Thu May 30 14:32:05 2019

@author: clausmichele
"""

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
		print("[*] Load successfully...")
		return self.data

	def __exit__(self, type, value, trace):
		del self.data
		gc.collect()


def load_data(filepath=''):
	print filepath
	return train_data(filepath=filepath)


def load_images(filelist):
	# pixel value range 0-255
	data = np.array( [cv2.imread(img) for img in filelist] )
	return data
