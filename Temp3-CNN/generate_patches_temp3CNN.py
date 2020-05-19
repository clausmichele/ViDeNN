#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
@author: clausmichele
"""

import argparse
import glob
import random
from utilis import *

# the pixel value range is '0-255'(uint8 ) of training data

DATA_AUG_TIMES = 3  # transform a sample to a different sample for DATA_AUG_TIMES times

parser = argparse.ArgumentParser(description='')
args = parser.parse_args()


def generate_patches():
	pat_size = 50
	stride = 50
	bat_size = 32
	step = 50
	global DATA_AUG_TIMES
	filepaths = glob.glob('./data/train/original/*/*.png') #takes all the paths of the png files in the train folder
	filepaths = sorted(filepaths)
	filepaths_noisy = glob.glob('./data/train/denoised/*/*.png')
	filepaths_noisy = sorted(filepaths_noisy)
	count = 0
	print("number of training data %d" % len(filepaths))
	scales = [1]

	# calculate the number of patches
	for i in range(len(filepaths)):
		img = cv2.imread(filepaths[i])
		for s in range(len(scales)):
			newsize = (int(img.shape[0] * scales[s]), int(img.shape[1] * scales[s]))
			img_s = cv2.resize(img,newsize, interpolation = cv2.INTER_CUBIC)
			im_h = img_s.shape[0]
			im_w = img_s.shape[1]
			for x in range(0, (im_h - pat_size), step):
				for y in range(0, (im_w - pat_size), step):
					count += 1

	origin_patch_num = count * DATA_AUG_TIMES

	if origin_patch_num % bat_size != 0:
		numPatches = (origin_patch_num / bat_size + 1) * bat_size #round 
	else:
		numPatches = origin_patch_num

	count = 0
	print("total patches = %d , batch size = %d, total batches = %d" % (numPatches, bat_size, numPatches / bat_size))
	# data matrix 4-D
	inputs = np.zeros((int(numPatches), pat_size, pat_size, 9), dtype="uint8")
	inputs2 = np.zeros((int(numPatches), pat_size, pat_size, 9), dtype="uint8")
	# generate patches
	ind = np.multiply(range(int(len(filepaths_noisy)/3)),3)
	random.shuffle(ind)
	for i in ind:
		img0 = cv2.imread(filepaths[i])
		img0_n = cv2.imread(filepaths_noisy[i])
		img1 = cv2.imread(filepaths[i+1])
		img1_n = cv2.imread(filepaths_noisy[i+1])
		img2 = cv2.imread(filepaths[i+2])
		img2_n = cv2.imread(filepaths_noisy[i+2])

		for s in range(len(scales)):
			newsize = (int(img0.shape[0] * scales[s]), int(img0.shape[1] * scales[s]))
			# print newsize
			img0s = cv2.resize(img0,newsize, interpolation = cv2.INTER_CUBIC)
			img0_ns = cv2.resize(img0_n,newsize, interpolation = cv2.INTER_CUBIC) 
			
			img1s = cv2.resize(img1,newsize, interpolation = cv2.INTER_CUBIC)
			img1_ns = cv2.resize(img1_n,newsize, interpolation = cv2.INTER_CUBIC) 
			
			img2s = cv2.resize(img2,newsize, interpolation = cv2.INTER_CUBIC)
			img2_ns = cv2.resize(img2_n,newsize, interpolation = cv2.INTER_CUBIC)

			img0s = np.reshape(np.array(img0s, dtype="uint8"), (img0s.shape[0], img0s.shape[1], 3))
			img0_ns = np.reshape(np.array(img0_ns, dtype="uint8"), (img0_ns.shape[0], img0_ns.shape[1], 3))   
			img1s = np.reshape(np.array(img1s, dtype="uint8"), (img1s.shape[0], img1s.shape[1], 3))		 
			img1_ns = np.reshape(np.array(img1_ns, dtype="uint8"), (img1_ns.shape[0], img1_ns.shape[1], 3))   
			img2s = np.reshape(np.array(img2s, dtype="uint8"), (img2s.shape[0], img2s.shape[1], 3))		 
			img2_ns = np.reshape(np.array(img2_ns, dtype="uint8"), (img2_ns.shape[0], img2_ns.shape[1], 3))
			for j in range(DATA_AUG_TIMES):
				im_h = img0s.shape[0]; im_w = img0s.shape[1]
				for x in range(0 + step, im_h - pat_size, stride):
					for y in range(0 + step, im_w - pat_size, stride):
						a=random.randint(0, 7)
						inputs[count, :, :, 0:3] = data_augmentation(img0s[x:x + pat_size, y:y + pat_size, :], a)
						inputs[count, :, :, 3:6] = data_augmentation(img1s[x:x + pat_size, y:y + pat_size, :], a)
						inputs[count, :, :, 6:] = data_augmentation(img2s[x:x + pat_size, y:y + pat_size, :], a)
						inputs2[count, :, :, 0:3] = data_augmentation(img0_ns[x:x + pat_size, y:y + pat_size, :], a)
						inputs2[count, :, :, 3:6] = data_augmentation(img1_ns[x:x + pat_size, y:y + pat_size, :], a)
						inputs2[count, :, :, 6:] = data_augmentation(img2_ns[x:x + pat_size, y:y + pat_size, :], a)
						count += 1

	inputs = inputs[:count-1]
	inputs2 = inputs2[:count-1]
	np.save('./data/train/frames_clean_pats', inputs)
	print("[*] Size of input clean tensor = " + str(inputs.shape))
	np.save('./data/train/frames_noisy_pats', inputs2)
	print("[*] Size of input noisy tensor = " + str(inputs2.shape))
	print("[*] Patches generated and saved!")

if __name__ == '__main__':
	generate_patches()
