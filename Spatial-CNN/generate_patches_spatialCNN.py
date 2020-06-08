# -*- coding: utf-8 -*-
"""
Created on Thu May 30 14:32:05 2019

@author: clausmichele
"""

import argparse
from glob import glob
import random
from utilis import *

DATA_AUG_TIMES = 3  # transform a sample to a different sample for DATA_AUG_TIMES times

parser = argparse.ArgumentParser(description='')
parser.add_argument('--src_dir', dest='src_dir', default='./data/train/original', help='dir of data')
parser.add_argument('--save_dir', dest='save_dir', default='./data/train', help='dir of patches')
parser.add_argument('--src_dir_noisy', dest='src_dir_noisy', default='./data/train/noisy', help='dir of noisy data')
parser.add_argument('--save_dir_noisy', dest='save_dir_noisy', default='./data/train', help='dir of noisy patches')
parser.add_argument('--patch_size', dest='pat_size', type=int, default=50, help='patch size')
parser.add_argument('--stride', dest='stride', type=int, default=100, help='stride')
parser.add_argument('--batch_size', dest='bat_size', type=int, default=64, help='batch size')
args = parser.parse_args()

def sortKeyFunc(s):
	 return int(os.path.basename(s)[:-4])
	 
def generate_patches():
	global DATA_AUG_TIMES
	count = 0
	filepaths = glob(args.src_dir + '/*.png') #takes all the paths of the png files in the train folder
	filepaths.sort(key=sortKeyFunc) #order the file list
	filepaths_noisy = glob(args.src_dir_noisy + '/*.png')
	filepaths_noisy.sort(key=sortKeyFunc)
	print ("[*] Number of training samples: %d" % len(filepaths))
	scales = [1, 0.8]
	
	# calculate the number of patches
	for i in range(len(filepaths)):
		img = cv2.imread(filepaths[i])
		for s in range(len(scales)):
			newsize = (int(img.shape[0] * scales[s]), int(img.shape[1] * scales[s]))
			img_s = cv2.resize(img,newsize, interpolation = cv2.INTER_CUBIC)
			im_h = img_s.shape[0]
			im_w = img_s.shape[1]
			for x in range(0, (im_h - args.pat_size), args.stride):
				for y in range(0, (im_w - args.pat_size), args.stride):
					count += 1

	origin_patch_num = count * DATA_AUG_TIMES
	
	if origin_patch_num % args.bat_size != 0:
		numPatches = (origin_patch_num / args.bat_size + 1) * args.bat_size #round 
	else:
		numPatches = origin_patch_num
	print ("[*] Number of patches = %d, batch size = %d, total batches = %d" % \
		(numPatches, args.bat_size, numPatches / args.bat_size))

	# data matrix 4-D
	inputs = np.zeros((int(numPatches), args.pat_size, args.pat_size, 3), dtype="uint8") # clean patches
	inputs2 = np.zeros((int(numPatches), args.pat_size, args.pat_size, 3), dtype="uint8") # noisy patches
	
	count = 0
	# generate patches
	for i in range(len(filepaths)):
		img = cv2.imread(filepaths[i])
		img2 = cv2.imread(filepaths_noisy[i])
		for s in range(len(scales)):
			newsize = (int(img.shape[0] * scales[s]), int(img.shape[1] * scales[s]))
			img_s = cv2.resize(img,newsize, interpolation = cv2.INTER_CUBIC)
			img_s2 = cv2.resize(img2,newsize, interpolation = cv2.INTER_CUBIC) 
			img_s = np.reshape(np.array(img_s, dtype="uint8"), (img_s.shape[0], img_s.shape[1], 3))  # extend one dimension
			img_s2 = np.reshape(np.array(img_s2, dtype="uint8"), (img_s2.shape[0], img_s2.shape[1], 3))  # extend one dimension
			
			for j in range(DATA_AUG_TIMES):
				im_h = img_s.shape[0]; im_w = img_s.shape[1]
				for x in range(0, im_h - args.pat_size, args.stride):
					for y in range(0, im_w - args.pat_size, args.stride):
						a=random.randint(0, 7)
						inputs[count, :, :, :] = data_augmentation(img_s[x:x + args.pat_size, y:y + args.pat_size, :], a)
						inputs2[count, :, :, :] = data_augmentation(img_s2[x:x + args.pat_size, y:y + args.pat_size, :], a)
						count += 1
	# pad the batch
	if count < numPatches:
		to_pad = int(numPatches - count)
		inputs[-to_pad:, :, :, :] = inputs[:to_pad, :, :, :]
		inputs2[-to_pad:, :, :, :] = inputs2[:to_pad, :, :, :]


	if not os.path.exists(args.save_dir):
		os.mkdir(args.save_dir)
	np.save(os.path.join(args.save_dir, "img_clean_pats"), inputs)
	print("[*] size of input clean tensor = " + str(inputs.shape))
	if not os.path.exists(args.save_dir_noisy):
		os.mkdir(args.save_dir_noisy)
	np.save(os.path.join(args.save_dir_noisy, "img_noisy_pats"), inputs2)
	print("[*] size of input noisy tensor = " + str(inputs2.shape))
	print("[*] Patches generated and saved!")

if __name__ == '__main__':
	generate_patches()
