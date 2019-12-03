#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu May 30 14:32:05 2019

@author: clausmichele
"""

import argparse
from glob import glob
import tensorflow as tf
import os
import numpy as np

from model_temp3CNN import TemporalDenoiser
from utilis import *

parser = argparse.ArgumentParser(description='')
parser.add_argument('--epoch', dest='epoch', type=int, default=10, help='# of epoch')
parser.add_argument('--batch_size', dest='batch_size', type=int, default=32, help='# images in batch')
parser.add_argument('--lr', dest='lr', type=float, default=0.0001, help='initial learning rate for adam')
parser.add_argument('--use_gpu', dest='use_gpu', type=int, default=1, help='gpu flag, 1 for GPU and 0 for CPU')
parser.add_argument('--phase', dest='phase', default='train', help='train or test')
parser.add_argument('--checkpoint_dir', dest='ckpt_dir', default='./ckpt', help='models are saved here')
parser.add_argument('--save_dir', dest='save_dir', default='./data/test/temporal_denoised', help='denoised sample are saved here')
args = parser.parse_args()

def sortKeyFunc(s):
	 return int(os.path.basename(s)[:-4])
	 
def denoiserTrain(TemporalDenoiser, lr):

	with load_data(filepath='./data/train/frames_clean_pats.npy') as data_:
		data = data_
	with load_data(filepath='./data/train/frames_noisy_pats.npy') as data_noisy_:
		data_noisy = data_noisy_

	noisy_eval_files = glob('./data/test/noisy/*.png')
	noisy_eval_files.sort(key=sortKeyFunc)
	eval_data_noisy = load_images(noisy_eval_files)
	  
	eval_files = glob('./data/test/original/*.png')
	eval_files.sort(key=sortKeyFunc)
	eval_data = load_images(eval_files)





	TemporalDenoiser.train(data, data_noisy, eval_data, eval_data_noisy, batch_size=args.batch_size, ckpt_dir=args.ckpt_dir, epoch=args.epoch, lr=lr)


def denoiserTest(TemporalDenoiser):
	noisy_eval_files = glob('./data/test/spatial_denoised/*.png')
	noisy_eval_files = sorted(noisy_eval_files)
	eval_files = glob('./data/test/original/*.png')
	eval_files = sorted(eval_files)
	TemporalDenoiser.test(noisy_eval_files, eval_files, ckpt_dir=args.ckpt_dir, save_dir=args.save_dir)


def main(_):
	if not os.path.exists(args.ckpt_dir):
		os.makedirs(args.ckpt_dir)
	if not os.path.exists(args.save_dir):
		os.makedirs(args.save_dir)

	lr = args.lr * np.ones([args.epoch])
	lr[30:] = lr[0] / 10.0
	if args.use_gpu:
		# added to control the gpu memory
		print("GPU\n")
		gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.7)
		with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
			model = TemporalDenoiser(sess)
			if args.phase == 'train':
				denoiserTrain(model, lr=lr)
			elif args.phase == 'test':
				denoiserTest(model)
			else:
				print('[!]Unknown phase')
				exit(0)
	else:
		print("CPU\n")
		with tf.Session() as sess:
			model = TemporalDenoiser(sess)
			if args.phase == 'train':
				denoiserTrain(model, lr=lr)
			elif args.phase == 'test':
				denoiserTest(model)
			else:
				print('[!]Unknown phase')
				exit(0)


if __name__ == '__main__':
	tf.app.run()
