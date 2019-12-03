# -*- coding: utf-8 -*-
"""
Created on Thu May 30 14:32:05 2019

@author: clausmichele
"""

import argparse
from glob import glob
import sys
import tensorflow as tf
import os
from model_spatialCNN import denoiser
from utilis import *
import numpy as np

parser = argparse.ArgumentParser(description='')
parser.add_argument('--epoch', dest='epoch', type=int, default=50, help='# of epoch')
parser.add_argument('--batch_size', dest='batch_size', type=int, default=64, help='# images in batch')
parser.add_argument('--lr', dest='lr', type=float, default=0.001, help='initial learning rate for adam')
parser.add_argument('--use_gpu', dest='use_gpu', type=int, default=1, help='gpu flag, 1 for GPU and 0 for CPU')
parser.add_argument('--phase', dest='phase', default='train', help='train or test')
parser.add_argument('--checkpoint_dir', dest='ckpt_dir', default='./ckpt', help='checkpoints are saved here')
parser.add_argument('--save_dir', dest='save_dir', default='./data/denoised', help='denoised sample are saved here')
args = parser.parse_args()

def sortKeyFunc(s):
	 return int(os.path.basename(s)[:-4])
	 
def denoiser_train(denoiser, lr):
	with load_data('./data/train/img_clean_pats.npy') as data_:
		data = data_
	with load_data('./data/train/img_noisy_pats.npy') as data_noisy_:
		data_noisy = data_noisy_

	noisy_eval_files = glob('./data/test/noisy/*.png')
	noisy_eval_files = sorted(noisy_eval_files)
	eval_data_noisy = load_images(noisy_eval_files)
	eval_files = glob('./data/test/original/*.png')
	eval_files = sorted(eval_files)

	eval_data = load_images(eval_files)
	denoiser.train(data, data_noisy, eval_data[0:20], eval_data_noisy[0:20], batch_size=args.batch_size, ckpt_dir=args.ckpt_dir, epoch=args.epoch, lr=lr)


def denoiser_test(denoiser):
	noisy_eval_files = glob('./data/test/noisy/*.jpg')
	noisy_eval_files = sorted(noisy_eval_files)
	eval_files = glob('./data/test/original/*.jpg')
	eval_files = sorted(eval_files)
	denoiser.test(noisy_eval_files, eval_files, ckpt_dir=args.ckpt_dir, save_dir=args.save_dir)

def denoiser_for_temp3_training(denoiser):
	noisy_eval_files = glob('../Temp3-CNN/data/train/noisy/*/*.png')
	noisy_eval_files = sorted(noisy_eval_files)
	eval_files = glob('../Temp3-CNN/data/train/original/*/*.png')
	eval_files = sorted(eval_files)
	denoiser.test(noisy_eval_files, eval_files, ckpt_dir=args.ckpt_dir, save_dir='../Temp3-CNN/data/train/denoised/')


def main(_):
	if not os.path.exists(args.ckpt_dir):
		os.makedirs(args.ckpt_dir)
	if not os.path.exists(args.save_dir):
		os.makedirs(args.save_dir)

	lr = args.lr * np.ones([args.epoch])
	lr[3:] = lr[0] / 10.0
	if args.use_gpu:
		# Control the gpu memory setting per_process_gpu_memory_fraction
		print("GPU\n")
		gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.7)
		with tf.Session(config = tf.ConfigProto(gpu_options=gpu_options)) as sess:
			model = denoiser(sess)
			if args.phase == 'train':
				denoiser_train(model, lr=lr)
			elif args.phase == 'test':
				denoiser_test(model)
			elif args.phase == 'test_temp':
				denoiser_for_temp3_training(model)
			else:
				print('[!] Unknown phase')
				exit(0)
	else:
		print("CPU\n")
		with tf.device('/cpu:0'):
			with tf.Session() as sess:
				model = denoiser(sess)
				if args.phase == 'train':
					denoiser_train(model, lr=lr)
				elif args.phase == 'test':
					denoiser_test(model)
				elif args.phase == 'test_temp':
					denoiser_for_temp3_training(model)
				else:
					print('[!] Unknown phase')
					exit(0)


if __name__ == '__main__':
	tf.app.run()
