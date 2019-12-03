# -*- coding: utf-8 -*-
"""
@author: clausmichele
"""

import argparse
from glob import glob
import tensorflow as tf
import os

from model_ViDeNN import ViDeNN

parser = argparse.ArgumentParser(description='')
parser.add_argument('--use_gpu', dest='use_gpu', type=int, default=1, help='gpu flag, 1 for GPU and 0 for CPU')
parser.add_argument('--save_dir', dest='save_dir', default='./data/denoised', help='denoised sample are saved here')
parser.add_argument('--test_dir', dest='test_dir', default='./data', help='directory of noisy frames')
parser.add_argument('--img_format', dest='img_format', default='png', help='denoised sample are saved here')
parser.add_argument('--checkpoint_dir', dest='ckpt_dir', default=None, help='path of ViDeNN checkpoint')
args = parser.parse_args()

def ViDeNNDenoise(ViDeNN):
	eval_files_noisy = glob(args.test_dir + "/noisy/*." + args.img_format)
	eval_files_noisy = sorted(eval_files_noisy)
	eval_files = glob(args.test_dir + "/original/*." + args.img_format)
	print_psnr = True
	if eval_files == []:
		eval_files = eval_files_noisy
		print_psnr = False
		print("[*] No original frames found, not printing PSNR values...")
	eval_files = sorted(eval_files)
	ViDeNN.denoise(eval_files, eval_files_noisy, print_psnr, args.ckpt_dir, args.save_dir)

def main(_):
	if not os.path.exists(args.test_dir):
		os.makedirs(args.test_dir)
	if not os.path.exists(args.save_dir):
		os.makedirs(args.save_dir)
	if args.use_gpu:
		# added to control the gpu memory
		print("GPU\n")
		gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.8)
		with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
			model = ViDeNN(sess)
			ViDeNNDenoise(model)
	else:
		print("CPU\n")
		with tf.device('/cpu:0'):
			with tf.Session() as sess:
				model = ViDeNN(sess)
				ViDeNNDenoise(model)


if __name__ == '__main__':
	tf.app.run()
