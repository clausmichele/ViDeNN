# -*- coding: utf-8 -*-
"""
@author: clausmichele
"""

import time
import tensorflow as tf
import cv2
import numpy as np
from tqdm import tqdm

def SpatialCNN(input, is_training=False, output_channels=3, reuse=tf.AUTO_REUSE):
	with tf.variable_scope('block1',reuse=reuse):
		output = tf.layers.conv2d(input, 128, 3, padding='same', activation=tf.nn.relu)
	for layers in range(2, 20):
		with tf.variable_scope('block%d' % layers,reuse=reuse):
			output = tf.layers.conv2d(output, 64, 3, padding='same', name='conv%d' % layers, use_bias=False)
			output = tf.nn.relu(tf.layers.batch_normalization(output, training=is_training))
	with tf.variable_scope('block20', reuse=reuse):
		output = tf.layers.conv2d(output, output_channels, 3, padding='same', use_bias=False)
	return input - output

def Temp3CNN(input, is_training=False, output_channels=3, reuse=tf.AUTO_REUSE):
	input_middle = input[:,:,:,3:6]
	with tf.variable_scope('temp-block1',reuse=reuse):
		output = tf.layers.conv2d(input, 128, 3, padding='same', activation=tf.nn.leaky_relu)
	for layers in range(2, 20):
		with tf.variable_scope('temp-block%d' % layers,reuse=reuse):
			output = tf.layers.conv2d(output, 64, 3, padding='same', name='conv%d' % layers, use_bias=False)
			output = tf.nn.leaky_relu(output)
	with tf.variable_scope('temp-block20', reuse=reuse):
		output = tf.layers.conv2d(output, output_channels, 3, padding='same', use_bias=False)
	return input_middle - output

class ViDeNN(object):
	def __init__(self, sess):
		self.sess = sess
		# build model
		self.Y_ = tf.placeholder(tf.float32, [None, None, None, 3],name='clean_image')
		self.X = tf.placeholder(tf.float32, [None, None, None, 3],name='noisy_image')
		self.Y = SpatialCNN(self.X)
		self.Y_frames = tf.placeholder(tf.float32, [None, None, None, 9],name='clean_frames')
		self.Xframes = tf.placeholder(tf.float32, [None, None, None, 9],name='noisy_frames')
		self.Yframes = Temp3CNN(self.Xframes)
		init = tf.global_variables_initializer()
		self.sess.run(init)
		print("[*] Initialize model successfully...")

	def denoise(self, eval_files, eval_files_noisy, print_psnr, ckpt_dir, save_dir):
		# init variables
		tf.global_variables_initializer().run()
		assert len(eval_files) != 0, '[!] No testing data!'
		if ckpt_dir is None:
			full_path = tf.train.latest_checkpoint('./Temp3-CNN/ckpt')
			if(full_path is None):
				print('[!] No Temp3-CNN checkpoint!')
				quit()
			vars_to_restore_temp3CNN = {}
			for i in range(len(tf.global_variables())):
				if tf.global_variables()[i].name[0] != 'b':
					a = tf.global_variables()[i].name.split(':')[0]
					vars_to_restore_temp3CNN[a] = tf.global_variables()[i]
			saver_t = tf.train.Saver(var_list=vars_to_restore_temp3CNN)
			saver_t.restore(self.sess, full_path)
	
			full_path = tf.train.latest_checkpoint('./Spatial-CNN/ckpt_awgn')
			if(full_path is None):
				print('[!] No Spatial-CNN checkpoint!')
				quit()
			vars_to_restore_spatialCNN = {}
			for i in range(len(tf.global_variables())):
				if tf.global_variables()[i].name[0] != 't':
					a = tf.global_variables()[i].name.split(':')[0]
					vars_to_restore_spatialCNN[a] = tf.global_variables()[i]
			saver_s = tf.train.Saver(var_list=vars_to_restore_spatialCNN)
			saver_s.restore(self.sess, full_path)
		else:
			load_model_status, _ = self.load(ckpt_dir)
		print("[*] Model restore successfully!")
#
		psnr_sum = 0
		start = time.time()
		for idx in tqdm(range(len(eval_files)-1)):
			if idx==0:
				test = cv2.imread(eval_files[idx])
				test1 = cv2.imread(eval_files[idx+1])
				test2 = cv2.imread(eval_files[idx+2])
				noisy = cv2.imread(eval_files_noisy[idx])
				noisy1 = cv2.imread(eval_files_noisy[idx+1])
				noisy2 = cv2.imread(eval_files_noisy[idx+2])
				
				test = test.astype(np.float32) / 255.0
				test1 = test1.astype(np.float32) / 255.0
				test2 = test2.astype(np.float32) / 255.0
				noisy = noisy.astype(np.float32) / 255.0
				noisy1 = noisy1.astype(np.float32) / 255.0
				noisy2 = noisy2.astype(np.float32) / 255.0
				
				noisyin2 = np.zeros((1,test.shape[0],test.shape[1],9))	
				current = np.zeros((test.shape[0],test.shape[1],3)) 
				previous = np.zeros((test.shape[0],test.shape[1],3)) 
				
				noisyin = np.zeros((3,test.shape[0],test.shape[1],3))
				noisyin[0] = noisy
				noisyin[1] = noisy1
				noisyin[2] = noisy2 
				out = self.sess.run([self.Y],feed_dict={self.X:noisyin})
				out = np.asarray(out)

				noisyin2[0,:,:,0:3] = out[0,0]
				noisyin2[0,:,:,3:6] = out[0,0]
				noisyin2[0,:,:,6:] = out[0,1]
				temp_clean_image= self.sess.run([self.Yframes],feed_dict={self.Xframes:noisyin2})
				temp_clean_image = np.asarray(temp_clean_image)
				cv2.imwrite(save_dir + '/%04d.png'%idx,temp_clean_image[0,0]*255)
				psnr = psnr_scaled(test,temp_clean_image[0,0])
				psnr1 = psnr_scaled(test,out[0,0])
				psnr_sum += psnr
				if print_psnr: print(" frame %d denoised, PSNR: %.2f" % (idx, psnr))
				else: print(" frame %d denoised" % (idx))

				noisyin2[0,:,:,0:3] = out[0,0]
				noisyin2[0,:,:,3:6] = out[0,1]
				noisyin2[0,:,:,6:] = out[0,2]
				current[:,:,:] = out[0,2,:,:,:]
				previous[:,:,:] = out[0,1,:,:,:]
			else:
				if idx<(len(eval_files)-2):
					test3 = cv2.imread(eval_files[idx+2])
					test3 = test3.astype(np.float32) / 255.0
					noisy3 = cv2.imread(eval_files_noisy[idx+2])
					noisy3 = noisy3.astype(np.float32) / 255.0
	
					out2 = self.sess.run([self.Y],feed_dict={self.X:np.expand_dims(noisy3,0)})
					out2 = np.asarray(out2)
					noisyin2[0,:,:,0:3] = previous
					noisyin2[0,:,:,3:6] = current
					noisyin2[0,:,:,6:] = out2[0,0]
					previous = current
					current = out2[0,0]
				else:
					try:
						out2
					except NameError:
						out2 = np.zeros((out.shape))
						out2=out
						out2[0,0]=out[0,2]
					noisyin2[0,:,:,0:3] = current
					noisyin2[0,:,:,3:6] = out2[0,0]
					noisyin2[0,:,:,6:] = out2[0,0]
			temp_clean_image= self.sess.run([self.Yframes],feed_dict={self.Xframes:noisyin2})

			temp_clean_image = np.asarray(temp_clean_image)
			cv2.imwrite(save_dir+ '/%04d.png'%(idx+1),temp_clean_image[0,0]*255)

			# calculate PSNR
			if idx==0:
				psnr1 = psnr_scaled(test1,out[0,1])
				psnr = psnr_scaled(test1, temp_clean_image[0,0])
			else:
				psnr1 = psnr_scaled(test2,previous)
				psnr = psnr_scaled(test2, temp_clean_image[0,0])
				try:
					test3
				except NameError:
					test3=test2
				test2=test3
			if print_psnr: print(" frame %d denoised, PSNR: %.2f" % (idx+1, psnr))
			else: print(" frame %d denoised" % (idx+1))
			psnr_sum += psnr
		avg_psnr = psnr_sum / len(eval_files)
		if print_psnr: print("--- Average PSNR %.2f ---" % avg_psnr)
		print("--- Elapsed time: %.4fs" %(time.time()-start))

	def load(self, checkpoint_dir):
		print("[*] Reading checkpoint...")
		saver = tf.train.Saver()
		ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
		if ckpt and ckpt.model_checkpoint_path:
			full_path = tf.train.latest_checkpoint(checkpoint_dir)
			global_step = int(full_path.split('/')[-1].split('-')[-1])
			saver.restore(self.sess, full_path)
			return True, global_step
		else:
			return False, 0

def psnr_scaled(im1, im2): # PSNR function for 0-1 values
	mse = ((im1 - im2) ** 2).mean()
	mse = mse * (255 ** 2)
	psnr = 10 * np.log10(255 **2 / mse)
	return psnr
