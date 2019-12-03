# -*- coding: utf-8 -*-
"""
@author: clausmichele
"""

import time
import tensorflow as tf
import numpy as np
import os
from utilis import *
from tensorflow.python.tools.inspect_checkpoint import print_tensors_in_checkpoint_file

def Temp3CNN(input, is_training=True, output_channels=3, reuse=tf.AUTO_REUSE):
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

def shuffle_in_unison(a, b):
	rng_state = np.random.get_state()
	np.random.shuffle(a)
	np.random.set_state(rng_state)
	np.random.shuffle(b)

class TemporalDenoiser(object):
	def __init__(self, sess, input_c_dim=9, batch_size=32):
		self.sess = sess
		self.input_c_dim = input_c_dim
		# build model
		self.Y_ = tf.placeholder(tf.float32, [None, None, None, self.input_c_dim],name='clean_frames')
		self.is_training = tf.placeholder(tf.bool, name='is_training')
		self.X = tf.placeholder(tf.float32, [None, None, None, self.input_c_dim],name='noisy_frames')
		self.Y = Temp3CNN(self.X, is_training=self.is_training)
		self.loss = (1.0 / batch_size) * tf.nn.l2_loss(self.Y_[:,:,:,3:6] - self.Y)
		self.lr = tf.placeholder(tf.float32, name='learning_rate')
		self.eva_psnr = tf_psnr(self.Y, self.Y_[:,:,:,3:6])
		optimizer = tf.train.AdamOptimizer(self.lr, name='AdamOptimizer')
		update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
		with tf.control_dependencies(update_ops):
			self.train_op = optimizer.minimize(self.loss)
		init = tf.global_variables_initializer()
		self.sess.run(init)
		print("[*] Initialize model successfully...")


	def train(self, data, data_noisy, eval_data, eval_data_noisy, batch_size, ckpt_dir, epoch, lr, eval_every_epoch=2):
		# assert data range is between 0 and 1
		numBatch = int(data.shape[0] / batch_size)
		# load pretrained model
		load_model_status, global_step = self.load(ckpt_dir)
		if load_model_status:
			iter_num = global_step
			start_epoch = global_step // numBatch
			start_step = global_step % numBatch
			print("[*] Model restore successfully!")
		else:
			iter_num = 0
			start_epoch = 0
			start_step = 0
			print("[*] No pretrained model found!")
		# make summary
		tf.summary.scalar('loss', self.loss)
		tf.summary.scalar('lr', self.lr)
		writer = tf.summary.FileWriter('./logs', self.sess.graph)
		merged = tf.summary.merge_all()
		print("[*] Start training, with start epoch %d start iter %d : " % (start_epoch, iter_num))
		start_time = time.time()
		for epoch in range(start_epoch, epoch):
			shuffle_in_unison(data,data_noisy)
			for batch_id in range(start_step, numBatch):
				batch_images = data[batch_id * batch_size:(batch_id + 1) * batch_size, :, :, :]
				batch_images = batch_images.astype(np.float32) / 255.0 # normalize the data to 0-1
				batch_noisy = data_noisy[batch_id * batch_size:(batch_id + 1) * batch_size, :, :, :]
				batch_noisy = batch_noisy.astype(np.float32) / 255.0 # normalize the data to 0-1
				
				_, loss, summary = self.sess.run([self.train_op, self.loss, merged],
												 feed_dict={self.Y_: batch_images, self.X: batch_noisy, self.lr: lr[epoch],
															self.is_training: True})
				print("Epoch: [%2d] [%4d/%4d] time: %4.4f, loss: %.6f"
					  % (epoch + 1, batch_id + 1, numBatch, time.time() - start_time, loss))
				iter_num += 1
				writer.add_summary(summary, iter_num)
			self.save(iter_num, ckpt_dir)
		print("[*] Finish training.")

	def save(self, iter_num, ckpt_dir, model_name='Temp3-CNN'):
		saver = tf.train.Saver()
		checkpoint_dir = ckpt_dir
		if not os.path.exists(checkpoint_dir):
			os.makedirs(checkpoint_dir)
		print("[*] Saving model...")
		saver.save(self.sess,os.path.join(checkpoint_dir, model_name),global_step=iter_num)

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

	def test(self, noisy_total, eval_data, ckpt_dir, save_dir):
		"""Test Temp3-CNN"""
		# init variables
		tf.global_variables_initializer().run()
		assert len(eval_data) != 0, 'No testing data!'
		load_model_status, global_step = self.load(ckpt_dir)
		assert load_model_status == True, '[!] Load weights FAILED...'
		print(" [*] Load weights SUCCESS...")
		psnr_sum = 0
		ind = np.multiply(range(len(eval_data)/3),3)

		start = time.time()
		for idx in range(len(eval_data)):
			if idx>(len(eval_data)-3):
				test = cv2.imread(eval_data[idx])
				noisy = cv2.imread(noisy_total[idx])
				psnr = cal_psnr(test,noisy)
				psnr_sum += psnr
				print("img%d PSNR: %.2f %.2f" % (idx, psnr, psnr))
				_, path = eval_data[idx].split('al')
#				cv2.imwrite(('./data/denoised/%s' %path), noisy)
				print 'last frames'
				continue
			test = cv2.imread(eval_data[idx])
			test1 = cv2.imread(eval_data[idx+1])
			test2 = cv2.imread(eval_data[idx+2])
			noisy = cv2.imread(noisy_total[idx])
			noisy1 = cv2.imread(noisy_total[idx+1])
			noisy2 = cv2.imread(noisy_total[idx+2])
			if idx==0:
				psnr = cal_psnr(test,noisy)
				psnr_sum += psnr
				_, path = eval_data[idx].split('al')
#				cv2.imwrite(('./data/denoised/%s' %path), noisy)

			test = test.astype(np.float32) / 255.0
			test1 = test1.astype(np.float32) / 255.0
			test2 = test2.astype(np.float32) / 255.0
			noisy = noisy.astype(np.float32) / 255.0
			noisy1 = noisy1.astype(np.float32) / 255.0
			noisy2 = noisy2.astype(np.float32) / 255.0
			
			orig = np.zeros((1,test.shape[0],test.shape[1],9))
			noisyin = np.zeros((1,test.shape[0],test.shape[1],9))
			
			orig[0,:,:,0:3] = test
			orig[0,:,:,3:6] = test1
			orig[0,:,:,6:] = test2
			noisyin[0,:,:,0:3] = noisy
			noisyin[0,:,:,3:6] = noisy1
			noisyin[0,:,:,6:] = noisy2

			output_clean_image= self.sess.run(
				[self.Y],feed_dict={self.Y_:orig,self.X:noisyin,self.is_training: False})
			
			out = np.asarray(output_clean_image)
#			cv2.imshow('',out[0,0])
#			cv2.waitKey(0)
			# calculate PSNR
			psnr = psnr_scaled(test1, out[0,0])
			psnr1 = psnr_scaled(test1, noisy1)
			print("img%d PSNR: %.2f %.2f" % (idx+1, psnr, psnr1))
			psnr_sum += psnr
			_, path = eval_data[idx+1].split('al')
#			cv2.imwrite(('./data/denoised/%s' %path), out[0,0]*255)
#			cv2.imwrite(('./data/original_frames/%s' %path), test1*255)
		avg_psnr = psnr_sum / len(eval_data)
		print("--- Average PSNR %.2f ---" % avg_psnr)
		print("--- Elapsed time: %.4f" %(time.time()-start))

def cal_psnr(im1, im2): # PSNR function for 0-255 values
	mse = ((im1.astype(np.float) - im2.astype(np.float)) ** 2).mean()
	psnr = 10 * np.log10(255 ** 2 / mse)
	return psnr
	
def psnr_scaled(im1, im2): # PSNR function for 0-1 values
	mse = ((im1 - im2) ** 2).mean()
	mse = mse * (255 ** 2)
	psnr = 10 * np.log10(255 **2 / mse)
	return psnr
