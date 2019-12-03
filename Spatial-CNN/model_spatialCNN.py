# -*- coding: utf-8 -*-
"""
@author: clausmichele
"""

import time
import tensorflow as tf
import numpy as np
from utilis import *

def SpatialCNN(input, is_training=True, output_channels=3, reuse=tf.AUTO_REUSE):
	with tf.variable_scope('block1',reuse=reuse):
		output = tf.layers.conv2d(input, 128, 3, padding='same', activation=tf.nn.relu)
	for layers in range(2, 20):
		with tf.variable_scope('block%d' % layers,reuse=reuse):
			output = tf.layers.conv2d(output, 64, 3, padding='same', name='conv%d' % layers, use_bias=False)
			output = tf.nn.relu(tf.layers.batch_normalization(output, training=is_training))
	with tf.variable_scope('block20', reuse=reuse):
		output = tf.layers.conv2d(output, output_channels, 3, padding='same', use_bias=False)
	return input - output

def shuffle_in_unison(a, b):
	rng_state = np.random.get_state()
	np.random.shuffle(a)
	np.random.set_state(rng_state)
	np.random.shuffle(b)

class denoiser(object):
	def __init__(self, sess, input_c_dim=3, batch_size=64):
		self.sess = sess
		self.input_c_dim = input_c_dim
		# build model
		self.Y_ = tf.placeholder(tf.float32, [None, None, None, self.input_c_dim], name='clean_image')
		self.is_training = tf.placeholder(tf.bool, name='is_training')
		self.X = tf.placeholder(tf.float32, [None, None, None, self.input_c_dim], name="noisy_image")
		self.Y = SpatialCNN(self.X, is_training=self.is_training)
		self.loss = (1.0 / batch_size) * tf.nn.l2_loss(self.Y_ - self.Y)
		self.lr = tf.placeholder(tf.float32, name='learning_rate')
		self.eva_psnr = tf_psnr(self.Y, self.Y_)
		optimizer = tf.train.AdamOptimizer(self.lr, name='AdamOptimizer')
		update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
		with tf.control_dependencies(update_ops):
			self.train_op = optimizer.minimize(self.loss)
		init = tf.global_variables_initializer()
		self.sess.run(init)
		print("[*] Initialize model successfully...")

	def evaluate(self, iter_num, eval_data_noisy, eval_data, summary_merged, summary_writer):
		print("[*] Evaluating...")
		psnr_sum = 0.0
		
		for idx in range(len(eval_data)):
			clean_image = eval_data[idx].astype(np.float32) / 255.0
			clean_image = clean_image[np.newaxis, ...]

			noisy = eval_data_noisy[idx].astype(np.float32) / 255.0
			noisy = noisy[np.newaxis, ...]
			
			output_clean_image, noisy_image, psnr_summary = self.sess.run(
				[self.Y, self.X, summary_merged],
				feed_dict={self.Y_: clean_image,
						   self.X: noisy,
						   self.is_training: False})
			groundtruth = np.clip(eval_data[idx], 0, 255).astype('uint8')
			noisyimage = np.clip(255 * noisy_image, 0, 255).astype('uint8')
			summary_writer.add_summary(psnr_summary, iter_num)
			outputimage = np.clip(255 * output_clean_image, 0, 255).astype('uint8')
			psnr = cal_psnr(groundtruth, outputimage)
			print("img%d PSNR: %.2f" % (idx, psnr))
			psnr_sum += psnr
		avg_psnr = psnr_sum / len(eval_data)

		print("--- Test ---- Average PSNR %.2f ---" % avg_psnr)


	def train(self, data, data_noisy, eval_data, eval_data_noisy, batch_size, ckpt_dir, epoch, lr, eval_every_epoch=1):
		numBatch = int(data.shape[0] / batch_size)
		# load pretrained model
		load_model_status, global_step = self.load(ckpt_dir)
		if load_model_status:
			iter_num = global_step
			start_epoch = global_step // numBatch
			start_step = global_step % numBatch
			print("[*] Model restore success!")
		else:
			iter_num = 0
			start_epoch = 0
			start_step = 0
			print("[*] Not find pretrained model!")
		# Summary
		tf.summary.scalar('loss', self.loss)
		tf.summary.scalar('lr', self.lr)
		writer = tf.summary.FileWriter('./logs', self.sess.graph)
		merged = tf.summary.merge_all()
		summary_psnr = tf.summary.scalar('eva_psnr', self.eva_psnr)
		print("[*] Start training, with start epoch %d start iter %d : " % (start_epoch, iter_num))
		start_time = time.time()
		self.evaluate(iter_num, eval_data_noisy, eval_data, summary_merged=summary_psnr,
					  summary_writer=writer)  # eval_data value range is 0-255
		for epoch in range(start_epoch, epoch):
			shuffle_in_unison(data,data_noisy)			
			for batch_id in range(0, numBatch):
				batch_images = data[batch_id * batch_size:(batch_id + 1) * batch_size, :, :, :]
				batch_images = batch_images.astype(np.float32) / 255.0 # normalize the data to 0-1
				batch_noisy = data_noisy[batch_id * batch_size:(batch_id + 1) * batch_size, :, :, :]
				batch_noisy = batch_noisy.astype(np.float32) / 255.0 # normalize the data to 0-1
				_, loss, summary = self.sess.run([self.train_op, self.loss, merged],
												 feed_dict={self.Y_: batch_images, self.X: batch_noisy, self.lr: lr[0],
															self.is_training: True})
				print("Epoch: [%2d] [%4d/%4d] time: %4.4f, loss: %.6f"
					  % (epoch + 1, batch_id + 1, numBatch, time.time() - start_time, loss))
				iter_num += 1
				writer.add_summary(summary, iter_num)
			self.evaluate(iter_num, eval_data_noisy, eval_data, summary_merged=summary_psnr,
							  summary_writer=writer)  # eval_data value range is 0-255
			self.save(iter_num, ckpt_dir)
		print("[*] Finish training.")


	def test(self, eval_data_noisy, eval_data, ckpt_dir, save_dir):
		"""Test Spatial-CNN"""
		# init variables
		tf.global_variables_initializer().run()

		assert len(eval_data) != 0, 'No testing data!'
		load_model_status, global_step = self.load(ckpt_dir)
		assert load_model_status == True, '[!] Load weights FAILED...'
		print(" [*] Load weights SUCCESS...")
		psnr_sum = 0.0
		start = time.time()
		for idx in range(len(eval_data)):
			test = cv2.imread(eval_data[idx])
			test1 = test.astype(np.float32) / 255.0
			test1 = test1[np.newaxis, ...]
			noisy = cv2.imread(eval_data_noisy[idx])
			noisy2 = noisy.astype(np.float32) / 255.0
			noisy2 = noisy2[np.newaxis, ...]
			output_clean_image= self.sess.run( [self.Y],feed_dict={self.X:noisy2,self.is_training: False})
			out2 = np.asarray(output_clean_image)
			psnr = psnr_scaled(test1[0], out2[0,0])
			psnr1 = psnr_scaled(test1[0], noisy2[0])
			print("img%d PSNR: %.2f %.2f" % (idx, psnr, psnr1))
			psnr_sum += psnr
			path = eval_data[idx].split('al')[-1]
			cv2.imwrite((save_dir + "/" + path), out2[0,0]*255)
		avg_psnr = psnr_sum / len(eval_data)
		print("--- Average PSNR %.2f ---" % avg_psnr)
		print("--- Elapsed time: %.4f" %(time.time()-start))

	def save(self, iter_num, ckpt_dir, model_name='Spatial-CNN'):
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

def tf_psnr(im1, im2): # PSNR function for tensors
	mse = tf.losses.mean_squared_error(labels=im2 * 255.0, predictions=im1 * 255.0)
	return 10.0 * (tf.log(255.0 ** 2 / mse) / tf.log(10.0))

def cal_psnr(im1, im2): # PSNR function for 0-255 values
	mse = ((im1.astype(np.float) - im2.astype(np.float)) ** 2).mean()
	psnr = 10 * np.log10(255 ** 2 / mse)
	return psnr
	
def psnr_scaled(im1, im2): # PSNR function for 0-1 values
	mse = ((im1 - im2) ** 2).mean()
	mse = mse * (255 ** 2)
	psnr = 10 * np.log10(255 **2 / mse)
	return psnr
