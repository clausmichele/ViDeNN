# -*- coding: utf-8 -*-
"""
Created on Thu May 30 14:32:05 2019

@author: clausmichele
"""

import numpy as np
import cv2
from glob import glob
import os
from tqdm import tqdm

def gaussian_noise(sigma,image):
	gaussian = np.random.normal(0,sigma,image.shape)
	noisy_image = np.zeros(image.shape, np.float32)
	noisy_image = image + gaussian
	noisy_image = np.clip(noisy_image,0,255)
	noisy_image = noisy_image.astype(np.uint8)
	return noisy_image

def realistic_noise(Ag,Dg,image):
	CT1=1.25e-4
	CT2=1.11e-4
	Nsat=7480
	image = image/255.0
	M=np.sqrt( ((Ag*Dg)/(Nsat*image)+(Dg**2)*((Ag * CT1 + CT2)**2)))
	N = np.random.normal(0,1,image.shape)
	noisy_image = image + N*M
	cv2.normalize(noisy_image, noisy_image, 0, 1.0, cv2.NORM_MINMAX, dtype=-1)
	return noisy_image

if __name__=="__main__":

	imgs_path = glob("./data/pristine_images/*.bmp")
	num_of_samples = len(imgs_path)
	imgs_path_train = imgs_path[:int(num_of_samples*0.7)]
	imgs_path_test = imgs_path[int(num_of_samples*0.7):]

	sigma_train = np.linspace(0,50,int(num_of_samples*0.7)+1)
	for i in tqdm(range(int(num_of_samples*0.7)),desc="[*] Creating original-noisy train set..."):
		img_path = imgs_path_train[i]
		img_file = os.path.basename(img_path).split('.bmp')[0]
		sigma = sigma_train[i]
		img_original = cv2.imread(img_path)
		img_noisy = gaussian_noise(sigma,img_original)

		cv2.imwrite("./data/train/noisy/"+img_file+".png",img_noisy)
		cv2.imwrite("./data/train/original/"+img_file+".png",img_original)

	for i in tqdm(range(int(num_of_samples*0.3)),desc="[*] Creating original-noisy test set..."):
		img_path = imgs_path_test[i]
		img_file = os.path.basename(img_path).split('.bmp')[0]
		sigma = np.random.randint(0,50)

		img_original = cv2.imread(img_path)
		img_noisy = gaussian_noise(sigma,img_original)

		cv2.imwrite("./data/test/noisy/"+img_file+".png",img_noisy)
		cv2.imwrite("./data/test/original/"+img_file+".png",img_original)

