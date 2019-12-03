#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
@author: clausmichele
"""
import argparse
import numpy as np
import cv2
from glob import glob
import os
from tqdm import tqdm

parser = argparse.ArgumentParser(description='')
parser.add_argument('--download_videos', dest='download_videos', type=int, default=1, help='Set it to False if you have downloaded the videos')
args = parser.parse_args()

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

	if not os.path.isdir("./data"):
		os.mkdir("./data")
	if not os.path.isdir("./data/train"):
		os.mkdir("./data/train")
	if not os.path.isdir("./data/train/original"):
		os.mkdir("./data/train/original")
	if not os.path.isdir("./data/train/noisy"):
		os.mkdir("./data/train/noisy")
	if not os.path.isdir("./data/train/denoised"):
		os.mkdir("./data/train/denoised")
	if not os.path.isdir("./data/test"):
		os.mkdir("./data/test")
	if not os.path.isdir("./data/test/original"):
		os.mkdir("./data/test/original")
	if not os.path.isdir("./data/test/noisy"):
		os.mkdir("./data/test/noisy")

	# Download Videos
	download_videos = args.download_videos
	if download_videos:
		videos = ["akiyo_cif.y4m","bowing_cif.y4m","bridge_close_cif.y4m",
			   "bridge_far_cif.y4m","bus_cif.y4m","city_cif.y4m","coastguard_cif.y4m",
			   "container_cif.y4m","crew_cif.y4m","deadline_cif.y4m","flower_cif.y4m",
			   "flower_garden_422_cif.y4m","football_422_cif.y4m","football_cif.y4m",
			   "galleon_422_cif.y4m"]
	
		root_link = "https://media.xiph.org/video/derf/y4m/"
		for video in tqdm(videos,desc="[*] Downloading videos..."):
			os.system("wget -q --no-check-certificate " + root_link + video)
		os.system("mv *.y4m ./data")

	videos = glob("./data/*.y4m")
	num_vids = len(videos)
	for i in tqdm(range(int(num_vids)),desc="[*] Extracting frames..."):
		if not os.path.isdir("./data/train/original/" + str(i)):
			os.mkdir("./data/train/original/" + str(i))
		if not os.path.isdir("./data/train/noisy/" + str(i)):
			os.mkdir("./data/train/noisy/" + str(i))
		if not os.path.isdir("./data/train/denoised/" + str(i)):
			os.mkdir("./data/train/denoised/" + str(i))
		os.system("ffmpeg -loglevel quiet -i " + videos[i] + " ./data/train/original/" + str(i) + "/%05d.png")
		frames = glob("./data/train/original/" + str(i) + "/*.png")
		frames = sorted(frames)
		while(len(frames)>300): # Max 300 frames each video, you can set it depending on your ram size
			os.remove(frames[-1])
			frames.pop()
		if len(frames)%3==2:
			os.remove(frames[-1])
			frames.pop()
		if len(frames)%3==1: # Making the frames number divisible by 3, necessary for training
			os.remove(frames[-1])
			frames.pop()
	imgs_path_train = sorted(glob("./data/train/original/*/*.png"))
	num_of_samples = int(len(imgs_path_train)/3)

	sigma_train = np.linspace(0,50,num_of_samples+1)
	np.random.shuffle(sigma_train)
	for i in tqdm(range(num_of_samples),desc="[*] Creating original-noisy set..."):
		sigma = sigma_train[i]
		img_path = imgs_path_train[i*3]
		folder = img_path.split('/')[-2]

		img_file = os.path.basename(img_path)
		img_original = cv2.imread(img_path)
		img_noisy = gaussian_noise(sigma,img_original)
		cv2.imwrite("./data/train/noisy/"+folder+"/"+img_file,img_noisy)
		
		img_path = imgs_path_train[i*3+1]
		img_file = os.path.basename(img_path)
		img_original = cv2.imread(img_path)
		img_noisy = gaussian_noise(sigma,img_original)
		cv2.imwrite("./data/train/noisy/"+folder+"/"+img_file,img_noisy)

		img_path = imgs_path_train[i*3+2]
		img_file = os.path.basename(img_path)
		img_original = cv2.imread(img_path)
		img_noisy = gaussian_noise(sigma,img_original)
		cv2.imwrite("./data/train/noisy/"+folder+"/"+img_file,img_noisy)
