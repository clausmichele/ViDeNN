#!/bin/bash
echo ViDeNN - Deep Blind Video Denoising
echo [*] Insert the video file name you want to denoise:
read vidname
mkdir noisy
echo [*] Extracting frames to the Noisy folder...
ffmpeg -i ./$vidname ./noisy/%04d.png
echo [*] Denoising frames, will be stored in ./data/denoised/
python main.py --use_gpu=1 --ckpt_dir='./ckpt_videnn-g' --save_dir='./data/denoised' --test_dir='./'
