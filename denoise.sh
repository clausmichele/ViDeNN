#!/bin/bash
echo ViDeNN - Deep Blind Video Denoising
echo Usage: sh denoise.sh video_file_path
if test ! -f "$1"; then
    echo "[!] $1 does not exist!"
    exit 0
fi
DIR=./data
if [ ! -d "$DIR" ]; then
    mkdir data && mkdir data/noisy
fi
DIR=./data/noisy
if [ ! -d "$DIR" ]; then
    mkdir data/noisy
fi
DIR=./data/original
if [ -d "$DIR" ]; then
    rm -rf ./data/original
fi

echo [*] Extracting frames to the Noisy folder...
ffmpeg -nostats -loglevel 0 -i ./$1 ./data/noisy/%04d.png
echo [*] Denoising video...denoised frames stored in ./data/denoised/
python main_ViDeNN.py --use_gpu=1 --checkpoint_dir=ckpt_videnn --save_dir='./data/denoised' --test_dir='./data'
