#!/bin/sh
echo [*] ViDeNN: AWGN image dataset preparation script. Run it once before training. Author: Claus Michele

wget http://ivc.uwaterloo.ca/database/WaterlooExploration/exploration_database_and_code.rar
unrar x exploration_database_and_code.rar
mv exploration_database_and_code data
cd data
rm -rf result data support_functions demo.m ssim.m
mkdir train train/noisy train/original test test/noisy test/original
cd ..
python add_noise_spatialCNN.py
python generate_patches_spatialCNN.py
