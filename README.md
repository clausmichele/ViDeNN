# ViDeNN: Deep Blind Video Denoising

![](./img/ViDeNN.png)

This repository contains my master thesis project called ViDeNN - Deep Blind Video Denoising. 
The provided code is for testing purposes, I have not included the training part yet nor the self-recorded test videos.

# Paper
ArXiv: https://arxiv.org/abs/1904.10898

CVPR 2019: http://openaccess.thecvf.com/content_CVPRW_2019/papers/NTIRE/Claus_ViDeNN_Deep_Blind_Video_Denoising_CVPRW_2019_paper.pdf

# Introduction

With this pretrained tensorflow model you will be able to denoise videos affected by different types of degradation, such as Additive White Gaussian Noise and videos in Low-Light conditions. The latter has been tested only on one particular camera raw data, so it might not work on different sources. ViDeNN works in blind conditions, it does not require any information over the content of the input noisy video.
IMPORTANT! If you want to denoise data affected by Gaussian noise (AWGN), it has to be clipped between 0 and 255 before denoising it, otherwise you will get strange artifacts in your denoised output.

![](./img/tennis_gauss.png)


# Architecture

ViDeNN is a fully convolutional neural network and can denoise all different sizes of video, depending on the available memory on your machine.

# Requirements
```
tensorflow >= 1.4 (tested on 1.4 and 1.9)
numpy
opencv
ffmpeg
```

# How to denoise my own video?

Important: the network has not been trained for general-purpose denoising of compressed videos. If the output includes some artifacts try to use the other checkpoint, modifying the last line of the script with --ckpt_dir='./ckpt_vidcnn-g'.
If you have a noisy video file, you can use the script calling it in a terminal:
```
$ sh denoise.sh
```
It will first extract all the frames using FFmpeg and then start ViDeNN to perform blind video denoising.

# Issues?

Feel free to open an issue if you have any problem, I will do my best to help you.
