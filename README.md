[![AUR](https://img.shields.io/aur/license/yaourt.svg?style=plastic)](LICENSE)

# VidCNN - Learning Blind Video Denoising

![](./img/vidcnn.png)

This repository contains my master thesis project called VidCNN - Learning Blind Video Denoising.
[The paper](https://github.com/clausmichele/VidCNN---Learning-Blind-Video-Denoising/blob/master/VidCNN_MICHELE_CLAUS.pdf) explains the network architecture and the chioces I made during the design and testing process.
The provided code is for testing purposes, I have not included the training part yet.

# Introduction

With this pretrained tensorflow model you will be able to denoise videos affected by different types of degradation, such as Additive White Gaussian Noise and videos in Low-Light conditions. The latter has been tested only on one particular camera raw data, so it might not work on different sources. VidCNN works in blind conditions, it does not require any information over the content of the input noisy video.

![](./img/tennis_gauss.png)


# Architecture

VidCNN is a fully convolutional neural network and can denoise all different sizes of video, depending on the available memory on your machine.

# Requirements
```
tensorflow >= 1.4 (tested on 1.4 and 1.9)
numpy
opencv
ffmpeg
```

# How to denoise my own video?

