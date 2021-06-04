# ViDeNN: Deep Blind Video Denoising #

<img src="./img/ViDeNN.png" align="center" width="480">

This repository contains my master thesis project called ViDeNN - Deep Blind Video Denoising. 

## Paper ##
<img src="http://cvpr2019.thecvf.com/images/CVPRLogo.png" align="left" width="256">
 http://openaccess.thecvf.com/content_CVPRW_2019/papers/NTIRE/Claus_ViDeNN_Deep_Blind_Video_Denoising_CVPRW_2019_paper.pdf


## Introduction ##

With this pretrained tensorflow model you will be able to denoise videos affected by different types of degradation, such as Additive White Gaussian Noise and videos in Low-Light conditions. The latter has been tested only on one particular camera raw data, so it might not work on different sources. ViDeNN works in blind conditions, it does not require any information over the content of the input noisy video.
IMPORTANT! If you want to denoise data affected by Gaussian noise (AWGN), it has to be clipped between 0 and 255 before denoising it, otherwise you will get strange artifacts in your denoised output.

![](./img/tennis_gauss_new.png)


## Architecture ##

ViDeNN is a fully convolutional neural network and can denoise all different sizes of video, depending on the available memory on your machine.

## Requirements ##
```
from apt:
ffmpeg
unrar

from pip:
tensorflow-gpu >= 1.4 (tested on 1.4 and 1.14)
numpy
opencv
tqdm
```
You need at least 25GB of free disk space for training.
The code has been tested with Python 2.7.
Edit : As of 20th May, this code works with Python 3 as well.

## Windows Users ##
Please check this issue https://github.com/clausmichele/ViDeNN/issues/19 if you have problems with the code. The code has been developed and tested on Linux/Ubuntu, but Windows has a different path structure, using \\ instead of /.

## Training ##

The training process is divided in two steps (see the paper for more details)

1. Spatial-CNN training
```
 $ cd Spatial-CNN
 $ sh dataset_preparation.sh
```
 - The dataset preparation script downloads the image dataset and prepare the correct folder structure, running it once is enough.
```
 $ python add_noise_spatialCNN.py
```
 - This script creates clean-noisy samples adding noise. Currently AWGN noise is implemented. (There is also realistic_noise function if you want to play with it).
```
 $ python generate_patches_spatialCNN.py
```
 - This script generates the 50x50 clean-noisy patches which will be used for training. They will be stored in two big binary files, which will be loaded in your ram during training.
```
 $ python main_spatialCNN.py
```
 - This starts training, storing a checkpoint every epoch. Check the available arguments with ``` python main_spatialCNN.py -h ``` if you want to set the number of epochs, learning rate, batch size etc.
If you want to fine tune the pre-trained AWGN model use:
```
 python main_spatialCNN.py --checkpoint_dir=ckpt_awgn --epoch=500
```
2. Temp3-CNN training
```
 $ cd Temp3-CNN
 $ python add_noise_temp3-CNN.py --download_videos=1
```
 - This will automatically download the videos used for training. If you want to run it a second time just set ```--download_videos=0``` .
```
 $ cd ../Spatial-CNN
 $ python main_spatialCNN.py --phase=test_temp
```
 - This will spatially denoise the frames we previously generated, which will be used to train Temp3-CNN.
```
 $ cd ../Temp3-CNN
 $ python generate_patches_temp3CNN.py
```
 - This script generates triplets of 50x50 clean-noisy patches from 3 consecutive frames which will be used for temporal denoising training.
```
 $ python main_temp3CNN.py
```
 - This starts training, storing a checkpoint every epoch. Check the available arguments with ``` python main_spatialCNN.py -h ``` if you want to set the number of epochs, learning rate, batch size etc.

## Testing ##

Important: the network has not been trained for general-purpose denoising of compressed (h264, h265 ecc) videos. If the output includes some artifacts try to use the other checkpoint, modifying the last line of the script with --checkpoint_dir='./ckpt_vidcnn-g'.
The denoising process requires a lot of memory, if you don't have a GPU with enough memory available, try to set --use_gpu=0 and denoise using CPU, or downscale/crop the video.
If you have a noisy video file, you can use the script calling it in a terminal:
```
$ sh denoise.sh video_file_path
```
It will first extract all the frames using FFmpeg and then start ViDeNN to perform blind video denoising.

## Future development ##

- Dynamic loading of training data to overcome memory limitations.
- Porting to Keras and Tensorflow 2.0

## Issues? ##

Feel free to open an issue if you have any problem, I will do my best to help you.
