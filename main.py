import argparse
from glob import glob
import os
import tensorflow as tf
from model_videnn import videnn

parser = argparse.ArgumentParser(description='')
parser.add_argument('--use_gpu', dest='use_gpu', type=int, default=1, help='set it to 0 for CPU')
parser.add_argument('--ckpt_dir', dest='ckpt_dir', default='./ckpt_videnn', help='checkpoint directory')
parser.add_argument('--save_dir', dest='save_dir', default='./data/denoised', help='denoised sample are saved here')
parser.add_argument('--test_dir', dest='test_dir', default='./data/test', help='folder containing original and noisy folders to be denoised')
parser.add_argument('--frame_fmt', dest='frame_fmt', default='.png', help='frame extension, png is the default')
args = parser.parse_args()
     
def videnn_test(videnn):
    noisy_files = glob(args.test_dir + '/noisy/*' + args.frame_fmt)
    noisy_files = sorted(noisy_files)
    orig_files = glob(args.test_dir + '/original/*' + args.frame_fmt)
    if not os.path.exists(args.test_dir + '/original'): ##check if the original folder exists, otherwise we use the noisy image for PSNR calculation
        print('[!] Folder with original files not found, PSNR values will be WRONG!')        
        orig_files=noisy_files
    orig_files = sorted(orig_files)
    videnn.test(noisy_files, orig_files, ckpt_dir=args.ckpt_dir, save_dir='./data/denoised')

def main(_):
    if not os.path.exists(args.ckpt_dir):
        os.makedirs(args.ckpt_dir)
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    if not os.path.exists(args.test_dir):
        os.makedirs(args.test_dir)

    if args.use_gpu:
        print("GPU\n")
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.6)
        with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
            model = videnn(sess)
            videnn_test(model)
    else:
        print("CPU\n")
        with tf.device('/cpu:0'):
            with tf.Session() as sess:
                model = videnn(sess)
                videnn_test(model)


if __name__ == '__main__':
    tf.app.run()
