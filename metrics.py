#! /usr/bin/env python3


import os
import argparse
import numpy as np
from PIL import Image

from skimage.io import imread
from skimage import img_as_float
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr


def compute_metrics(gt_dir, noise_dir):
    gt_files = os.listdir(gt_dir)
    noise_files = os.listdir(noise_dir)

    common_filenames = set(noise_files).intersection(set(gt_files))

    ssim_values = []
    psnr_values = []

    for filename in common_filenames:
        gt_path = os.path.join(gt_dir, filename)
        noise_path = os.path.join(noise_dir, filename)

        gt_img = img_as_float(Image.open(gt_path))
        noise_img = img_as_float(Image.open(noise_path))

        ssim_value = ssim(gt_img, noise_img, multichannel=True)
        psnr_value = psnr(gt_img, noise_img)

        ssim_values.append(ssim_value)
        psnr_values.append(psnr_value)

    ssim_mean = np.mean(ssim_values)
    psnr_mean = np.mean(psnr_values)

    print(f'SSIM: {ssim_mean:.4f}\t|\PSNR: {psnr_mean:.4f}')

    return ssim_mean, psnr_mean


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Metrics between datasets')
    parser.add_argument(
        'gt_dir', type=str,
        help='path to the directory with ground truth images'
    )
    parser.add_argument(
        'corr_dir', type=str,
        help='path to the directory with corrupted images'
    )
    args = parser.parse_args()

    compute_metrics(args.gt_dir, args.corr_dir)
