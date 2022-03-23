import os
import re
import argparse

import numpy as np
import matplotlib.pyplot as plt
import pylops
from skimage.io import imread, imsave

def show_image(image, figsize=(3, 3)):
    plt.figure(figsize=figsize)
    plt.imshow(image, cmap=plt.cm.gray)
    plt.axis("tight")
    plt.axis('off'); 

def get_conv(image_shape, kernel, kernel_shape):
    return pylops.signalprocessing.Convolve2D(
        N=np.prod(image_shape),
        dims=image_shape,
        h=kernel,
        offset=(kernel_shape[0] // 2, kernel_shape[1] // 2),
        dtype="float32",
    )

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='deblur images')

    parser.add_argument(
        '-m', '--method', type=str,
        help='Optimization Method: ista or fista',
        required=True,
    )
    parser.add_argument(
        '-k', '--kernel', type=int,
        help='Kernel number: 0-7',
        required=True,
    )
    parser.add_argument(
        '-nimg', '--n_images', type=int,
        help='The number of images: from 1 up to 80. E.g. 50 gives the first 50 images',
        required=True,
    )
    parser.add_argument(
        '-a', '--alpha', type=float,
        help='Learning rate. If -1.0 then it is evaluated internally',
        required=True,
    )
    parser.add_argument(
        '-eps', '--epsilon', type=float,
        help='Regularization parameter: e.g. 50, 75, 100',
        required=True,
    )
    parser.add_argument(
        '-nit', '--n_iter', type=int,
        help='The number of iterations for optimizer: [100, 200]',
        required=True,
    )
    parser.add_argument(
        '-p', '--path', type=str,
        help='Path',
        required=True,
    )
    parser.add_argument(
        '-s', '--sigma', type=str,
        help='sigma id 0.01 or 0.05',
        required=True,
    )
    parser.add_argument(
        '-i', '--input',
        type=str,
        help='Input_dir',
        required=True,
    )
    parser.add_argument(
        '-o', '--output',
        type=str,
        help='Output_dir',
        required=True,
    )

    
    args = parser.parse_args()


    if args.method == "ista":
        method = pylops.optimization.sparsity.ISTA
    elif args.method == "fista":
        method = pylops.optimization.sparsity.FISTA
    

    save_to_path = args.path + f"{args.output}{args.sigma}/kernel_{args.kernel}/"
    get_from_path = args.path + f"{args.input}{args.sigma}/kernel_{args.kernel}/"
    img_names = [filename for filename in os.listdir(get_from_path)]

    kern_number = args.kernel
    n_img = args.n_images

    kernel_path = args.path + f"kernels/kernel_{kern_number}.png"
    kern = imread(kernel_path).astype(np.float32)
    kern = kern / 255
    
    alpha = args.alpha
    eps = args.epsilon
    n_iter = args.n_iter

    for i, ni in enumerate(img_names):
        img_path = get_from_path + ni
        blurred = imread(img_path).astype(np.float32)

        Cop = get_conv(blurred.shape, kern, kern.shape)
        Wop = pylops.signalprocessing.DWT2D(blurred.shape, wavelet="haar", level=3)

        imdeblurfista = method(
            Cop*Wop.H,
            blurred.ravel(),
            alpha=alpha,
            eps=eps,
            niter=n_iter,
            show=False,
            x0=Wop*blurred.ravel()
        )[0]

        imdeblurfista = Wop.H * imdeblurfista
        imdeblurfista = imdeblurfista.reshape(blurred.shape)
        imdeblurfista = (imdeblurfista / imdeblurfista.max() * 255).astype(np.uint8)
        imsave(save_to_path + ni, imdeblurfista)

    with open(save_to_path + "config.txt", "w") as f:
        f.write(f"alpha:{alpha}\neps:{eps}\nniter:{n_iter}")
    
