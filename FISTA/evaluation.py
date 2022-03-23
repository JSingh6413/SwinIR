import argparse
import numpy as np
from skimage.io import imread
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
import metrics as mt


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='evaluation')
    parser.add_argument('-m', '--method', type=str, help='ista/fista')
    parser.add_argument('-in', '--input', type=str, help='input directory')
    parser.add_argument('-out', '--output', type=str, help='output directory')
    args = parser.parse_args()

    noises = ['0.01', '0.05']
    kernels = [f'kernel_{i}' for i in range(0, 8)]

    mt.compute_metrics(args.input, args.output)
    res = {}
    for noise in noises:
        for k in kernels:
            ssim_mean, psnr_mean = mt.compute_metrics(
                args.input,
                args.output + f"{noise}/kernel_{k}"
            )
        
                            
            res[f'{args.method}, {200}, {noise}, ssim_error']=np.round(ssim_mean, 4)
            res[f'{args.method}, {200}, {noise}, psnr_error']=np.round(psnr_mean, 2)

    with open(f"{args.output}res.txt", "w") as f:
        f.write(f'{args.output}res\n\n')
        for key, value in res.items():
            f.write(f'{key} : {value}\n')