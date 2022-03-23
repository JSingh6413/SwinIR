#! /usr/bin/env python3

import torch
from torchvision.transforms import ToPILImage, PILToTensor

from datasets.common import get_filelist
from datasets.transforms import img_to_np, np_to_img

from models.swin import ProjectionSwinIR
from models.utils import load_model

import os
import shutil
import argparse
from tqdm import tqdm
from PIL import Image


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train network on dataset')
    parser.add_argument(
        'input_dir', type=str, default='data/noised',
        help='path to the directory with noisy images'
    )
    parser.add_argument(
        '-o', '--output_dir', type=str, default='results',
        help='path to the directory for resutls'
    )
    parser.add_argument(
        '-m', '--model_path', type=str, default='pretrained_models/projection.pt',
        help='path to the pretrained model'
    )
    parser.add_argument(
        '-s', '--std', type=float, default=0.05,
        help='noise standard deviation: used only with ProjectionSwinIR'
    )
    parser.add_argument(
        '--cpu_only', default=False, action='store_true',
        help='use only CPU for the evaluation'
    )
    args = parser.parse_args()

    # Choose device
    device = "cuda" if not args.cpu_only and torch.cuda.is_available() else "cpu"
    print(f"[torch] Using [{device.upper()}] device\n")

    # Load model
    model = load_model(args.model_path).to(device)

    # Load filelist
    filelist = get_filelist(args.input_dir)

    shutil.rmtree(args.output_dir,  ignore_errors=True)
    os.makedirs(args.output_dir)

    # Start evaluation
    def img_to_torch(img):
        img = np_to_img(img_to_np(img))
        return PILToTensor()(img)

    torch_to_img = ToPILImage()

    for filepath in tqdm(filelist, desc="Image #", ncols=80):
        filename = os.path.basename(filepath)
        x = (img_to_torch(Image.open(filepath))[None, ...] / 255.0).to(device)

        grayscale = (x.shape[1] == 1)

        if isinstance(model, ProjectionSwinIR):
            x = (x, args.std)

        with torch.no_grad():
            y = torch.squeeze(model(x))

        if grayscale:
            y = y[0] * 0.2989 + y[1] * 0.5870 + y[2] * 0.1140

        img = torch_to_img(y)
        img.save(os.path.join(args.output_dir, filename))
