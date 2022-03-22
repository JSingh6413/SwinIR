#! /usr/bin/env python3

import os
import shutil
import argparse
from functools import partial

import transforms
from common import get_filelist, enumerate_filenames
from common import save_images, load_images
from common import IMG_EXTS


def process_dataset(input_dir, output_dir, transform=None, target_ext=None, extensions=IMG_EXTS):
    '''
    '''

    # cleanup
    shutil.rmtree(output_dir,  ignore_errors=True)
    os.makedirs(output_dir)

    # process
    filelist = get_filelist(input_dir, extensions)
    out_filelist = [
        os.path.join(output_dir, filename)
        for filename in enumerate_filenames(filelist, target_ext)
    ]

    if target_ext is None and transform is None:
        # simply copy images with new names
        for src, dst in zip(filelist, out_filelist):
            shutil.copyfile(src, dst)
    else:
        # load images, tranform them and save with new names
        save_images(
            load_images(filelist, transform),
            out_filelist
        )


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Dataset preprocessor')
    parser.add_argument(
        'input_dir', type=str,
        help='path to input directory with images dataset'
    )
    parser.add_argument(
        'output_dir', type=str,
        help='path to the directory for the outputs'
    )
    parser.add_argument(
        '-r', '--rescale', default=None, type=float,
        help='rescale coefficient '
    )
    parser.add_argument(
        '-s', '--std', default=None, type=float,
        help='std for gaussian noise'
    )
    parser.add_argument(
        '-b', '--blur', default=False, action='store_true',
        help='add blur to image: default kernel is gaussian; can be override with --kernel_path'
    )
    parser.add_argument(
        '--kernel_size', default=3, type=int,
        help="gaussian kernel size: it's only used when blurring is enabled"
    )
    parser.add_argument(
        '--kernel_sigma', default=2, type=int,
        help="gaussian kernel sigma: it's only used when blurring is enabled"
    )
    parser.add_argument(
        '--kernel_path', default=None, type=str,
        help="path to custom blur kernel: it's only used when blurring is enabled"
    )
    parser.add_argument(
        '-e', '--target_ext', default=None, type=str,
        help='target extenstion for processed images'
    )
    parser.add_argument(
        '--extensions', nargs='+', type=str, default=IMG_EXTS,
        help='files extensions to filter: ".ext_1" ".ext_2" ... ".ext_n"'
    )
    parser.add_argument(
        '-t', '--transpose', default=False, action='store_true',
        help='rotate image to horizontal orientation'
    )
    args = parser.parse_args()

    transforms_list = []

    if args.transpose:
        transforms_list.append(transforms.to_horizontal)

    if args.rescale is not None:
        transforms_list.append(partial(transforms.rescale, scale=args.rescale))

    if args.blur:
        if args.kernel_path is None:
            kernel = transforms.gaussian_kernel(
                args.kernel_size, args.kernel_sigma
            )
        else:
            kernel = transforms.load_kernel(args.kernel_path)

        transforms_list.append(partial(transforms.blur, kernel=kernel))

    if args.std is not None:
        transforms_list.append(
            partial(transforms.gaussian_noise, std=args.std)
        )

    def transform(img):
        for func in transforms_list:
            img = func(img)
        return img

    process_dataset(
        args.input_dir, args.output_dir,
        transform=transform if transforms_list else None,
        target_ext=args.target_ext, extensions=args.extensions
    )
