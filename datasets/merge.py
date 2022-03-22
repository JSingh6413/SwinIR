#! /usr/bin/env python3

import os
import shutil
import argparse

from common import get_filelist, enumerate_filenames
from common import IMG_EXTS


def merge_datasets(input_dirs: list, output_dir: str, target_ext=None, extensions=IMG_EXTS):
    idx_shift = 0

    # cleanup
    shutil.rmtree(output_dir,  ignore_errors=True)
    os.makedirs(output_dir)

    # merge
    for input_dir in input_dirs:
        filelist = get_filelist(input_dir, extensions)
        out_filelist = enumerate_filenames(
            filelist, target_ext, start_idx=idx_shift
        )

        for src, dst in zip(filelist, out_filelist):
            shutil.copyfile(src, os.path.join(output_dir, dst))

        idx_shift += len(filelist)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Merge multiple datasets')
    parser.add_argument(
        'input_dirs', nargs='+', type=str,
        help='paths to datasets: "path_1" "path_2" ... "path_n"'
    )
    parser.add_argument(
        '-o', '--output_dir', type=str, default='merged',
        help='path to the directory for the outputs'
    )
    parser.add_argument(
        '-e', '--target_ext', default=None, type=str,
        help='target extenstion for processed images'
    )
    parser.add_argument(
        '--extensions', nargs='+', type=str, default=IMG_EXTS,
        help='files extensions to filter: ".ext_1" ".ext_2" ... ".ext_n"'
    )
    args = parser.parse_args()

    merge_datasets(
        args.input_dirs, args.output_dir,
        args.target_ext, args.extensions
    )
