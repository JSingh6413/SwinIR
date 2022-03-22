#! /usr/bin/env python3

import os
import shutil
import random
import argparse

from common import get_filelist, enumerate_filenames


def split(input_dir, output_dir, shuffle=False, random_state=None, n_test=None, percentage=0.2):
    '''
    '''

    # cleanup
    shutil.rmtree(os.path.join(output_dir, 'train'),  ignore_errors=True)
    shutil.rmtree(os.path.join(output_dir, 'test'),  ignore_errors=True)

    os.makedirs(os.path.join(output_dir, 'train'))
    os.makedirs(os.path.join(output_dir, 'test'))

    # split data
    filelist = get_filelist(input_dir)

    if n_test is None:
        assert 0.0 < percentage < 1.0, "percentage must be between 0 and 1"
        n_test = round(len(filelist) * percentage)

    assert 0 < n_test < len(filelist), "n_test must be between 1" \
        "and the length of the dataset"

    if shuffle:
        if random_state is not None:
            random.seed(random_state)
        random.shuffle(filelist)

    train_set = enumerate_filenames(filelist[:-n_test])
    test_set = enumerate_filenames(filelist[-n_test:])

    for src, dst in zip(filelist[:-n_test], train_set):
        shutil.copyfile(src, os.path.join(output_dir, 'train', dst))

    for src, dst in zip(filelist[-n_test:], test_set):
        shutil.copyfile(src, os.path.join(output_dir, 'test', dst))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Split data into train and test'
    )
    parser.add_argument(
        'input_dir', type=str,
        help='path to input directory with images dataset'
    )
    parser.add_argument(
        'output_dir', type=str,
        help='path to the directory for the outputs'
    )
    parser.add_argument(
        '-p', '--percentage', type=float, default=0.2,
        help='percentage of test set'
    )
    parser.add_argument(
        '-n', '--n_test', type=int,
        help='number of items in test set: override precentage parameter'
    )
    parser.add_argument(
        '-s', '--shuffle', action='store_true', default=False,
        help='shuffle filenames'
    )
    parser.add_argument(
        '-r', '--random_state', default=None, type=int,
        help="random state: it's used only with -s"
    )
    args = parser.parse_args()

    split(
        args.input_dir, args.output_dir,
        args.shuffle, args.random_state,
        args.n_test, args.percentage
    )
