import shutil
import os
import argparse


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='deblur images')
    parser.add_argument(
        '-m', '--method',
        type=str,
        help='ista or fista',
        required=True,
    )
    parser.add_argument(
        '-i', '--input',
        type=str,
        help='path to the directory with dataset',
        required=True,
    )
    parser.add_argument(
        '-o', '--output_dir',
        type=str,
        help='output_dir',
        required=True,
    )
    args = parser.parse_args()

    # process noisy_swin:
    os.system(
        f"python FISTA/deblur_noisy1.py -m {args.method} -nimg 5"
        + f" -a 2e-3 -eps 75 -p {'./data/blur80/'} -i {args.input} -o {args.output_dir}"
    )
