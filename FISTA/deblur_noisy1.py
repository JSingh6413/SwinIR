import os
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='deblur')

    parser.add_argument(
        '-m', '--method', type=str,
        help='Optimization Method: ista or fista',
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
        '-p', '--path', type=str,
        help='path',
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

    for kernel in range(0, 8):
        for n_iter in [200]:
            for sigma_id in ['0.01', '0.05']:
                comm_str = (
                    f"python ./fista/deblur_base1.py -m {args.method} -k {kernel} -nimg {args.n_images}" 
                    + f" -a {args.alpha} -eps {args.epsilon} -nit {n_iter}"
                    + f" -p {args.path} -s {sigma_id} -i {args.input} -o {args.output}"
                )
                os.system(comm_str)
            