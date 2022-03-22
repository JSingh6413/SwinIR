#! /usr/bin/env python3

import torch
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor
import numpy as np

from datasets.utils import NoisyDataset
from models.swin import BlindSwinIR, ProjectionSwinIR
from models.dummy import DummyNet
from models.utils import charbonnier_loss, train_loop, test_loop
from models.utils import save_model

import os
import argparse
from glob import glob
from time import time
from functools import partial


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train network on dataset')
    parser.add_argument(
        '-m', '--model', type=str, default='dummy',
        help='model to train: {"dummy",  "blind", "projection"} '
    )
    parser.add_argument(
        '-i', '--train_dir', type=str, default='data/train',
        help='path to the training dataset'
    )
    parser.add_argument(
        '-t', '--test_dir', type=str, default='data/test',
        help='path to the testing dataset: can be disabled with --train_only'
    )
    parser.add_argument(
        '-r', '--results_dir', default='pretrained_models',
        help='path to the directory with pretrained models'
    )
    parser.add_argument(
        '-e', '--epochs', type=int, default=100,
        help='number of training epochs'
    )
    parser.add_argument(
        '--train_only', default=False, action='store_true',
        help='disable test while trainig model'
    )
    parser.add_argument(
        '-b', '--batch_size', type=int, default=4,
        help='batch_size for training'
    )
    args = parser.parse_args()

    models = {
        'dummy': DummyNet,
        'blind': BlindSwinIR,
        'projection': ProjectionSwinIR
    }

    assert args.model in models, "Model must be in list! -h for help"

    # Choose device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[torch] Using [{device.upper()}] device\n")

    # Choose model
    model = models[args.model]().to(device)
    optimizer = torch.optim.Adam(model.parameters())
    print(f'Using model: {args.model}\n')

    # Init directory for pretrained
    runs = glob(os.path.join(args.results_dir, args.model, '*'))
    if runs:
        current_run = max(int(os.path.basename(name)) for name in runs) + 1
    else:
        current_run = 0

    results_dir = os.path.join(
        args.results_dir, args.model,  f'{current_run:02}'
    )
    os.makedirs(results_dir, exist_ok=True)

    # DataLoaders
    print("Loading data...", end='', flush=True)
    start = time()
    NoisyDataset = partial(NoisyDataset, transform=ToTensor())
    if args.model == 'projection':
        NoisyDataset = partial(NoisyDataset, return_std=True)

    train_dataset = NoisyDataset(args.train_dir)
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True
    )

    if not args.train_only:
        test_dataset = NoisyDataset(args.test_dir)
        test_dataloader = DataLoader(
            test_dataset,
            batch_size=args.batch_size,
            shuffle=True
        )
    print(f'\t[done in {time()-start:4.4f}s]')

    print(f'\nTraining [{args.epochs}] epochs...\n')

    best_loss = np.inf
    for epoch in range(args.epochs):
        print(f"Epoch {epoch+1}:\n-------------------------------")
        train_loss = train_loop(
            model, train_dataloader,
            charbonnier_loss, optimizer, device=device
        )
        compare_loss = train_loss

        if not args.train_only:
            test_loss = test_loop(
                model, test_dataloader,
                charbonnier_loss, device=device
            )
            compare_loss = test_loss

        print(
            f'\tTrain loss: [{train_loss:.4f}]' +
            ('\n' if args.train_only else f'\t\tTest loss: [{test_loss:.4f}]\n')
        )

        if compare_loss < best_loss:
            best_loss = compare_loss
            path = os.path.join(
                results_dir, f'e[{epoch+1:03d}]_l[{best_loss:.4f}]'
            )
            save_model(model, path)
            print('\tMODEL SAVED\n')

    print(f'[DONE IN {time()-start:4.4f}s]')
