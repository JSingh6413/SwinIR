import torch
import numpy as np

from tqdm import tqdm


def train_loop(model, dataloader, loss_fn, optimizer, device='cpu'):

    losses = []
    for X, y in tqdm(dataloader, desc="\tBatch #", ncols=80):
        X, y = X.to(device), y.to(device)

        # evaluate
        pred = model(X)
        loss = loss_fn(pred, y)

        # backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses.append(loss.item())

    return np.mean(losses)


def test_loop(model, dataloader, loss_fn, device='cpu'):

    losses = []
    with torch.no_grad():
        for X, y in tqdm(dataloader, desc="\tBatch #", ncols=80):

            X, y = X.to(device), y.to(device)

            pred = model(X)
            losses.append(loss_fn(pred, y).item())

    return np.mean(losses)


def charbonnier_loss(output, target, eps=1e-3):
    return (torch.norm(target - output) ** 2 + eps ** 2) ** 0.5
