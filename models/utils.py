import torch
import numpy as np

from tqdm import tqdm


def train_loop(model, dataloader, loss_fn, optimizer, device='cpu'):

    losses = []
    for x, y in tqdm(dataloader, desc="\tBatch #", ncols=80):
        if isinstance(x, tuple):
            x = (x[0].to(device), x[1].to(device))
            y = y.to(device)
        else:
            x, y = x.to(device), y.to(device)

        # evaluate
        loss = loss_fn(model(x), y) / len(x)

        # backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses.append(loss.item())

    return np.mean(losses)


def test_loop(model, dataloader, loss_fn, device='cpu'):

    loss = 0
    with torch.no_grad():
        for x, y in tqdm(dataloader, desc="\tBatch #", ncols=80):
            if isinstance(x, list):
                x = (x[0].to(device), x[1].to(device))
                y = y.to(device)
            else:
                x, y = x.to(device), y.to(device)
            loss += loss_fn(model(x), y).item()

    return loss / len(dataloader.dataset)


def charbonnier_loss(output, target, eps=1e-3):
    return (torch.norm(target - output) ** 2 + eps ** 2) ** 0.5


def save_model(model, path):
    torch.save(model, path)


def load_model(path):
    model = torch.load(path)
    model.eval()
    return model
