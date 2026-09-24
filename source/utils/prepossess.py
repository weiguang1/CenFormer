import torch
import numpy as np


def continus_mixup_data(*xs, y=None, alpha=1.0, device='cuda'):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
    batch_size = y.size()[0]
    index = torch.randperm(batch_size).to(device)
    new_xs = [lam * x + (1 - lam) * x[index, :] for x in xs]
    y = lam * y + (1-lam) * y[index]
    return *new_xs, y
