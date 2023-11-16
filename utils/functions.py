import torch

def flat_mean(x, start_dim=1):
    reduce_dim = [i for i in range(start_dim, x.ndim)]
    return torch.mean(x, dim=reduce_dim)