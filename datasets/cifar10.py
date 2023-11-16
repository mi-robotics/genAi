import csv
import numpy as np
import os
import PIL
import re
import torch
from collections import namedtuple
from torch.utils.data import DataLoader, Subset, Sampler
from torch.utils.data.distributed import DistributedSampler
from torchvision import transforms, datasets as tvds


class CIFAR10(tvds.CIFAR10):
    resolution = (32, 32)
    channels = 3
    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        # transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    # _transform = transforms.PILToTensor() #TODO i think this can be removed
    train_size = 50000
    test_size = 10000

    def __init__(self, root, mode="train", transform=None):
        super().__init__(root=root, train=mode != "test", transform=transform or self.transform, download=True)

    def __getitem__(self, index):
        return super().__getitem__(index)[0]