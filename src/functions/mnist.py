#!/usr/bin/env python
# coding: utf-8

import os

from torchvision import datasets
from torch.utils.data import DataLoader


def get_dataloader(path, transform, train=True):
    os.makedirs(path, exist_ok=True)
    train_dataset = datasets.MNIST(
        root=path, 
        train=train, 
        download=True, 
        transform=transform
    )
    dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    return dataloader