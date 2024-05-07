import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import numpy as np
import os
from typing import Optional, Dict, List

class MNISTDataloaderPreparation:
    def __init__(self, data_directory: str = './data', size=int):
        print('Loading MNIST dataset...')
        self.mnist = datasets.MNIST(root=data_directory, train=True, download=True,
                                transform=transforms.Compose([
                                    transforms.Resize((size, size)),  # Resize to 16x16
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.1307,), (0.3081,))  # Normalize
                                ]))

    def filter_by(self, labels: List):
        _data = list()
        _targets = list()
        for image, label in self.mnist:
            if label in labels:
                _data.append(image.squeeze())
                _targets.append(label)
        _data = torch.stack(_data)
        _targets = torch.tensor(_targets)

import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import numpy as np
import os

# Download MNIST and prepare transforms
mnist_train = datasets.MNIST(root='./data', train=True, download=True,
                             transform=transforms.Compose([
                                transforms.Resize((16, 16)),  # Resize to 16x16
                                transforms.ToTensor(),
                                transforms.Normalize((0.1307,), (0.3081,))  # Normalize
                             ]))

# Filter for zeros and ones
data = []
targets = []
for image, label in self.mnist:
    if label in [0, 1]:
        data.append(image.squeeze())
        targets.append(label)

data = torch.stack(data)
targets = torch.tensor(targets)

# Select 1024 zeros and 1024 ones for speed
zeros_indices = (targets == 0)
ones_indices = (targets == 1)

zeros = data[zeros_indices]
ones = data[ones_indices]

# take a subsample of the dataset for simplicity
zeros = zeros[:1024]
ones = ones[:1024]

zeros_max = torch.max(zeros.reshape(-1, 16 * 16), dim=1)
zeros_min = torch.min(zeros.reshape(-1, 16 * 16), dim=1)
ones_max = torch.max(ones.reshape(-1, 16 * 16), dim=1)
ones_min = torch.min(ones.reshape(-1, 16 * 16), dim=1)


def normalize(imgs):
    maxes, _ = torch.max(imgs.reshape(-1, 16 * 16), dim=1)
    mins, _ = torch.min(imgs.reshape(-1, 16 * 16), dim=1)

    mins = mins.unsqueeze(1).unsqueeze(2)
    maxes = maxes.unsqueeze(1).unsqueeze(2)

    return (imgs - mins) / (maxes - mins)


zeros = normalize(zeros)
ones = normalize(ones)

assert torch.allclose(zeros.min(), torch.tensor(0., dtype=torch.float32), atol=1e-5)
assert torch.allclose(zeros.max(), torch.tensor(1., dtype=torch.float32), atol=1e-5)
assert torch.allclose(ones.min(), torch.tensor(0., dtype=torch.float32), atol=1e-5)
assert torch.allclose(ones.max(), torch.tensor(1., dtype=torch.float32), atol=1e-5)

# concatenate the two datasets
zeros = zeros.flatten(start_dim=1)
ones = ones.flatten(start_dim=1)
dataset = torch.cat((zeros, ones), dim=0)

# add labels
labels = torch.cat((torch.zeros((zeros.shape[0], 1)), torch.ones((ones.shape[0], 1))), dim=0).squeeze()

# build dataloader
dataset = torch.utils.data.TensorDataset(dataset, labels)
