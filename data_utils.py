import torch
import torchvision.datasets as datasets
from torch.utils.data import DataLoader, random_split, TensorDataset
from typing import List

def normalize(images: torch.Tensor) -> torch.Tensor:
    """
    Normalize images between zero and 1.

    :param images: images to normalize.
    :return: normalized images.

    """
    max, _ = torch.max(images.reshape(-1, 16 * 16), dim=1)
    min, _ = torch.min(images.reshape(-1, 16 * 16), dim=1)

    min = min.unsqueeze(1).unsqueeze(2)
    maxes = max.unsqueeze(1).unsqueeze(2)

    return (images - min) / (maxes - min)


def mnist_preparation(dataset: datasets.MNIST, labels: List[int], train_test_ratio: float, batch_size: int):
    """
    Preprocess MNIST dataset normalizing images and selecting only the ones corresponding to the indicated labels.

    :param dataset: mnist dataset.
    :param labels:
    :param train_test_ratio: train/test split.
    :param batch_size: batch size of the dataloader.
    :return: Train and Test dataloader with the requested labels.
    """

    data = []
    targets = []
    for image, label in dataset:
        if label in labels:
            data.append(image.squeeze())
            targets.append(label)

    data = torch.stack(data)
    data = normalize(data)
    data = data.flatten(start_dim=1)
    targets = torch.tensor(targets)
    targets = targets.type(torch.float32)
    dataset = TensorDataset(data, targets)

    train_size = int(train_test_ratio * len(dataset))
    test_size = len(dataset) - train_size

    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    return train_dataloader, test_dataloader