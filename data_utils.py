import torch
from torch.utils.data import DataLoader, random_split, TensorDataset
from torchvision import datasets, transforms


def download_and_prepare_mnist(labels_to_consider, batch_size, split_ratio=0.8):
    # Download MNIST and prepare transforms
    mnist_train = datasets.MNIST(root='./data', train=True, download=True,
                                 transform=transforms.Compose([
                                     transforms.Resize((16, 16)),  # Resize to 16x16
                                     transforms.ToTensor(),
                                     transforms.Normalize((0.1307,), (0.3081,))  # Normalize
                                 ]))

    # Filter for specified labels
    data, targets = [], []
    for image, label in mnist_train:
        if label in labels_to_consider:
            data.append(image.squeeze())
            targets.append(label)

    data = torch.stack(data)
    targets = torch.tensor(targets)

    # Normalize between 0 and 1
    def normalize(imgs):
        maxes, _ = torch.max(imgs.reshape(-1, 16 * 16), dim=1)
        mins, _ = torch.min(imgs.reshape(-1, 16 * 16), dim=1)

        mins = mins.unsqueeze(1).unsqueeze(2)
        maxes = maxes.unsqueeze(1).unsqueeze(2)

        return (imgs - mins) / (maxes - mins)

    data = normalize(data)

    # Assert images have min 0 and max 1 within an error of 1e-5
    assert torch.allclose(data.min(), torch.tensor(0., dtype=torch.float32), atol=1e-5)
    assert torch.allclose(data.max(), torch.tensor(1., dtype=torch.float32), atol=1e-5)

    # Flatten images and create labels tensor
    data = data.flatten(start_dim=1)
    labels = torch.tensor([labels_to_consider.index(label) for label in targets])

    # Build dataset and dataloaders
    dataset = TensorDataset(data, labels)

    # Train/test split
    train_size = int(split_ratio * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    # Dataloader
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    return train_dataloader, test_dataloader








