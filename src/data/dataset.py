"""Data loading and preprocessing utilities for MNIST dataset."""

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


def get_data_loaders(
    batch_size: int = 64,
    test_batch_size: int = 1000,
    use_cuda: bool = True,
    data_dir: str = "data",
) -> tuple[DataLoader, DataLoader]:
    """Create train and test data loaders for MNIST dataset.

    Args:
        batch_size: Batch size for training data
        test_batch_size: Batch size for test data
        use_cuda: Whether to use CUDA-specific DataLoader settings
        data_dir: Directory to store the dataset

    Returns:
        A tuple containing (train_loader, test_loader)
    """
    train_kwargs = {"batch_size": batch_size}
    test_kwargs = {"batch_size": test_batch_size}

    if use_cuda:
        cuda_kwargs = {"num_workers": 1, "pin_memory": True, "shuffle": True}
        train_kwargs.update(cuda_kwargs)
        test_kwargs.update(cuda_kwargs)

    # Define data transformations
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))])

    # Load datasets
    train_dataset = datasets.CIFAR10(data_dir, train=True, download=True, transform=transform)
    test_dataset = datasets.CIFAR10(data_dir, train=False, transform=transform)

    # Create data loaders
    train_loader = DataLoader(train_dataset, **train_kwargs)
    test_loader = DataLoader(test_dataset, **test_kwargs)

    return train_loader, test_loader
