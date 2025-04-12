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
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.4914, 0.4822, 0.4465),
            std=(0.247, 0.243, 0.261),
        ),
    ])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.4914, 0.4822, 0.4465),
            std=(0.247, 0.243, 0.261),
        ),
    ])

    # --- DataLoader 用の共通パラメータ ---
    train_kwargs = {"batch_size": batch_size, "shuffle": True}
    test_kwargs = {"batch_size": test_batch_size, "shuffle": False}

    if use_cuda:
        cuda_kwargs = {"num_workers": 2, "pin_memory": True}
        train_kwargs.update(cuda_kwargs)
        test_kwargs.update(cuda_kwargs)

    # --- Dataset の作成 ---
    train_dataset = datasets.CIFAR10(
        root=data_dir,
        train=True,
        download=True,
        transform=train_transform,  # データ拡張込みの transform
    )
    test_dataset = datasets.CIFAR10(
        root=data_dir,
        train=False,
        download=True,
        transform=test_transform,   # ランダム操作なし
    )

    # --- DataLoader の作成 ---
    train_loader = DataLoader(train_dataset, **train_kwargs)
    test_loader = DataLoader(test_dataset, **test_kwargs)

    return train_loader, test_loader
