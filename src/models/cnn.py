"""CNN model for MNIST digit classification."""

from typing import Any

import torch
import torch.nn.functional as func
from torch import nn


class Net(nn.Module):
    """Convolutional Neural Network for MNIST digit classification.

    This model consists of two convolutional layers followed by two fully connected layers.
    Dropout is applied after max pooling and before the final layer for regularization.
    """

    def __init__(self) -> None:
        """Initialize the CNN model architecture."""
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(12544, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x: Any) -> torch.Tensor:  # noqa: ANN401
        """Forward pass through the network.

        Args:
            x: Input tensor containing CIFAR10 images (batch_size, 3, 32, 32)

        Returns:
            Log probabilities for each digit class (0-9)
        """
        x = self.conv1(x)
        x = func.relu(x)
        x = self.conv2(x)
        x = func.relu(x)
        x = func.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = func.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        return func.log_softmax(x, dim=1)
