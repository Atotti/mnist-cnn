import argparse

import torch
import torch.nn.functional as func
from torch import nn, optim
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader


def train(
    args: argparse.Namespace,
    model: nn.Module,
    device: torch.device,
    train_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    epoch: int,
) -> tuple[float, float]:
    """Train the model for one epoch.

    Args:
        args: Command line arguments
        model: Neural network model to train
        device: Device to train on (cuda, mps, or cpu)
        train_loader: DataLoader for training data
        optimizer: Optimizer for updating model weights
        epoch: Current epoch number
    """
    model.train()
    total_loss = 0
    num_batches = 0
    correct = 0
    total = 0

    for batch_idx, (data, target) in enumerate(train_loader):
        device_data, device_target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(device_data)
        loss = func.nll_loss(output, device_target)
        loss.backward()
        optimizer.step()

        # Calculate accuracy
        pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
        correct += pred.eq(device_target.view_as(pred)).sum().item()
        total += len(device_data)

        total_loss += loss.item()
        num_batches += 1

        if batch_idx % args.log_interval == 0:
            print(
                f"Train Epoch: {epoch} [{batch_idx * len(device_data)}/{len(train_loader.dataset)} "
                f"({100.0 * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}"
            )
            if args.dry_run:
                break

    # Calculate average loss and accuracy for the epoch
    avg_loss = total_loss / num_batches
    train_accuracy = 100.0 * correct / total
    print(f"Train Epoch: {epoch} Average Loss: {avg_loss:.6f}, Accuracy: {correct}/{total} ({train_accuracy:.1f}%)")

    return avg_loss, train_accuracy


def test(model: nn.Module, device: torch.device, test_loader: DataLoader) -> tuple[float, float]:
    """Evaluate the model on test data.

    Args:
        model: Neural network model to evaluate
        device: Device to evaluate on (cuda, mps, or cpu)
        test_loader: DataLoader for test data

    Returns:
        Tuple containing (test loss, test accuracy as a percentage)
    """
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            device_data, device_target = data.to(device), target.to(device)
            output = model(device_data)
            test_loss += func.nll_loss(output, device_target, reduction="sum").item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(device_target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    accuracy = 100.0 * correct / len(test_loader.dataset)

    print(
        f"\nTest set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(test_loader.dataset)} ({accuracy:.1f}%)\n"
    )

    return test_loss, accuracy


def train_model(
    model: nn.Module,
    device: torch.device,
    train_loader: DataLoader,
    test_loader: DataLoader,
    args: argparse.Namespace,
) -> tuple[nn.Module, list[float], list[float], list[float]]:
    """Train the model for multiple epochs.

    Args:
        model: Neural network model to train
        device: Device to train on (cuda, mps, or cpu)
        train_loader: DataLoader for training data
        test_loader: DataLoader for test data
        args: Command line arguments

    Returns:
        Tuple containing (trained model, training losses, evaluation losses, eval accurency)
    """
    optimizer = optim.Adadelta(model.parameters(), lr=args.lr)
    scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)

    train_losses = []
    eval_losses = []
    eval_accurencyes = []
    train_accurencyes = []

    for epoch in range(1, args.epochs + 1):
        # Train and get average loss for the epoch
        train_loss, train_accuracy = train(args, model, device, train_loader, optimizer, epoch)
        train_losses.append(train_loss)

        # Test and get test loss and accuracy
        eval_loss, accuracy = test(model, device, test_loader)
        eval_losses.append(eval_loss)
        eval_accurencyes.append(accuracy)
        train_accurencyes.append(train_accuracy)

        scheduler.step()

    if args.save_model:
        torch.save(model.state_dict(), "mnist_cnn.pt")
        print("Saved model to mnist_cnn.pt")

    return model, train_losses, eval_losses, eval_accurencyes, train_accurencyes
