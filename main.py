import argparse

import torch
import torchvision

from src.data import get_data_loaders
# from src.models import Net
from src.training.trainer import train_model
from src.visualization import plot_accurency, plot_losses


def main() -> None:
    """Main function to parse arguments and run the training/evaluation pipeline."""
    # Training settings
    parser = argparse.ArgumentParser(description="PyTorch MNIST")
    parser.add_argument(
        "--batch-size", type=int, default=64, metavar="N", help="input batch size for training (default: 64)"
    )
    parser.add_argument(
        "--test-batch-size", type=int, default=1000, metavar="N", help="input batch size for testing (default: 1000)"
    )
    parser.add_argument("--epochs", type=int, default=20, metavar="N", help="number of epochs to train (default: 14)")
    parser.add_argument("--lr", type=float, default=0.01, metavar="LR", help="learning rate (default: 1.0)")
    parser.add_argument("--gamma", type=float, default=0.7, metavar="M", help="Learning rate step gamma (default: 0.7)")
    parser.add_argument("--no-cuda", action="store_true", default=False, help="disables CUDA training")
    parser.add_argument("--no-mps", action="store_true", default=False, help="disables macOS GPU training")
    parser.add_argument("--dry-run", action="store_true", default=False, help="quickly check a single pass")
    parser.add_argument("--seed", type=int, default=42, metavar="S", help="random seed (default: 42)")
    parser.add_argument(
        "--log-interval",
        type=int,
        default=10,
        metavar="N",
        help="how many batches to wait before logging training status",
    )
    parser.add_argument("--save-model", action="store_true", default=True, help="For Saving the current Model")
    parser.add_argument("--plot-losses", action="store_true", default=True, help="Plot training and evaluation losses")
    parser.add_argument(
        "--save-plot", action="store_true", default=True, help="Save the loss plot to a file"
    )
    parser.add_argument(
        "--plot-path", type=str, default="works/loss_plot.png",
        help="Path to save the loss plot (default: works/loss_plot.png)"
    )
    args = parser.parse_args()

    # Set up device
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    use_mps = not args.no_mps and torch.backends.mps.is_available()

    torch.manual_seed(args.seed)

    if use_cuda:
        device = torch.device("cuda")
    elif use_mps:
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    print("-" * 10, f"Using {device}", "-" * 10, sep="\n")

    # Load data
    train_loader, test_loader = get_data_loaders(
        batch_size=args.batch_size,
        test_batch_size=args.test_batch_size,
        use_cuda=use_cuda,
    )

    # Create model
    model = torchvision.models.resnet34()
    model.conv1 = torch.nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.fc = torch.nn.Sequential(
        torch.nn.Linear(model.fc.in_features, 10),
        torch.nn.LogSoftmax(dim=1)
    )
    model = model.to(device)


    # Train and evaluate model
    model, train_losses, eval_losses, eval_accurency, train_accurency = train_model(model, device, train_loader, test_loader, args)

    # Plot losses if requested
    if args.plot_losses:
        save_path = args.plot_path if args.save_plot else None
        plot_losses(train_losses, eval_losses, save_path=save_path)
        plot_accurency(eval_accurency, train_accurency, save_path=save_path.replace(".", "_acurrency."))


if __name__ == "__main__":
    main()
