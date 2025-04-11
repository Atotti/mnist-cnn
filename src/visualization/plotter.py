"""Plotting utilities for visualizing training and evaluation metrics."""

from typing import List, Optional, Tuple

import matplotlib

# Use Agg backend for environments without a display
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def plot_losses(
    train_losses: list[float],
    eval_losses: list[float],
    save_path: str | None = None,
    show_plot: bool = True,
) -> None:
    """Plot training and evaluation losses.

    Args:
        train_losses: List of training losses per epoch
        eval_losses: List of evaluation losses per epoch
        save_path: Path to save the plot image (optional)
        show_plot: Whether to display the plot (default: True)
    """
    plt.figure(figsize=(10, 6))
    epochs = range(1, len(train_losses) + 1)

    plt.plot(epochs, train_losses, "b-", label="Training Loss")
    plt.plot(epochs, eval_losses, "r-", label="Evaluation Loss")

    plt.title("Training and Evaluation Losses")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.7)

    # Add annotations for minimum loss points
    min_train_epoch = train_losses.index(min(train_losses)) + 1
    min_eval_epoch = eval_losses.index(min(eval_losses)) + 1

    plt.annotate(
        f"Min: {min(train_losses):.4f}",
        xy=(min_train_epoch, min(train_losses)),
        xytext=(min_train_epoch, min(train_losses) * 1.1),
        arrowprops=dict(facecolor="blue", shrink=0.05, alpha=0.7),
    )

    plt.annotate(
        f"Min: {min(eval_losses):.4f}",
        xy=(min_eval_epoch, min(eval_losses)),
        xytext=(min_eval_epoch, min(eval_losses) * 1.1),
        arrowprops=dict(facecolor="red", shrink=0.05, alpha=0.7),
    )

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
        print(f"Plot saved to {save_path}")

    plt.close()

def plot_accurency(
    eval_accurency: list[float],
    save_path: str | None = None,
    show_plot: bool = True,
):
    plt.figure(figsize=(10, 6))
    epochs = range(1, len(eval_accurency) + 1)

    plt.plot(epochs, eval_accurency, "r-", label="Evaluation Accurency")

    plt.title("Training and Evaluation Accurency")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.7)


    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
        print(f"Plot saved to {save_path}")

    plt.close()
