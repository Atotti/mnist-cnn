import os

import numpy as np
import torch
import torch.nn.functional as func
from PIL import Image
from torchvision import transforms

from src.models.cnn import Net


def load_model(model_path: str = "mnist_cnn.pt", device: str = "cpu") -> Net:
    """Load a trained MNIST model from disk.

    Args:
        model_path: Path to the saved model file
        device: Device to load the model on ('cpu', 'cuda', or 'mps')

    Returns:
        Loaded model ready for inference

    Raises:
        FileNotFoundError: If the model file doesn't exist
    """
    if not os.path.exists(model_path):
        msg = f"Model file not found: {model_path}"
        raise FileNotFoundError(msg)

    # Create a new model instance
    device_obj = torch.device(device)
    model = Net().to(device_obj)

    # Load the saved state dictionary
    model.load_state_dict(torch.load(model_path, map_location=device_obj))

    # Set the model to evaluation mode
    model.eval()

    return model


def predict_digit(
    model: Net,
    image: str | Image.Image | torch.Tensor | np.ndarray,
    device: str = "cpu",
) -> tuple[int, list[float]]:
    """Predict the digit in an image using a trained MNIST model.

    Args:
        model: Trained MNIST model
        image: Input image as a file path, PIL Image, PyTorch tensor, or numpy array
        device: Device to run inference on ('cpu', 'cuda', or 'mps')

    Returns:
        Tuple containing (predicted_digit, class_probabilities)

    Raises:
        ValueError: If the image format is not supported
    """
    # Convert the input to a PyTorch tensor
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])

    if isinstance(image, str):
        # Load image from file path
        img = Image.open(image).convert("L")  # Convert to grayscale
        tensor = transform(img).unsqueeze(0)  # Add batch dimension
    elif isinstance(image, Image.Image):
        # Convert PIL Image to tensor
        img = image.convert("L")  # Convert to grayscale
        tensor = transform(img).unsqueeze(0)  # Add batch dimension
    elif isinstance(image, np.ndarray):
        # Convert numpy array to tensor
        if image.ndim == 2:
            # Single grayscale image
            img = Image.fromarray(image.astype(np.uint8))
            tensor = transform(img).unsqueeze(0)
        else:
            msg = "Numpy array must be 2D (grayscale image)"
            raise ValueError(msg)
    elif isinstance(image, torch.Tensor):
        # Use the tensor directly
        if image.dim() == 2:
            # Add channel and batch dimensions
            tensor = transform(image.unsqueeze(0)).unsqueeze(0)
        elif image.dim() == 3:
            # Add batch dimension if needed
            tensor = image.unsqueeze(0) if image.size(0) == 1 else image
        else:
            tensor = image  # Assume it's already batched
    else:
        msg = "Unsupported image format"
        raise TypeError(msg)

    # Move tensor to the appropriate device
    device_obj = torch.device(device)
    tensor = tensor.to(device_obj)

    # Get prediction
    with torch.no_grad():
        output = model(tensor)
        # Convert log probabilities to probabilities
        probabilities = torch.exp(output).cpu().numpy()[0]
        # Get the predicted class
        predicted_class = output.argmax(dim=1, keepdim=True).item()

    return predicted_class, probabilities.tolist()
