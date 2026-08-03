"""
Image preprocessing for AI segmentation models.

Steps:
  1. Load image from disk (RGB).
  2. Resize to the model's expected input size.
  3. Normalise pixel values to [0, 1] with ImageNet mean/std.
  4. Convert to a (1, C, H, W) float32 tensor ready for inference.
"""

import cv2
import numpy as np
from PIL import Image

# Default input size used by U²-Net / BiRefNet / RMBG-2.0
DEFAULT_SIZE = (1024, 1024)

# ImageNet normalisation constants
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def load_image(image_path: str) -> np.ndarray:
    """
    Load an image from *image_path* and return it as an RGB uint8 NumPy array.

    Args:
        image_path: Path to the source image file.

    Returns:
        NumPy array of shape (H, W, 3), dtype uint8, in RGB channel order.
    """
    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Could not load image: {image_path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def resize_image(image: np.ndarray, size: tuple[int, int] = DEFAULT_SIZE) -> np.ndarray:
    """
    Resize *image* to *size* using high-quality Lanczos interpolation.

    Args:
        image: RGB uint8 NumPy array.
        size:  Target (width, height).

    Returns:
        Resized NumPy array.
    """
    return cv2.resize(image, size, interpolation=cv2.INTER_LANCZOS4)


def normalise(image: np.ndarray) -> np.ndarray:
    """
    Normalise pixel values to zero mean / unit variance using ImageNet stats.

    Args:
        image: RGB uint8 array of shape (H, W, 3).

    Returns:
        float32 array of shape (H, W, 3) normalised to roughly [-2, 2].
    """
    img = image.astype(np.float32) / 255.0
    return (img - MEAN) / STD


def to_tensor(image: np.ndarray) -> np.ndarray:
    """
    Convert a (H, W, C) float32 array to a (1, C, H, W) batch tensor.

    Args:
        image: Normalised float32 array.

    Returns:
        float32 array of shape (1, C, H, W).
    """
    # HWC → CHW → NCHW
    return np.expand_dims(image.transpose(2, 0, 1), axis=0).astype(np.float32)


def preprocess(image_path: str, size: tuple[int, int] = DEFAULT_SIZE) -> tuple[np.ndarray, tuple[int, int]]:
    """
    Full preprocessing pipeline: load → resize → normalise → tensorise.

    Args:
        image_path: Path to the source image.
        size:       Model input size (width, height).

    Returns:
        Tuple of:
          - input_tensor: float32 array of shape (1, 3, H, W).
          - original_size: (width, height) of the original image,
                           used to upsample the mask after inference.
    """
    image = load_image(image_path)
    original_size = (image.shape[1], image.shape[0])  # (W, H)
    resized = resize_image(image, size)
    normalised = normalise(resized)
    tensor = to_tensor(normalised)
    return tensor, original_size
