"""
Mask post-processing and transparent PNG generation.

Steps:
  1. Upsample the raw probability mask to the original image dimensions.
  2. Binarise / refine edges using morphological operations.
  3. Apply the mask as an alpha channel to the original image.
  4. Save the result as a transparent PNG.
"""

import cv2
import numpy as np
from PIL import Image


def upsample_mask(mask: np.ndarray, original_size: tuple[int, int]) -> np.ndarray:
    """
    Resize a predicted probability mask back to the original image resolution.

    Args:
        mask:          2-D float32 array with values in [0, 1].
        original_size: Target (width, height).

    Returns:
        Upsampled float32 mask of shape (H, W).
    """
    return cv2.resize(mask, original_size, interpolation=cv2.INTER_LINEAR)


def binarise_mask(mask: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    """
    Convert a soft probability mask to a hard binary mask.

    Args:
        mask:      float32 array in [0, 1].
        threshold: Pixel values >= threshold are set to 255 (foreground).

    Returns:
        uint8 array with values 0 or 255.
    """
    binary = (mask >= threshold).astype(np.uint8) * 255
    return binary


def refine_mask(mask: np.ndarray) -> np.ndarray:
    """
    Clean up the binary mask with morphological operations.

    - Removes small isolated noise blobs (opening).
    - Fills small holes in the foreground (closing).
    - Applies Gaussian blur for smooth, anti-aliased edges.

    Args:
        mask: uint8 binary mask (values 0 or 255).

    Returns:
        Refined uint8 mask.
    """
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

    # Remove noise
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    # Fill holes
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)
    # Smooth edges
    mask = cv2.GaussianBlur(mask, (5, 5), sigmaX=1.5)

    return mask


def apply_mask(image_path: str, mask: np.ndarray, output_path: str) -> None:
    """
    Apply a refined alpha mask to the original image and save as a
    transparent PNG.

    Args:
        image_path:  Path to the original source image.
        mask:        uint8 mask of shape (H, W) to use as the alpha channel.
        output_path: Destination path for the RGBA PNG output.
    """
    image = Image.open(image_path).convert("RGBA")
    alpha = Image.fromarray(mask, mode="L")

    # Ensure the mask matches the image size
    if alpha.size != image.size:
        alpha = alpha.resize(image.size, Image.LANCZOS)

    r, g, b, _ = image.split()
    result = Image.merge("RGBA", (r, g, b, alpha))
    result.save(output_path, format="PNG")


def postprocess(
    raw_mask: np.ndarray,
    image_path: str,
    output_path: str,
    original_size: tuple[int, int],
    threshold: float = 0.5,
) -> None:
    """
    Full post-processing pipeline:
    upsample → binarise → refine → apply alpha → save PNG.

    Args:
        raw_mask:      Model output probability map, shape (H, W), float32.
        image_path:    Path to the original source image.
        output_path:   Where to write the transparent PNG.
        original_size: (width, height) of the original image.
        threshold:     Binarisation threshold (default 0.5).
    """
    mask = upsample_mask(raw_mask, original_size)
    mask = binarise_mask(mask, threshold)
    mask = refine_mask(mask)
    apply_mask(image_path, mask, output_path)
