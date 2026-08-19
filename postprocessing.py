"""
Mask post-processing and transparent PNG generation.

Steps:
  1. Upsample the raw probability mask to the original image dimensions.
  2. Binarise / refine edges using morphological operations.
  3. (New) Build a trimap and run ML-based alpha matting for soft,
     natural edges around hair/fur/fine detail.
  4. Apply the mask (or matte) as an alpha channel to the original image.
  5. Save the result as a transparent PNG.
"""

import cv2
import numpy as np
from PIL import Image
from pymatting import estimate_alpha_cf


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


def generate_trimap(
    mask: np.ndarray,
    erosion_size: int = 15,
    dilation_size: int = 15,
) -> np.ndarray:
    """
    Build a 3-zone trimap from a binary mask, for use with alpha matting.

    - Eroded region  -> definitely foreground (255)
    - Outside the dilated region -> definitely background (0)
    - Everything in between (the fuzzy border, e.g. hair) -> unknown (128)

    Args:
        mask:          uint8 binary mask (values 0 or 255), e.g. from
                       binarise_mask().
        erosion_size:  Kernel size used to shrink the foreground region.
                       Larger = more of the border marked "unknown".
        dilation_size: Kernel size used to expand the foreground region.
                       Larger = more of the border marked "unknown".

    Returns:
        uint8 trimap with values 0, 128, or 255.
    """
    erosion_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (erosion_size, erosion_size)
    )
    dilation_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (dilation_size, dilation_size)
    )

    sure_fg = cv2.erode(mask, erosion_kernel, iterations=1)
    sure_bg_inverse = cv2.dilate(mask, dilation_kernel, iterations=1)

    trimap = np.full(mask.shape, 128, dtype=np.uint8)
    trimap[sure_bg_inverse == 0] = 0
    trimap[sure_fg == 255] = 255

    return trimap


def apply_alpha_matting(image_path: str, trimap: np.ndarray) -> np.ndarray:
    """
    Run closed-form alpha matting to produce a soft alpha matte from a
    trimap, instead of a hard binary cutout. This is what gives natural,
    semi-transparent edges around hair/fur/fine detail.

    Args:
        image_path: Path to the original source image.
        trimap:     uint8 trimap (0 / 128 / 255) matching the image's
                    pixel dimensions, e.g. from generate_trimap().

    Returns:
        uint8 alpha matte of shape (H, W), values 0-255.
    """
    image = Image.open(image_path).convert("RGB")
    image_arr = np.asarray(image, dtype=np.float64) / 255.0
    trimap_arr = trimap.astype(np.float64) / 255.0

    alpha = estimate_alpha_cf(image_arr, trimap_arr)
    alpha_uint8 = np.clip(alpha * 255.0, 0, 255).astype(np.uint8)

    return alpha_uint8


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
    use_matting: bool = True,
) -> None:
    """
    Full post-processing pipeline.

    With matting (default):
        upsample -> binarise -> refine -> trimap -> alpha matting -> apply -> save
    Without matting (legacy path, e.g. as a fallback):
        upsample -> binarise -> refine -> apply -> save

    Args:
        raw_mask:      Model output probability map, shape (H, W), float32.
        image_path:    Path to the original source image.
        output_path:   Where to write the transparent PNG.
        original_size: (width, height) of the original image.
        threshold:     Binarisation threshold (default 0.5).
        use_matting:   If True, refine edges with ML-based alpha matting
                       for soft hair/fur detail. If False, use the plain
                       binarised + morphologically refined mask.
    """
    mask = upsample_mask(raw_mask, original_size)
    mask = binarise_mask(mask, threshold)
    mask = refine_mask(mask)

    if use_matting:
        try:
            trimap = generate_trimap(mask)
            mask = apply_alpha_matting(image_path, trimap)
        except Exception as exc:
            # Fall back to the plain refined mask rather than failing
            # the whole request if matting errors out on an odd input.
            print(f"[postprocess] Alpha matting failed, falling back "
                  f"to binary mask: {exc}")

    apply_mask(image_path, mask, output_path)