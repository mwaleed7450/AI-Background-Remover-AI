"""
 test for the alpha matting edge refinement.

"""

import numpy as np
from PIL import Image

from postprocessing import postprocess

# --- EDIT THESE TWO LINES to point at a real test photo ---
INPUT_IMAGE = "D:/test_images/dog.jpeg"
OUTPUT_IMAGE = "D:/test_images/dog_output.png"
# ------------------------------------------------------------


def fake_raw_mask(image_path: str) -> np.ndarray:
    """
    Stand-in for a real model mask, since we're testing postprocessing
    in isolation. Produces a rough centered oval as a placeholder mask
    at low resolution, the same way a model's raw output would look
    before upsampling.
    """
    img = Image.open(image_path)
    w, h = img.size
    small_w, small_h = 256, 256
    mask = np.zeros((small_h, small_w), dtype=np.float32)
    yy, xx = np.ogrid[:small_h, :small_w]
    cy, cx = small_h // 2, small_w // 2
    ry, rx = small_h * 0.35, small_w * 0.3
    ellipse = ((yy - cy) / ry) ** 2 + ((xx - cx) / rx) ** 2 <= 1
    mask[ellipse] = 1.0
    return mask, (w, h)


def resize_for_matting(image_path: str, max_dim: int = 300) -> str:
    """
    Downscale images before matting to avoid memory errors — closed-form
    matting's memory use grows fast with pixel count. Saves a resized
    copy alongside the original and returns its path.
    """
    img = Image.open(image_path)
    w, h = img.size
    if max(w, h) <= max_dim:
        return image_path
    scale = max_dim / max(w, h)
    new_size = (int(w * scale), int(h * scale))
    resized = img.resize(new_size, Image.LANCZOS)
    base, ext = image_path.rsplit(".", 1)
    resized_path = f"{base}_small.{ext}"
    resized.save(resized_path)
    return resized_path


if __name__ == "__main__":
    input_resized = resize_for_matting(INPUT_IMAGE)
    raw_mask, original_size = fake_raw_mask(input_resized)
    postprocess(
        raw_mask=raw_mask,
        image_path=input_resized,
        output_path=OUTPUT_IMAGE,
        original_size=original_size,
        use_matting=True,
    )
    print(f"Done. Output saved to {OUTPUT_IMAGE}")