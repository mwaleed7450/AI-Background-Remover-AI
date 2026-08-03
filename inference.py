"""
AI inference pipeline.

Supports three interchangeable segmentation back-ends:
  - ONNX Runtime  (fastest CPU/GPU inference via exported .onnx weights)
  - PyTorch       (load .pth / .pt weights directly)
  - rembg         (drop-in U²-Net wrapper, great for quick prototyping)

Set the MODEL_BACKEND environment variable to one of:
  "onnx"   – uses ONNX Runtime (default)
  "torch"  – uses PyTorch
  "rembg"  – uses the rembg library
"""

import os
import numpy as np
from ai.preprocessing import preprocess
from ai.postprocessing import postprocess

MODEL_BACKEND = os.getenv("MODEL_BACKEND", "onnx")
ONNX_MODEL_PATH = os.getenv("ONNX_MODEL_PATH", "ai/models/model.onnx")
TORCH_MODEL_PATH = os.getenv("TORCH_MODEL_PATH", "ai/models/model.pth")


# ---------------------------------------------------------------------------
# Back-end implementations
# ---------------------------------------------------------------------------

def _run_onnx(input_tensor: np.ndarray) -> np.ndarray:
    """
    Run inference with ONNX Runtime.

    Args:
        input_tensor: float32 array of shape (1, 3, H, W).

    Returns:
        Probability mask, shape (H, W), float32, values in [0, 1].
    """
    import onnxruntime as ort

    providers = (
        ["CUDAExecutionProvider", "CPUExecutionProvider"]
        if ort.get_device() == "GPU"
        else ["CPUExecutionProvider"]
    )
    session = ort.InferenceSession(ONNX_MODEL_PATH, providers=providers)
    input_name = session.get_inputs()[0].name
    output = session.run(None, {input_name: input_tensor})[0]  # shape: (1, 1, H, W)
    mask = output[0, 0]  # → (H, W)
    # Normalise to [0, 1] in case the model outputs logits
    mask = (mask - mask.min()) / (mask.max() - mask.min() + 1e-8)
    return mask.astype(np.float32)


def _run_torch(input_tensor: np.ndarray) -> np.ndarray:
    """
    Run inference with a PyTorch model loaded from disk.

    Args:
        input_tensor: float32 array of shape (1, 3, H, W).

    Returns:
        Probability mask, shape (H, W), float32, values in [0, 1].
    """
    import torch

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = torch.load(TORCH_MODEL_PATH, map_location=device)
    model.eval()

    tensor = torch.from_numpy(input_tensor).to(device)
    with torch.no_grad():
        output = model(tensor)

    # Most segmentation models return (B, 1, H, W) or (B, H, W)
    if output.dim() == 4:
        output = output[:, 0]
    mask = output[0].cpu().numpy()
    mask = (mask - mask.min()) / (mask.max() - mask.min() + 1e-8)
    return mask.astype(np.float32)


def _run_rembg(image_path: str, output_path: str) -> None:
    """
    Use the rembg library (U²-Net) to remove the background.

    rembg handles its own I/O, so this helper reads and writes files
    directly and returns None — the caller must not call postprocess().

    Args:
        image_path:  Source image path.
        output_path: Destination PNG path.
    """
    from rembg import remove
    from PIL import Image

    with open(image_path, "rb") as inp:
        result = remove(inp.read())
    with open(output_path, "wb") as out:
        out.write(result)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def run_inference(image_path: str, output_path: str) -> None:
    """
    End-to-end background removal for a single image.

    Selects the active back-end via the MODEL_BACKEND env variable,
    runs preprocessing → inference → postprocessing, and writes the
    transparent PNG to *output_path*.

    Args:
        image_path:  Path to the source image.
        output_path: Destination path for the transparent PNG result.
    """
    if MODEL_BACKEND == "rembg":
        _run_rembg(image_path, output_path)
        return

    input_tensor, original_size = preprocess(image_path)

    if MODEL_BACKEND == "onnx":
        raw_mask = _run_onnx(input_tensor)
    elif MODEL_BACKEND == "torch":
        raw_mask = _run_torch(input_tensor)
    else:
        raise ValueError(
            f"Unknown MODEL_BACKEND '{MODEL_BACKEND}'. "
            "Choose from: 'onnx', 'torch', 'rembg'."
        )

    postprocess(raw_mask, image_path, output_path, original_size)
