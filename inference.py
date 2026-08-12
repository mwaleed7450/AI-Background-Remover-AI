"""
AI inference pipeline.

Supports three interchangeable segmentation back-ends:
  - ONNX Runtime  (fastest CPU/GPU inference via exported .onnx weights)
  - PyTorch       (load .pth / .pt weights directly)
  - rembg         (drop-in wrapper — supports multiple model IDs)

Set the MODEL_BACKEND environment variable to one of:
  "onnx"   – uses ONNX Runtime (requires ONNX_MODEL_PATH)
  "torch"  – uses PyTorch (requires TORCH_MODEL_PATH)
  "rembg"  – uses the rembg library (default)

Per-request quality selector (only applies to the "rembg" backend):
  quality="fast"    → u2net          (small model, fast CPU inference)
  quality="quality" → birefnet-general (BiRefNet, best edge quality)

The model session is cached after the first load so switching quality
levels within the same process reuses the cached session automatically.
"""

import os
import threading
import numpy as np
from pathlib import Path
from dotenv import load_dotenv
from preprocessing import preprocess
from postprocessing import postprocess

# Load AI-Background-Remover-AI/.env (the file lives next to inference.py)
load_dotenv(dotenv_path=Path(__file__).parent / ".env")

MODEL_BACKEND    = os.getenv("MODEL_BACKEND",    "rembg")
ONNX_MODEL_PATH  = os.getenv("ONNX_MODEL_PATH",  "models/model.onnx")
TORCH_MODEL_PATH = os.getenv("TORCH_MODEL_PATH", "models/model.pth")

# Default quality when callers do not specify (env override supported)
DEFAULT_QUALITY = os.getenv("DEFAULT_QUALITY", "fast")

# Map user-facing quality strings to rembg model IDs
_REMBG_MODEL_IDS: dict[str, str] = {
    "fast":    "u2net",
    "quality": "birefnet-general",
}

# ---------------------------------------------------------------------------
# rembg session cache
# One session per model ID; thread-safe via a lock.
# ---------------------------------------------------------------------------

_session_cache: dict[str, object] = {}
_session_lock  = threading.Lock()


def _get_rembg_session(model_id: str):
    """
    Return a cached rembg InferenceSession for *model_id*, creating it on
    first access.  Thread-safe.

    Args:
        model_id: A rembg model identifier, e.g. "u2net" or "birefnet-general".

    Returns:
        A rembg session object.
    """
    with _session_lock:
        if model_id not in _session_cache:
            from rembg import new_session
            _session_cache[model_id] = new_session(model_id)
        return _session_cache[model_id]


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
    output = session.run(None, {input_name: input_tensor})[0]  # (1, 1, H, W)
    mask = output[0, 0]  # → (H, W)
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

    if output.dim() == 4:
        output = output[:, 0]
    mask = output[0].cpu().numpy()
    mask = (mask - mask.min()) / (mask.max() - mask.min() + 1e-8)
    return mask.astype(np.float32)


def _run_rembg(image_path: str, output_path: str, quality: str = "fast") -> None:
    """
    Use the rembg library to remove the background.

    Args:
        image_path:  Source image path.
        output_path: Destination PNG path.
        quality:     "fast" (u2net) or "quality" (birefnet-general).
    """
    from rembg import remove

    model_id = _REMBG_MODEL_IDS.get(quality, _REMBG_MODEL_IDS["fast"])
    session  = _get_rembg_session(model_id)

    with open(image_path, "rb") as inp:
        result = remove(inp.read(), session=session)
    with open(output_path, "wb") as out:
        out.write(result)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def run_inference(
    image_path: str,
    output_path: str,
    quality: str | None = None,
) -> None:
    """
    End-to-end background removal for a single image.

    Selects the active back-end via MODEL_BACKEND, runs the full pipeline,
    and writes the transparent PNG to *output_path*.

    Args:
        image_path:  Path to the source image.
        output_path: Destination path for the transparent PNG result.
        quality:     "fast" or "quality" (rembg backend only).
                     Falls back to DEFAULT_QUALITY env var when None.
    """
    effective_quality = quality or DEFAULT_QUALITY

    if MODEL_BACKEND == "rembg":
        _run_rembg(image_path, output_path, quality=effective_quality)
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
