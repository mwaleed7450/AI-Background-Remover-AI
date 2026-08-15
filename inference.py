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
  quality="fast"     → isnet-general-use   (smallest+fastest, good general subjects)
  quality="standard" → u2net_human_seg     (portrait-tuned U²-Net, best for people)
  quality="quality"  → birefnet-general    (BiRefNet, best edge quality for anything)

Speed notes (CPU):
  isnet-general-use is ~40% faster than u2net on CPU inference.
  u2net_human_seg is similar speed to isnet but trained exclusively on human
  subjects — gives noticeably cleaner hair and skin edges for portraits.
  BiRefNet gives the best edges but is significantly heavier — warm-up at
  startup eliminates the first-request delay.

The model session is cached after the first load so every subsequent request
reuses the in-memory session — no reload cost.
"""

import os
import io
import threading
import numpy as np
from pathlib import Path
from dotenv import load_dotenv
from preprocessing import preprocess
from postprocessing import postprocess, refine_rembg_output

# Load AI-Background-Remover-AI/.env (the file lives next to inference.py)
load_dotenv(dotenv_path=Path(__file__).parent / ".env")

MODEL_BACKEND    = os.getenv("MODEL_BACKEND",    "rembg")
ONNX_MODEL_PATH  = os.getenv("ONNX_MODEL_PATH",  "models/model.onnx")
TORCH_MODEL_PATH = os.getenv("TORCH_MODEL_PATH", "models/model.pth")

# Default quality when callers do not specify (env override supported)
DEFAULT_QUALITY = os.getenv("DEFAULT_QUALITY", "fast")

# Map user-facing quality strings to rembg model IDs.
# isnet-general-use  — fastest, good for products/objects
# u2net_human_seg    — portrait-tuned, best for people/faces
# birefnet-general   — heaviest, best overall edge quality
_REMBG_MODEL_IDS: dict[str, str] = {
    "fast":     "isnet-general-use",
    "standard": "u2net_human_seg",
    "quality":  "birefnet-general",
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
        model_id: A rembg model identifier, e.g. "isnet-general-use".

    Returns:
        A rembg session object.
    """
    with _session_lock:
        if model_id not in _session_cache:
            from rembg import new_session
            _session_cache[model_id] = new_session(model_id)
        return _session_cache[model_id]


# ---------------------------------------------------------------------------
# Public warm-up helper  (call once at server startup)
# ---------------------------------------------------------------------------

def warm_up_models() -> None:
    """
    Pre-load every rembg session into the cache so the very first real
    request does not pay the model-download + initialisation cost.

    This runs the session constructors (which download weights on first call)
    but does NOT run inference — it is fast enough to call synchronously
    inside the FastAPI lifespan startup hook.

    Should only be called when MODEL_BACKEND == "rembg".
    """
    if MODEL_BACKEND != "rembg":
        return

    print("🔥 Pre-warming AI models…", flush=True)
    for quality_label, model_id in _REMBG_MODEL_IDS.items():
        try:
            _get_rembg_session(model_id)
            print(f"   ✅  {quality_label} ({model_id}) ready", flush=True)
        except Exception as exc:
            print(f"   ⚠️  Could not pre-warm {quality_label} ({model_id}): {exc}", flush=True)
    print("🔥 Model warm-up complete.", flush=True)


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


def _run_rembg_bytes(image_bytes: bytes, quality: str = "fast") -> bytes:
    """
    Use the rembg library to remove the background, operating entirely in
    memory (no temporary files), then apply guided-matting post-processing
    to sharpen edges.

    Args:
        image_bytes: Raw image file content (JPEG / PNG / WebP bytes).
        quality:     "fast" (isnet-general-use) or "quality" (birefnet-general).

    Returns:
        Refined transparent PNG as raw bytes.
    """
    from rembg import remove

    model_id = _REMBG_MODEL_IDS.get(quality, _REMBG_MODEL_IDS["fast"])
    session  = _get_rembg_session(model_id)
    raw      = remove(image_bytes, session=session)

    # Apply guided-matting + morphological refinement on top of rembg output
    return refine_rembg_output(raw)


def _run_rembg(image_path: str, output_path: str, quality: str = "fast") -> None:
    """
    Use the rembg library to remove the background (file-based interface,
    kept for compatibility with the ONNX/PyTorch pipeline paths).

    Args:
        image_path:  Source image path.
        output_path: Destination PNG path.
        quality:     "fast" or "quality".
    """
    with open(image_path, "rb") as inp:
        result = _run_rembg_bytes(inp.read(), quality=quality)
    with open(output_path, "wb") as out:
        out.write(result)


# ---------------------------------------------------------------------------
# Public entry points
# ---------------------------------------------------------------------------

def run_inference(
    image_path: str,
    output_path: str,
    quality: str | None = None,
) -> None:
    """
    End-to-end background removal for a single image (file-based interface).

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


def run_inference_bytes(
    image_bytes: bytes,
    quality: str | None = None,
) -> bytes:
    """
    End-to-end background removal operating entirely in memory.

    Avoids writing the source image to a temporary file — the bytes come
    straight from the HTTP request and the result PNG bytes go straight
    back to the caller.  Only supported for the rembg backend.

    Args:
        image_bytes: Raw source image bytes (JPEG / PNG / WebP).
        quality:     "fast" or "quality".

    Returns:
        Transparent PNG as raw bytes.

    Raises:
        NotImplementedError: If MODEL_BACKEND is not "rembg".
    """
    effective_quality = quality or DEFAULT_QUALITY

    if MODEL_BACKEND == "rembg":
        return _run_rembg_bytes(image_bytes, quality=effective_quality)

    raise NotImplementedError(
        "run_inference_bytes is only supported for MODEL_BACKEND='rembg'. "
        f"Current backend: '{MODEL_BACKEND}'."
    )
