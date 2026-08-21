"""
Mask post-processing and transparent PNG generation.

Two pipelines live here:

  1. raw_mask pipeline  (ONNX / PyTorch backends)
     upsample → binarise → refine → apply alpha → save PNG
     Alpha refinement defaults to guided-filter matting; closed-form
     trimap-based matting (pymatting) is available as an opt-in for
     higher-fidelity hair/fur edges on smaller images.

  2. rgba_bytes pipeline  (rembg backend)
     rembg already returns an RGBA PNG — we refine the alpha channel it
     produced to fix jagged edges, fill small holes, and recover
     semi-transparent pixels at subject boundaries via guided matting.
"""

import io

import cv2
import numpy as np
from PIL import Image
from pymatting import estimate_alpha_cf


# ─────────────────────────────────────────────────────────────────────────────
# Shared alpha-channel refinement
# Applied to ANY alpha mask regardless of where it came from.
# ─────────────────────────────────────────────────────────────────────────────

def refine_alpha(alpha: np.ndarray, aggressive: bool = False) -> np.ndarray:
    """
    Refine a uint8 alpha mask (0–255) to reduce jagged edges and noise.

    Steps
    ─────
    1. Close small holes inside the subject (MORPH_CLOSE, small kernel).
    2. Remove isolated noise blobs outside the subject (MORPH_OPEN, tiny kernel).
    3. Apply guided edge-preserving blur to smooth the boundary without
       eroding the subject core.

    Args:
        alpha:      uint8 array (H, W) with values 0–255.
        aggressive: If True, use a slightly larger kernel for noisy masks.

    Returns:
        Refined uint8 alpha array.
    """
    ksize = 3 if not aggressive else 5

    # ── Fill small holes ──────────────────────────────────────────────────
    close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))
    alpha = cv2.morphologyEx(alpha, cv2.MORPH_CLOSE, close_kernel, iterations=2)

    # ── Remove isolated noise blobs ───────────────────────────────────────
    # Only remove very small specks (kernel 3×3, 1 iteration) so we don't
    # accidentally erode fine detail (hair, fur, thin straps).
    open_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    alpha = cv2.morphologyEx(alpha, cv2.MORPH_OPEN, open_kernel, iterations=1)

    # ── Edge-preserving smooth ────────────────────────────────────────────
    # Bilateral filter on the alpha channel: smooths flat regions but
    # keeps sharp transitions at the subject boundary.
    alpha_f = alpha.astype(np.float32)
    alpha_f = cv2.bilateralFilter(alpha_f, d=5, sigmaColor=25, sigmaSpace=25)
    alpha = np.clip(alpha_f, 0, 255).astype(np.uint8)

    return alpha


def guided_alpha_matting(
    rgb: np.ndarray,
    alpha: np.ndarray,
    radius: int = 5,
) -> np.ndarray:
    """
    Guided filter matting — uses the original image colours to sharpen
    the alpha boundary.

    This recovers semi-transparent pixels at edges (hair, fine fabric)
    that a hard threshold would otherwise clip to fully opaque or fully
    transparent.  It is fast, runs on CPU, and requires no extra model.

    Args:
        rgb:    uint8 (H, W, 3) original RGB image.
        alpha:  uint8 (H, W) alpha mask from the segmentation model.
        radius: Guided filter radius (larger = smoother guidance).

    Returns:
        Refined uint8 alpha array.
    """
    # Guided filter expects float32 in [0, 1]
    guide  = rgb.astype(np.float32) / 255.0
    src    = alpha.astype(np.float32) / 255.0

    # cv2.ximgproc may not be installed — fall back gracefully
    try:
        gf = cv2.ximgproc.createGuidedFilter(guide, radius=radius, eps=1e-3)
        refined = gf.filter(src)
    except AttributeError:
        # ximgproc not available — use a fast box-filter approximation instead
        refined = _box_guided_approx(guide, src, radius)

    refined = np.clip(refined, 0.0, 1.0)
    return (refined * 255).astype(np.uint8)


def _box_guided_approx(
    guide: np.ndarray,   # float32 (H, W, 3)
    src: np.ndarray,     # float32 (H, W)
    radius: int,
) -> np.ndarray:
    """
    Lightweight approximation of guided filtering using box filters.

    Computes the per-pixel linear model (a, b) that minimises the
    difference between the guidance image and the output, then applies
    a mild Gaussian blur of the same radius for speed.  Produces slightly
    softer edges than the full guided filter but is robust on all OpenCV
    builds.
    """
    # Convert guide to grayscale for a simpler single-channel computation
    guide_gray = cv2.cvtColor((guide * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0

    ksize = 2 * radius + 1
    mean_I  = cv2.boxFilter(guide_gray, -1, (ksize, ksize))
    mean_p  = cv2.boxFilter(src,        -1, (ksize, ksize))
    corr_Ip = cv2.boxFilter(guide_gray * src, -1, (ksize, ksize))
    corr_II = cv2.boxFilter(guide_gray * guide_gray, -1, (ksize, ksize))

    eps = 1e-3
    var_I = corr_II - mean_I * mean_I
    cov   = corr_Ip - mean_I * mean_p

    a = cov / (var_I + eps)
    b = mean_p - a * mean_I

    mean_a = cv2.boxFilter(a, -1, (ksize, ksize))
    mean_b = cv2.boxFilter(b, -1, (ksize, ksize))

    return mean_a * guide_gray + mean_b


# ─────────────────────────────────────────────────────────────────────────────
# rembg output refinement  (main path used by the app)
# ─────────────────────────────────────────────────────────────────────────────

def refine_rembg_output(rgba_bytes: bytes) -> bytes:
    """
    Take the raw RGBA PNG bytes produced by rembg and apply additional
    post-processing to improve edge quality.

    Pipeline
    ────────
    1. Decode to numpy RGBA.
    2. Morphological refinement of the alpha channel.
    3. Guided matting using the RGB channels as guidance.
    4. Re-encode to PNG bytes.

    Args:
        rgba_bytes: Raw RGBA PNG bytes from rembg.

    Returns:
        Refined RGBA PNG bytes.
    """
    # ── Decode ────────────────────────────────────────────────────────────
    nparr = np.frombuffer(rgba_bytes, np.uint8)
    img   = cv2.imdecode(nparr, cv2.IMREAD_UNCHANGED)  # (H, W, 4) BGRA

    if img is None or img.shape[2] != 4:
        # Can't decode or no alpha channel — return original unchanged
        return rgba_bytes

    bgr   = img[:, :, :3]
    alpha = img[:, :, 3]
    rgb   = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    # ── Morphological refinement ──────────────────────────────────────────
    alpha = refine_alpha(alpha, aggressive=False)

    # ── Guided matting ────────────────────────────────────────────────────
    alpha = guided_alpha_matting(rgb, alpha, radius=5)

    # ── Re-encode to RGBA PNG ─────────────────────────────────────────────
    result_bgra = cv2.merge([bgr[:, :, 0], bgr[:, :, 1], bgr[:, :, 2], alpha])
    _, buf = cv2.imencode(".png", result_bgra)
    return buf.tobytes()


# ─────────────────────────────────────────────────────────────────────────────
# ONNX / PyTorch pipeline  (kept for non-rembg backends)
# ─────────────────────────────────────────────────────────────────────────────

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
    Legacy wrapper kept for the ONNX/PyTorch pipeline.
    Calls the shared refine_alpha function.
    """
    return refine_alpha(mask, aggressive=False)


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
    trimap, instead of a hard binary cutout. Gives natural, semi-transparent
    edges around hair/fur/fine detail, but is memory-hungry and can fail on
    larger images — callers should treat this as an opt-in, best-effort
    refinement and fall back to guided_alpha_matting() on failure.

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
    """
    image = Image.open(image_path).convert("RGBA")
    alpha = Image.fromarray(mask, mode="L")

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
    use_cf_matting: bool = False,
) -> None:
    """
    Full post-processing pipeline for ONNX/PyTorch backends:
    upsample → binarise → refine → alpha refinement → apply alpha → save PNG.

    Alpha refinement step:
      - use_matting=True, use_cf_matting=False (default): guided-filter
        matting — fast, robust, handles any image size.
      - use_matting=True, use_cf_matting=True: closed-form trimap-based
        matting (pymatting) for higher-fidelity hair/fur edges on smaller
        images. Falls back to guided-filter matting automatically if it
        errors (e.g. memory allocation failure on larger images).
      - use_matting=False: skip alpha refinement, use the plain
        binarised + morphologically refined mask.

    Args:
        raw_mask:       Model output probability map, shape (H, W), float32.
        image_path:     Path to the original source image.
        output_path:    Where to write the transparent PNG.
        original_size:  (width, height) of the original image.
        threshold:      Binarisation threshold (default 0.5).
        use_matting:    Whether to run any alpha-matting refinement at all.
        use_cf_matting: Whether to prefer closed-form (pymatting) matting
                        over the default guided-filter matting.
    """
    # Load original RGB for guided matting
    orig = cv2.imread(image_path, cv2.IMREAD_COLOR)
    rgb  = cv2.cvtColor(orig, cv2.COLOR_BGR2RGB) if orig is not None else None

    mask = upsample_mask(raw_mask, original_size)
    mask = binarise_mask(mask, threshold)
    mask = refine_mask(mask)

    def _run_guided_matting(current_mask: np.ndarray) -> np.ndarray:
        if rgb is None:
            return current_mask
        guide_rgb = rgb
        if guide_rgb.shape[:2] != current_mask.shape[:2]:
            guide_rgb = cv2.resize(guide_rgb, (current_mask.shape[1], current_mask.shape[0]))
        return guided_alpha_matting(guide_rgb, current_mask, radius=5)

    if use_matting:
        if use_cf_matting:
            try:
                trimap = generate_trimap(mask)
                mask = apply_alpha_matting(image_path, trimap)
            except Exception as exc:
                # Fall back to guided-filter matting rather than failing
                # the whole request if closed-form matting errors out.
                print(f"[postprocess] Closed-form alpha matting failed, "
                      f"falling back to guided filter matting: {exc}")
                mask = _run_guided_matting(mask)
        else:
            mask = _run_guided_matting(mask)

    apply_mask(image_path, mask, output_path)