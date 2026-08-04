# AI Background Remover — AI Module

> **Teams:** AI Team (pipeline integration) · ML Team (model research & training)
> **Repo:** `AI-Background-Remover-AI`
> **Parent repo:** `AI-Background-Remover` (this is a submodule)
> **Tech:** Python 3.11 · PyTorch · ONNX Runtime · OpenCV · Pillow · rembg

---

## What This Repo Is

The AI engine for the AI Background Remover application.
It receives an image path, runs it through a deep learning segmentation model, and produces a transparent PNG.

It is called by the backend (`services/bg_removal.py`) — it does not run a web server or accept HTTP requests itself.

---

## Who Works Here

| Team | Responsibility | Folders |
|------|---------------|---------|
| **AI Team** | Pipeline code — integrate models, write pre/postprocessing, optimize inference | `inference.py`, `preprocessing.py`, `postprocessing.py` |
| **ML Team** | Model research — evaluate architectures, train, benchmark, export weights | `research/` (create this folder for notebooks and scripts) |

---

## Folder Structure

```
AI-Background-Remover-AI/
│
├── __init__.py                 ← marks this as a Python package (ai.*)
│
├── inference.py                ← PUBLIC entry point
│                                 run_inference(image_path, output_path)
│                                 switches between backends via MODEL_BACKEND env var
│
├── preprocessing.py            ← image loading, resize, normalize, to tensor
│
├── postprocessing.py           ← upsample mask, binarise, refine edges,
│                                 apply alpha channel, save PNG
│
├── models/                     ← model weight files go here (not committed to git)
│   └── .gitkeep
│
└── research/                   ← ML Team workspace (create as needed)
    ├── notebooks/              ← Jupyter notebooks for experiments
    ├── train.py                ← training scripts
    ├── evaluate.py             ← benchmark accuracy / speed
    └── export_onnx.py          ← export trained PyTorch model to ONNX
```

---

## The Three Backends

The `inference.py` file supports three interchangeable backends.
Switch between them by setting the `MODEL_BACKEND` environment variable.

| Backend | Env value | When to use |
|---------|-----------|-------------|
| **rembg** | `MODEL_BACKEND=rembg` | Quickest start — no model file needed, downloads automatically |
| **ONNX Runtime** | `MODEL_BACKEND=onnx` | Production — fastest CPU/GPU inference |
| **PyTorch** | `MODEL_BACKEND=torch` | Development / fine-tuning — load `.pth` weights directly |

**Default:** `rembg` (set in `.env.example`).

---

## How the Pipeline Works

```
run_inference(image_path, output_path)
        │
        ├── if MODEL_BACKEND == "rembg"
        │       rembg handles everything internally → saves PNG → done
        │
        └── else (onnx or torch)
                │
                ▼
        preprocessing.preprocess(image_path)
          1. Load image from disk (OpenCV → RGB numpy array)
          2. Resize to 1024×1024 (Lanczos interpolation)
          3. Normalize: (pixel/255 - ImageNet_mean) / ImageNet_std
          4. Convert HWC → CHW → NCHW float32 tensor
          Returns: (input_tensor, original_size)
                │
                ▼
        _run_onnx(tensor)  OR  _run_torch(tensor)
          Loads model weights, runs forward pass
          Returns: raw probability mask (H, W) float32 in [0,1]
                │
                ▼
        postprocessing.postprocess(raw_mask, image_path, output_path, original_size)
          1. Upsample mask back to original image dimensions
          2. Binarise: pixels >= 0.5 become 255 (foreground), rest 0
          3. Refine edges: morphological open + close + Gaussian blur
          4. Apply mask as alpha channel to original RGBA image
          5. Save as transparent PNG
```

---

## Public API (what the backend calls)

```python
from ai.inference import run_inference

run_inference(
    image_path="uploads/abc123_photo.jpg",
    output_path="output/abc123_result.png"
)
```

That is the only function the backend ever calls. Everything else is internal.

---

## Supported Segmentation Models

These are the models the ML Team should evaluate and the AI Team integrates:

| Model | Architecture | Best for |
|-------|-------------|---------|
| **U²-Net** | Encoder-decoder with nested U-structure | General background removal, fast |
| **BiRefNet** | Bilateral Reference Network | Fine edge detail (hair, fur) |
| **RMBG-2.0** | Hugging Face model (via `transformers`) | Easy to load, good accuracy |
| **rembg** | U²-Net wrapper | Quickest to deploy, auto-downloads |

---

## Adding or Swapping a Model

### Using rembg (easiest)
No setup needed. Just set `MODEL_BACKEND=rembg` in `.env`.

### Using a custom ONNX model
1. Export your model to ONNX format (see `research/export_onnx.py`).
2. Place the `.onnx` file in `models/`.
3. Set in `.env`:
   ```
   MODEL_BACKEND=onnx
   ONNX_MODEL_PATH=ai/models/your_model.onnx
   ```

### Using a custom PyTorch model
1. Save the trained model with `torch.save(model, path)`.
2. Place the `.pth` file in `models/`.
3. Set in `.env`:
   ```
   MODEL_BACKEND=torch
   TORCH_MODEL_PATH=ai/models/your_model.pth
   ```

### Adding a brand-new backend
1. Add a `_run_yourbackend(input_tensor)` function in `inference.py`.
2. Add it to the `if/elif` chain in `run_inference()`.
3. Add the new env value to `.env.example`.

---

## Preprocessing Details

File: `preprocessing.py`

| Step | Function | Detail |
|------|----------|--------|
| Load | `load_image()` | OpenCV imread → BGR to RGB |
| Resize | `resize_image()` | Target 1024×1024, INTER_LANCZOS4 |
| Normalize | `normalise()` | ImageNet mean `[0.485, 0.456, 0.406]`, std `[0.229, 0.224, 0.225]` |
| Tensor | `to_tensor()` | HWC → CHW → NCHW float32 |

Input size `1024×1024` is the default for U²-Net / BiRefNet / RMBG-2.0.
Change it via the `size` parameter if your model needs a different resolution.

---

## Postprocessing Details

File: `postprocessing.py`

| Step | Function | Detail |
|------|----------|--------|
| Upsample | `upsample_mask()` | INTER_LINEAR resize back to original dims |
| Binarise | `binarise_mask()` | Threshold at 0.5, output 0 or 255 |
| Refine | `refine_mask()` | Morph open (remove noise) + close (fill holes) + Gaussian blur (smooth edges) |
| Apply | `apply_mask()` | Set mask as alpha channel, save RGBA PNG |

---

## Setup

```bash
# Copy the AI-specific environment config
cp AI-Background-Remover-AI/.env.example AI-Background-Remover-AI/.env
# Edit .env — set MODEL_BACKEND, and model paths if using onnx or torch
```

## Running Inference Directly (for testing)

```python
# From the project root with .venv active
python -c "
from AI_Background_Remover_AI.inference import run_inference
run_inference('path/to/test.jpg', 'path/to/output.png')
print('Done')
"
```

---

## ML Team — Research Workflow

1. Create a `research/` folder in this repo.
2. Use Jupyter notebooks in `research/notebooks/` for experiments.
3. Train your model, save checkpoints to `models/` (add to `.gitignore` — large files).
4. Write an `export_onnx.py` script to convert the best checkpoint.
5. Benchmark against rembg using metrics: IoU, F-measure, inference time.
6. When a model beats rembg on the benchmark, hand the `.onnx` file to the AI Team for pipeline integration.
7. Document results in `research/RESULTS.md`.

---

## What Is Done vs What Is Next

### Done
- [x] Full preprocessing pipeline (load, resize, normalize, tensorize)
- [x] Full postprocessing pipeline (upsample, binarise, refine, alpha apply)
- [x] ONNX Runtime backend
- [x] PyTorch backend
- [x] rembg backend (working out of the box)
- [x] `run_inference()` public entry point with env-based backend switching

### Next — AI Team
- [ ] Add BiRefNet backend integration
- [ ] Add RMBG-2.0 (Hugging Face transformers) backend
- [ ] GPU memory management for batched inference
- [ ] Model caching (load once on startup, reuse per request)
- [ ] Confidence threshold tuning via env variable

### Next — ML Team
- [ ] Set up `research/` folder structure
- [ ] Benchmark rembg as baseline
- [ ] Evaluate BiRefNet on standard datasets (P3M, DIS5K)
- [ ] Evaluate RMBG-2.0
- [ ] Export best model to ONNX
- [ ] Document results in `research/RESULTS.md`

---

## Contribution

See [CONTRIBUTING.md](../CONTRIBUTING.md) in the parent repo for branch naming, commit format, and PR rules.

Your branch always goes into this submodule repo (`AI-Background-Remover-AI`), not the parent.
