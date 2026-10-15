# MERIDIAN-Lite: GPU Hardware & Advanced Damage Model Setup Guide

This guide documents the GPU hardware environment, details why the verified default model executes on CPU, specifies the prerequisites for enabling CUDA execution, and outlines an optional future setup path in an isolated environment without disturbing the stable production demonstration.

---

## 1. Executive Summary & Critical Distinction

> [!IMPORTANT]
> **GPU Hardware Availability $\neq$ GPU Inference Runtime Availability**
>
> An NVIDIA GPU and driver can be fully operational at the host OS level (`nvidia-smi`), but an application's Python runtime cannot perform GPU inference unless:
> 1. The inference framework (e.g. `onnxruntime-gpu` or `torch`) was compiled with CUDA/cuDNN support for the exact Python minor version.
> 2. The matching CUDA toolkit runtime libraries (`libcudart.so`, `libcublas.so`) and cuDNN libraries (`libcudnn.so`) are present and discoverable via `LD_LIBRARY_PATH`.
>
> In our environment, **NVIDIA GeForce RTX 3050 6GB Laptop GPU** is detected with **Driver 595.99.02** and **CUDA Driver 13.2**. However, our Python environment is **Python 3.14.4**, for which official pre-built `onnxruntime-gpu` and `torch` CUDA wheels are not currently released on PyPI. 
> 
> Therefore, MERIDIAN-Lite strictly adheres to our engineering mandate:
> **The NVIDIA GPU is detected, but the installed ONNX Runtime does not include CUDAExecutionProvider. Therefore, the system performs real inference on CPU using the verified lightweight ONNX backend. No pretrained xBD/xView2 checkpoint is currently integrated.**

---

## 2. Hardware Environment Audit

- **GPU Model:** NVIDIA GeForce RTX 3050 6GB Laptop GPU
- **Compute Architecture:** Ampere (`sm_86`)
- **Dedicated VRAM:** 6,144 MiB (6.0 GB GDDR6)
- **NVIDIA Linux Driver:** `595.99.02`
- **Supported CUDA Driver API Version:** `13.2`
- **CUDA Toolkit Compiler (`nvcc`):** Not installed (`command not found`)
- **Python Version:** `3.14.4`
- **Installed ONNX Runtime:** `onnxruntime 1.24.4`
- **Reported Execution Providers:** `['AzureExecutionProvider', 'CPUExecutionProvider']`
- **`CUDAExecutionProvider` Available:** **No**
- **PyTorch Installed:** **No**

---

## 3. Why the Current Model Remains on CPU

1. **Deterministic Edge Performance:**  
   The active bi-temporal damage classifier is an ultra-lightweight 23.5 KB convolutional neural network (`meridian_damage_net`). On standard x86_64 CPU cores with SIMD vectorization (AVX2), it processes a full 58-building mosaic in **$113.4\text{ ms}$** with a working memory set of **$< 35\text{ MB}$**.
2. **Zero Dependency & Zero Network Risk:**  
   First-responder and tactical drone ground-control stations often run on battery-constrained field laptops without dedicated mobile GPUs. A CPU-first design guarantees that MERIDIAN-Lite operates 100% air-gapped without requiring gigabytes of CUDA runtime libraries.
3. **No Risk to Verified Baseline:**  
   Attempting to force-compile or install unstable development CUDA packages into Python 3.14 would destabilize the 85+ passing tests and the verified hackathon presentation.

---

## 4. VRAM & Batch-Size Guidance for NVIDIA RTX 3050 (6 GB)

When GPU acceleration is enabled in a dedicated future environment, operators must respect the 6144 MiB VRAM budget:

| Architecture | Parameters | Weight Footprint | Recommended Batch Size | Estimated VRAM Allocation |
|---|---|---|---|---|
| **MERIDIAN-Lite Lightweight CNN (ONNX)** | 2.6k | 23.5 KB | 32–64 patches | $\approx 280\text{ MiB}$ (includes CUDA context) |
| **RescueNet Patch Classifier (ResNet-18)** | 11.7M | 44.8 MB | 8–16 patches | $\approx 750\text{ MiB}$ |
| **xView2 Siamese U-Net (ResNet-50)** | 51.2M | 195.4 MB | 1–2 full tiles ($1024 \times 1024$) | $\approx 3,400\text{ MiB}$ |

> [!WARNING]
> For large dual-stream U-Net architectures, batch sizes $\ge 4$ on $1024 \times 1024$ full-resolution imagery will exceed 6 GB VRAM and trigger CUDA Out-Of-Memory (`CUDA OOM`). Always clamp batch size to 1 or 2 for scene-level segmentation.

---

## 5. Optional Future Setup Path (Isolated Python 3.11/3.12 Environment)

To experiment with GPU acceleration without affecting the stable MERIDIAN-Lite installation, set up a parallel, isolated virtual environment:

```bash
# 1. Install Python 3.11 or 3.12 (supported by pre-built wheels)
sudo apt update && sudo apt install -y python3.11 python3.11-venv

# 2. Create an isolated GPU virtual environment
python3.11 -m venv .venv-gpu
source .venv-gpu/bin/activate

# 3. Install GPU-enabled inference runtime with CUDA 12 support
pip install --upgrade pip
pip install onnxruntime-gpu==1.17.1 --extra-index-url https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/onnxruntime-cuda-12/pypi/simple/

# 4. Verify CUDAExecutionProvider
python -c "import onnxruntime as ort; print('Providers:', ort.get_available_providers()); assert 'CUDAExecutionProvider' in ort.get_available_providers()"

# 5. Run GPU inference safely
python -c "from models.gpu_predict import run_gpu_inference; res = run_gpu_inference('data/processed/hurricane_michael_pre.png', 'data/processed/hurricane_michael_post.png')"
```

---

## 6. Exact Reproduction Commands

Verify current environment diagnostics:

```bash
# Check nvidia-smi
nvidia-smi

# Check ONNX Runtime providers in current project venv
.venv/bin/python -c "import onnxruntime as ort; print('Providers:', ort.get_available_providers())"

# Run automated GPU backend test suite
.venv/bin/python -m pytest tests/test_gpu_backend.py tests/test_gpu_pipeline.py -v

# Run ground-truth pixel evaluation
.venv/bin/python scripts/evaluate_gpu_backend.py
```
