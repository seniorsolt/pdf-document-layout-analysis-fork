# PyTorch + PaddlePaddle on H200: resolving the nvidia-* conflict

**The cleanest path is PaddlePaddle CPU for OCR alongside PyTorch GPU — it eliminates all nvidia-* dependency conflicts entirely.** If GPU OCR is essential, `torch==2.7.0` from the PyTorch whl index combined with `paddlepaddle-gpu` cu126 is the next-best option, since torch's whl/cu126 wheel bundles CUDA internally and avoids declaring nvidia-* pip dependencies. The eslav recognition model works in all recommended configurations. Below is the full analysis of every approach, with concrete version pins and Docker image recommendations.

---

## Why cu118 fails on H200 and cu126 works

The root cause is a deliberate PaddlePaddle build configuration choice, not a CUDA toolkit limitation. In PaddlePaddle's `cmake/cuda.cmake`, the architecture lists are defined differently for each CUDA version target:

```cmake
paddle_known_gpu_archs11 = "61 70 75 80"         # ← NO sm_90
paddle_known_gpu_archs12 = "61 70 75 80 90"       # ← HAS sm_90
```

**CUDA 11.8 can compile sm_90 code** — NVIDIA added native Hopper cubin support in CUDA 11.8 (per the official Hopper Compatibility Guide). But PaddlePaddle's team chose not to include sm_90 in their cu118 wheel builds. The cu118 wheels contain kernels only through sm_80, so they produce silent failures (zero text regions) on H200/H100 hardware. This is tied to the **CUDA build variant**, not the PaddlePaddle release version — any cu118 build of any PaddlePaddle version will lack sm_90. You must use **cu126 or higher** for Hopper GPUs.

---

## The 14-package nvidia dependency matrix

Both torch (from PyPI) and paddlepaddle-gpu cu126 declare explicit version pins on ~14 nvidia-* pip packages. Here is the match analysis for the closest pair found:

| nvidia-* package | torch 2.7.x (PyPI) | paddle 3.2.x (cu126) | Match? |
|---|---|---|---|
| nvidia-cublas-cu12 | 12.6.4.1 | 12.6.4.1 | ✅ |
| nvidia-cuda-cupti-cu12 | 12.6.80 | 12.6.80 | ✅ |
| nvidia-cuda-nvrtc-cu12 | 12.6.77 | 12.6.77 | ✅ |
| nvidia-cuda-runtime-cu12 | 12.6.77 | 12.6.77 | ✅ |
| nvidia-cudnn-cu12 | 9.5.1.17 | 9.5.1.17 | ✅ |
| nvidia-cufft-cu12 | 11.3.0.4 | 11.3.0.4 | ✅ |
| nvidia-cufile-cu12 | 1.11.1.6 | 1.11.1.6 | ✅ |
| nvidia-curand-cu12 | 10.3.7.77 | 10.3.7.77 | ✅ |
| nvidia-cusolver-cu12 | 11.7.1.2 | 11.7.1.2 | ✅ |
| nvidia-cusparse-cu12 | 12.5.4.2 | 12.5.4.2 | ✅ |
| nvidia-cusparselt-cu12 | 0.6.3 | 0.6.3 | ✅ |
| nvidia-nvjitlink-cu12 | 12.6.85 | 12.6.85 | ✅ |
| nvidia-nvtx-cu12 | 12.6.77 | 12.6.77 | ✅ |
| **nvidia-nccl-cu12** | **2.26.2** | **2.25.1** | **❌** |

**13 of 14 packages match exactly.** Only nvidia-nccl-cu12 differs. All other torch+paddle version combinations (torch 2.4–2.6 with cu121/cu124 vs paddle cu126) produce massive conflicts across all 14 packages. Torch 2.8+ bumps several additional packages (cudnn to 9.10.x, nccl to 2.27.x), widening the gap.

The paddle 3.3.0 cu126 build (the latest confirmed available on the paddle index; **3.3.1 cu126 does not appear to have been released** — only 3.3.1 CPU exists on PyPI) most likely retains the same nvidia-* pins as 3.2.x, though this could not be definitively confirmed from wheel metadata.

---

## Three concrete installation strategies

### Strategy A: PaddlePaddle CPU — zero conflicts (recommended)

The `paddlepaddle` CPU package has **zero nvidia-* dependencies**. It is an entirely separate package from `paddlepaddle-gpu` and coexists cleanly with PyTorch GPU. This is the simplest, most maintainable approach.

**Base image:** `nvidia/cuda:12.6.3-cudnn-devel-ubuntu22.04`

The devel variant is required because detectron2 must be compiled from source (it needs `nvcc`). This image provides system-level CUDA 12.6 headers and libraries.

**Install commands:**
```dockerfile
FROM nvidia/cuda:12.6.3-cudnn-devel-ubuntu22.04

# System deps
RUN apt-get update && apt-get install -y python3.11 python3.11-dev python3-pip git

# PyTorch GPU (from pytorch whl index — bundles CUDA, no nvidia-* pip deps)
RUN pip install torch==2.7.0 torchvision==0.22.0 \
    --index-url https://download.pytorch.org/whl/cu126

# PaddlePaddle CPU (zero nvidia deps)
RUN pip install paddlepaddle==3.3.0 \
    -i https://www.paddlepaddle.org.cn/packages/stable/cpu/

# PaddleOCR
RUN pip install paddleocr>=3.4.0

# Detectron2 from source
ENV TORCH_CUDA_ARCH_LIST="9.0"
RUN pip install 'git+https://github.com/facebookresearch/detectron2.git'

# transformers
RUN pip install transformers==4.40.2
```

**Version pins:**

| Package | Version | Index |
|---|---|---|
| torch | 2.7.0+cu126 | https://download.pytorch.org/whl/cu126 |
| torchvision | 0.22.0+cu126 | https://download.pytorch.org/whl/cu126 |
| paddlepaddle | 3.3.0 (CPU) | https://www.paddlepaddle.org.cn/packages/stable/cpu/ |
| paddleocr | ≥3.4.0 | PyPI |
| transformers | 4.40.2 | PyPI |
| detectron2 | latest (source) | GitHub |

**Performance:** PaddleOCR on CPU with MKLDNN optimization enabled achieves **~1–6 seconds per image**, depending on model size and document complexity. Without MKLDNN, it degrades to ~30 seconds. The user's **6–7 second target is achievable** with server models and `enable_mkldnn=True`. Set `cpu_threads` to match physical core count.

```python
from paddleocr import PaddleOCR
ocr = PaddleOCR(
    text_recognition_model_name="eslav_PP-OCRv5_mobile_rec",
    device="cpu"
)
```

**Trade-off:** GPU sits idle during OCR. But this eliminates all pip dependency gymnastics, works cleanly with `uv pip compile`, and produces a reproducible lockfile.

### Strategy B: torch from whl index + paddlepaddle-gpu cu126

When torch is installed from `https://download.pytorch.org/whl/cu126` rather than from PyPI, CUDA libraries are **bundled inside the torch wheel** — no nvidia-* pip packages are declared or installed. PaddlePaddle-gpu cu126 can then install its own nvidia-* packages without any pip-level conflict.

```bash
# torch from whl index (CUDA bundled, no nvidia-* pip deps)
pip install torch==2.7.0 torchvision==0.22.0 \
    --index-url https://download.pytorch.org/whl/cu126

# paddle-gpu with its own nvidia-* deps
pip install paddlepaddle-gpu==3.2.0 \
    -i https://www.paddlepaddle.org.cn/packages/stable/cu126/
```

**Risk:** At runtime, torch loads its bundled CUDA libraries from within its wheel directory, while PaddlePaddle uses `dlopen()` to load from the nvidia-* pip packages. In a single Python process using both frameworks, **two different NCCL versions may be loaded simultaneously**, which can cause `undefined symbol` errors for multi-GPU operations. For single-GPU inference, NCCL is rarely invoked and this is unlikely to be problematic — but it is not zero-risk.

**uv compatibility:** This approach is harder to express in a clean `uv` lockfile because it requires mixing two package indices with different dependency metadata for the same package name.

### Strategy C: both from PyPI, override NCCL

Install both from PyPI/paddle index and let the later install override the NCCL package:

```bash
pip install paddlepaddle-gpu==3.2.0 \
    -i https://www.paddlepaddle.org.cn/packages/stable/cu126/
pip install torch==2.7.0  # overwrites nvidia-nccl-cu12 from 2.25.1 → 2.26.2
```

This results in nvidia-nccl-cu12==2.26.2 (torch's version wins). PaddlePaddle wanted 2.25.1 but receives 2.26.2 — a minor version bump with strong backward compatibility. **For single-GPU inference, this works.** NCCL 2.25→2.26 does not break the ABI for basic operations. But pip will emit warnings, and `uv pip compile` will refuse to generate a lockfile with this conflict unless you use `--no-deps` on one package.

---

## PaddleOCR version requirements and set_optimization_level

**`set_optimization_level()` was added in PaddlePaddle 3.0.0.** This method is called internally by PaddleOCR 3.x during inference configuration. Users running PaddleOCR 3.0+ with PaddlePaddle 2.x encounter `AttributeError: 'AnalysisConfig' object has no attribute 'set_optimization_level'` (confirmed in PaddleOCR issue #15846 and PaddleX issue #2578).

The minimum requirements chain is:

- **PaddleOCR ≥ 3.0**: requires PaddlePaddle ≥ 3.0.0 (hard requirement)
- **PaddleOCR ≥ 3.3.0**: explicitly tested with PaddlePaddle 3.1.0 and 3.1.1
- **PaddleOCR 3.4.x**: requires PaddlePaddle ≥ 3.0.0, recommended ≥ 3.1.0

For the recommended Strategy A, `paddlepaddle==3.3.0` (CPU) satisfies all PaddleOCR 3.4.x requirements.

---

## Detectron2 and transformers compatibility with torch 2.7

**Detectron2** has no prebuilt wheels for any PyTorch version above ~1.10. It must be built from source for all modern PyTorch versions. The project declares a minimum of PyTorch ≥ 1.8 with **no upper bound**, and community reports confirm successful builds against torch 2.4 through 2.7. Building requires:

- A `devel` Docker image (for `nvcc`)
- `TORCH_CUDA_ARCH_LIST="9.0"` (or `"8.0;9.0"` for broader compatibility)
- System CUDA version matching torch's CUDA version (12.6 in our case)
- A note about setuptools: versions ≥ 81 deprecate `pkg_resources`, which can cause build failures (tracked in detectron2 issue #5494; pin `setuptools<81` if needed)

**transformers 4.40.2** (released May 2024) has no upper bound on PyTorch and works with torch 2.7. It uses standard PyTorch APIs (tensors, nn.Module, optimizers). No compatibility issues are documented or expected. If the user later needs models released after May 2024, upgrading to transformers 4.52+ or 5.x would be necessary — v5.x requires PyTorch ≥ 2.4 and Python ≥ 3.10, both satisfied by this setup.

**Torchvision compatibility matrix** (from PyTorch wiki): torch 2.7.0 pairs with **torchvision 0.22.0** and torchaudio 2.7.0.

---

## The eslav model works in every recommended configuration

The `eslav_PP-OCRv5_mobile_rec` model (East Slavic languages — Russian, Belarusian, Ukrainian, Bulgarian, Serbian, Mongolian, Kazakh, and ~30 more Cyrillic-script languages plus English) is fully compatible across all approaches:

- **PaddleOCR CPU mode**: Native support. PaddleOCR auto-downloads the model by name. Usage: `PaddleOCR(text_recognition_model_name="eslav_PP-OCRv5_mobile_rec", device="cpu")`. This is the simplest path.
- **PaddleOCR GPU mode**: Identical API, just `device="gpu"`.
- **RapidOCR + ONNX Runtime**: Pre-converted ONNX models are available on HuggingFace at `monkt/paddleocr-onnx` (`languages/eslav/rec.onnx` + `languages/eslav/dict.txt`).
- **PaddleOCR with HPI/ONNX backend**: PaddleOCR 3.x natively supports ONNX Runtime via its High-Performance Inference (HPI) system, which auto-converts PaddlePaddle models to ONNX via `paddle2onnx`.

**The eslav model is not a constraint on any approach.** It is a standard PP-OCRv5 recognition model distributed through PaddlePaddle's model hub and available in both Paddle and ONNX formats.

---

## Why ONNX Runtime GPU and RapidOCR are weaker alternatives

**ONNX Runtime GPU** does support sm_90 (Hopper) and has no nvidia-* pip dependency conflicts with PyTorch (it declares none, loading CUDA dynamically at runtime). However, multiple sources report **very poor OCR performance** with onnxruntime-gpu due to dynamic input shapes common in OCR pipelines. The Immich ML community measured ~80 seconds with ORT GPU versus ~1 second with PaddleX for equivalent OCR tasks. RapidOCR's official documentation explicitly states GPU inference via onnxruntime is "very slow" (推理很慢).

**RapidOCR** (v3.2.0+) is a multi-backend OCR toolkit that uses PaddleOCR models converted to ONNX. It is **not a drop-in replacement** — the API differs significantly (different initialization, different result format, no auto-download by model name). Detection is **~2x slower** than PaddleOCR on CPU for the same models (101–210ms vs 30–59ms per detection pass). Its primary advantage — eliminating the PaddlePaddle dependency entirely — is unnecessary when using PaddlePaddle CPU, which already solves the conflict.

---

## Concrete recommendation

**Use Strategy A (PaddlePaddle CPU + PyTorch GPU)** for the following reasons:

- **Zero dependency conflicts** — `paddlepaddle` CPU has no nvidia-* packages, producing a clean `uv pip compile` lockfile with no overrides or `--no-deps` hacks
- **6–7 seconds per page is within the user's stated tolerance** for OCR processing time, achievable with MKLDNN enabled
- **eslav model works natively** with no model conversion or API changes
- **detectron2 and VGT layout detection** run on GPU via PyTorch as normal
- **transformers 4.40.2** is fully compatible
- **Reproducible Docker builds** with a single, coherent dependency tree

| Component | Exact pin | Source |
|---|---|---|
| Base image | `nvidia/cuda:12.6.3-cudnn-devel-ubuntu22.04` | Docker Hub |
| torch | `2.7.0+cu126` | `--index-url https://download.pytorch.org/whl/cu126` |
| torchvision | `0.22.0+cu126` | `--index-url https://download.pytorch.org/whl/cu126` |
| paddlepaddle | `3.3.0` (CPU) | `-i https://www.paddlepaddle.org.cn/packages/stable/cpu/` |
| paddleocr | `≥3.4.0` | PyPI |
| transformers | `4.40.2` | PyPI |
| detectron2 | latest from source | `git+https://github.com/facebookresearch/detectron2.git` |
| Python | `3.11` | System/conda |

If the OCR latency ever becomes unacceptable and GPU OCR is needed, the upgrade path is: switch `paddlepaddle==3.3.0` to `paddlepaddle-gpu==3.2.0` from `cu126` index, install torch from the PyTorch whl index (Strategy B), and accept the minor runtime risk of dual CUDA library loading. The single NCCL version difference (2.25.1 vs 2.26.2) is safe for single-GPU inference.

## Conclusion

The PyTorch + PaddlePaddle nvidia-* conflict is a well-documented ecosystem problem with **no perfect GPU-GPU solution** — even the best version pair (torch 2.7.x + paddle 3.2.x cu126) has a one-package NCCL mismatch. The PaddlePaddle CPU approach sidesteps the entire problem class, delivers acceptable OCR performance, preserves eslav model compatibility, and produces a lockfile-friendly dependency graph. The cu118 failure on H200 is not a PaddlePaddle bug but a deliberate build choice: sm_90 kernels are only compiled into cu126+ wheels, regardless of PaddlePaddle version. Any future attempt at GPU co-installation should target torch 2.7.x (not higher) with paddle 3.2.x cu126, as these versions share the tightest nvidia-* alignment found across all tested combinations.