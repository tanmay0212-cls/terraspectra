# TerraSpectra: Environment Setup & Hardware Configuration

## 1. Host Environment Profile

The TerraSpectra development environment is configured to run on Windows 11, Linux, or macOS with clear separation between core inference dependencies and training dependencies.

### Verified Local Workstation Specifications:
- **Operating System:** Windows 11 Home Single Language (64-bit)
- **CPU:** AMD64 Family 25 (8 physical / 16 logical cores)
- **RAM:** 16 GB Total (~4.8 GB Available)
- **GPU Detected:** NVIDIA GeForce RTX 3050 Laptop GPU (4096 MiB VRAM)
- **Python Version:** 3.11.9
- **Node.js / npm:** v24.22.0 / 11.19.1

---

## 2. Windows Smart App Control (SAC) Operational Note

### Background
Windows 11 includes Smart App Control (SAC) to block untrusted or unsigned binaries from executing. In Python environments installed under user directories (e.g. `AppData\Local\Programs\Python`), SAC may flag or delay loading dynamic link libraries (`.dll` files) such as PyTorch's `c10.dll` (`WinError 4551`) and blocks direct calls to `pip.exe` (while permitting `python -m pip`).

### TerraSpectra Architectural Safeguards
1. **Zero-Dependency Inference via ONNX Runtime:**
   * Production inference does **not** load PyTorch. Instead, it runs on `onnxruntime` with `CPUExecutionProvider` or DirectML.
   * `onnxruntime` has been verified operational on this system with zero security violations.
2. **Safe Diagnostics & Soft Imports:**
   * PyTorch imports are wrapped in safe exception handlers (`try...except OSError`), ensuring diagnostics and core pipelines never crash.
3. **Training Separation:**
   * Core inference dependencies (`numpy`, `scipy`, `pandas`, `scikit-learn`, `fastapi`, `onnxruntime`, `pydantic`) form the primary installation tier.
   * PyTorch and heavy GPU libraries are specified as optional dependencies under `[project.optional-dependencies].training`.

---

## 3. Installation Guide

### Step 1: Clone Repository
```bash
git clone https://github.com/tanmay0212-cls/terraspectra.git
cd terraspectra
```

### Step 2: Install Core Runtime (Inference & API)
Run via `python -m pip`:
```bash
python -m pip install -e .
```

### Step 3: Install Development & Testing Dependencies
```bash
python -m pip install -e .[dev]
```

### Step 4: Verify Environment & Hardware Diagnostics
Run the automated diagnostics script:
```bash
python scripts/check_environment.py
```

### Step 5: Execute Test Suite
```bash
python -m pytest tests/unit -v
```
All unit tests run on CPU without requiring an external network connection, GPU, or PyTorch.
