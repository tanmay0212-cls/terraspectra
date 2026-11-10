# TerraSpectra

**Hyperspectral Crop Stress Anomaly Detection and Spatial Disease Mapping**

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.11](https://img.shields.io/badge/Python-3.11%2B-brightgreen.svg)](https://www.python.org/)
[![Tests](https://img.shields.io/badge/Tests-19%2F19%20Passing-success.svg)](tests/)
[![Architecture](https://img.shields.io/badge/Inference-ONNX%20Decoupled-orange.svg)](docs/PROJECT_ARCHITECTURE.md)

---

## 1. Project Overview

TerraSpectra is an advanced computer vision and geospatial analytics platform engineered to detect agricultural crop stress, pre-symptomatic vegetative anomalies, and spatial pathogen distributions using contiguous imaging spectrometry (hyperspectral datacubes).

By analyzing contiguous narrowband spectral signatures across the VNIR (400–1000 nm) and SWIR (1000–2500 nm) regions, TerraSpectra identifies minute biochemical shifts—such as chlorophyll-a/b red-edge migration and canopy water absorption depth—well before visible necrosis appears to standard RGB cameras.

> **Scientific Boundary Notice:** Single-epoch static hyperspectral datacubes capture current physiological conditions and stress anomalies. TerraSpectra does **not** claim future temporal disease forecasting from static observations, as disease spread is an epidemiological process governed by atmospheric microclimates and pathogen incubation kinetics.

---

## 2. Key Architectural Features

- **Decoupled ONNX Serving:** Model training and model inference are decoupled. Inference runs via `onnxruntime` using verified CPU/DirectML execution providers, ensuring zero runtime crashes from heavy native deep-learning dependencies or Windows Smart App Control (SAC) policies.
- **Leakage-Free Spatial Partitioning:** Eliminates remote sensing spatial autocorrelation leakage by enforcing spatially disjoint block partitioning (Field-Level Holdout).
- **Narrowband Biochemical Extraction:** Modules for Bad Band Removal (BBR) covering atmospheric water vapor absorption windows (1.4 µm and 1.9 µm), along with narrowband indices (NDRE, PRI, mNDVI).
- **FastAPI Geospatial Backend:** High-throughput streaming and processing of multi-channel raster cubes with Pydantic v2 array validation.
- **Hardware-Aware Design:** Diagnostics module that dynamically detects available hardware (such as 4 GB VRAM limits) and safely routes execution to verified runtimes.

---

## 3. Directory Layout

```
terraspectra/
├── api/                  # FastAPI endpoints and route handlers
├── configs/              # Validated YAML configurations (Pydantic v2 schemas)
├── data/
│   ├── benchmarks/       # Public benchmark datasets (e.g. Salinas, Pavia)
│   ├── processed/        # Preprocessed tensors and spatial patches
│   └── raw/              # Raw hyperspectral cubes (.mat, .hdr, .tif)
├── docs/                 # Architectural specifications, setup guides, reports
├── frontend/             # React + Deck.gl GIS visualization dashboard
├── models/
│   ├── checkpoints/      # Training checkpoints
│   └── onnx/             # Serialized ONNX inference models
├── scripts/              # Operational utilities and environment diagnostics
├── src/terraspectra/     # Core Python package
│   ├── core/             # Configuration, types, exceptions, logging, diagnostics, validation
│   ├── data/             # Ingestion, bad-band removal, spatial tiler
│   ├── features/         # PCA reduction, narrowband spectral indices
│   ├── models/           # Classical baselines, 3D-CNN, Spatial ViT
│   └── evaluation/       # Disjoint cross-validation, OA/AA/Kappa metrics
└── tests/                # Pytest unit and integration test suite
```

---

## 4. Quickstart & Verification

### Prerequisites
- Python 3.10+ (Tested on Python 3.11.9 64-bit)
- Windows 11, Linux, or macOS

### 1. Installation
Install core inference dependencies:
```bash
python -m pip install -e .
```

Install development and testing dependencies:
```bash
python -m pip install -e .[dev]
```

### 2. Run Diagnostics
Inspect your hardware, CPU/RAM capacity, and runtime acceleration:
```bash
python scripts/check_environment.py
```

### 3. Run Test Suite
Execute the offline, zero-dependency unit tests:
```bash
python -m pytest tests/unit -v
```

---

## 5. Implementation Roadmap

- [x] **Phase 0: Master Technical Audit** (Completed)
- [x] **Phase 1: Foundation Scaffold & Environment Stabilization** (Completed)
- [ ] **Phase 2: Hyperspectral Ingestion & Preprocessing Engine**
- [ ] **Phase 3: Baseline & Hybrid 3D-CNN + ViT Modeling**
- [ ] **Phase 4: ONNX Export & Production FastAPI Service**
- [ ] **Phase 5: React & Deck.gl GIS Visualizer**
- [ ] **Phase 6: Integration, Testing & Final Portfolio Delivery**

---

## 6. License

This project is licensed under the MIT License — see the LICENSE file for details.
