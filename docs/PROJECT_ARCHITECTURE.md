# TerraSpectra: System Architecture & Design Philosophy

## 1. Scientific Overview & Project Scope

**Scientific Project Definition:**
> **TerraSpectra: Hyperspectral Crop Stress Anomaly Detection and Spatial Disease Mapping**

TerraSpectra is an advanced computer vision and geospatial analytics pipeline designed to detect agricultural vegetation stress, pre-symptomatic physiological anomalies, and spatial disease distributions using high-dimensional contiguous imaging spectrometry (hyperspectral datacubes).

### Scientific Boundary on Temporal Claims
In adherence to scientific integrity:
- **What TerraSpectra Performs:** Pre-symptomatic stress anomaly detection, spatial disease classification, biochemical absorption feature analysis (chlorophyll red-edge shift, water absorption depth), and field-level spatial severity mapping from imaging spectrometer passes.
- **What Is Explicitly NOT Claimed:** Single-epoch static hyperspectral datacubes **cannot predict 3 weeks into the future**. Plant disease progression is an epidemiological spatio-temporal phenomenon governed by pathogen incubation kinetics, ambient temperature, relative humidity, leaf wetness duration, and atmospheric spore transport. The system therefore disallows unvalidated "future forecasting" claims when evaluating static scenes.

---

## 2. Multi-Tier System Architecture

TerraSpectra employs a decoupled, production-oriented architecture separating heavy model training from lightweight, high-throughput geospatial inference:

```
+---------------------------------------------------------------------------------------+
| 1. INGESTION & RADIOMETRIC STAGE                                                      |
|    - Raw Datacube Parsing (H x W x B, B=100-224 bands, 400-2500 nm)                   |
|    - Monotonic Wavelength Calibration & Bad Band Flagging (1.4 µm, 1.9 µm water dips) |
|    - Digital Number (DN) to Surface Reflectance Normalization [0.0, 1.0]              |
+---------------------------------------------------------------------------------------+
                                           |
                                           v
+---------------------------------------------------------------------------------------+
| 2. SPATIAL ANTI-LEAKAGE PARTITIONING                                                  |
|    - Spatially Disjoint Block Tiling (Field-level holdout)                            |
|    - Strict isolation: zero spatial overlap between training and evaluation blocks    |
+---------------------------------------------------------------------------------------+
                                           |
                                           v
+---------------------------------------------------------------------------------------+
| 3. SPECTRAL COMPRESSION & FEATURE EXTRACTION                                          |
|    - Incremental Principal Component Analysis (B bands -> K components, K=16-30)      |
|    - Narrowband Vegetation Indices (NDRE, PRI, mNDVI)                                 |
+---------------------------------------------------------------------------------------+
                                           |
                                           v
+---------------------------------------------------------------------------------------+
| 4. MODELING & INFERENCE (DECOUPLED TIERS)                                             |
|    Tier A: Classical Baseline (Scikit-Learn Random Forest / Linear SVM on PCA)        |
|    Tier B: Spatial-Spectral 3D-CNN + Transformer Backbone                             |
|    Tier C: Serialized ONNX Artifact (DirectML / CPU Execution Engine)                 |
+---------------------------------------------------------------------------------------+
                                           |
                     +---------------------+---------------------+
                     |                                           |
                     v                                           v
+-----------------------------------------+   +-----------------------------------------+
| 5. FASTAPI GEOSPATIAL BACKEND           |   | 6. REACT + DECK.GL VISUALIZER           |
|    - POST /api/v1/predict/cube          |   |    - False-color RGB band selector      |
|    - POST /api/v1/spectrum/inspect      |   |    - WebGL hardware-accelerated heatmap |
|    - Pydantic v2 Array Validation       |   |    - Interactive spectral curve plot    |
+-----------------------------------------+   +-----------------------------------------+
```

---

## 3. Data Leakage Mitigation

A primary challenge in remote sensing computer vision is **spatial autocorrelation leakage**. When random pixel-wise or overlapping patch sampling is used across an airborne scene, pixels from the test set share identical illumination, soil chemistry, and canopy micro-climate with training pixels, resulting in artificially inflated test metrics (>99% accuracy).

**TerraSpectra Enforcement:**
1. **Spatial Disjoint Splitting:** The spatial domain $(H \times W)$ is partitioned into geographic grid blocks.
2. Entire blocks are allocated exclusively to either Train ($70\%$), Validation ($15\%$), or Test ($15\%$).
3. Validation and test patches are extracted exclusively from within their respective spatial blocks with strict boundary margins.

---

## 4. Decoupled Runtime & OS Security Alignment

To guarantee deployment reliability across diverse environments (including Windows 11 with Smart App Control):
1. **Inference Decoupling:** Model execution is completely decoupled from deep learning training frameworks using the standard **ONNX (Open Neural Network Exchange)** format.
2. **CPU/DirectML Execution Provider:** The local runtime uses `onnxruntime` with `CPUExecutionProvider` (and optionally `DmlExecutionProvider`), eliminating reliance on unsigned or blocked native C++ DLLs during serving.
3. **Training Isolation:** PyTorch training routines are kept modular and optional, designed to run in dedicated containerized or WSL2 environments without impacting local inference services.
