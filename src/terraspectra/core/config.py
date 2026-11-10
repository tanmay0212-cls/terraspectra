"""
Validated configuration schemas for TerraSpectra using Pydantic v2.
"""

from __future__ import annotations
from pathlib import Path
from typing import List, Optional, Tuple, Literal
import yaml
from pydantic import BaseModel, Field, field_validator, model_validator


class DataConfig(BaseModel):
    """Configuration for data ingestion, patching, and spatial partitioning."""
    raw_dir: Path = Field(default=Path("data/raw"), description="Directory for raw datacubes")
    processed_dir: Path = Field(default=Path("data/processed"), description="Directory for preprocessed tensors")
    benchmarks_dir: Path = Field(default=Path("data/benchmarks"), description="Directory for public benchmark datasets")
    
    # Dataset dimensions are optional/configurable, never hardcoded
    expected_bands: Optional[int] = Field(default=None, ge=1, le=1024, description="Expected spectral bands, or None if variable")
    spectral_range_nm: Optional[Tuple[float, float]] = Field(default=None, description="Min and max wavelength range in nm")
    
    # Patch extraction
    patch_size: int = Field(default=16, ge=4, le=256, description="Spatial height and width of extracted patches")
    stride: int = Field(default=16, ge=1, le=256, description="Spatial stride for patch sampling")
    
    # Partitioning & spatial anti-leakage
    train_ratio: float = Field(default=0.70, gt=0.0, lt=1.0, description="Proportion of spatial data for training")
    val_ratio: float = Field(default=0.15, ge=0.0, lt=1.0, description="Proportion of spatial data for validation")
    test_ratio: float = Field(default=0.15, gt=0.0, lt=1.0, description="Proportion of spatial data for testing")
    spatial_split_strategy: Literal["disjoint_blocks", "geographic_holdout"] = Field(
        default="disjoint_blocks",
        description="Strategy for spatial splitting to prevent autocorrelation leakage"
    )
    random_seed: int = Field(default=42, description="Global random seed for reproducibility")

    @model_validator(mode="after")
    def validate_split_ratios(self) -> DataConfig:
        total = self.train_ratio + self.val_ratio + self.test_ratio
        if not (0.99 <= total <= 1.01):
            raise ValueError(f"train_ratio ({self.train_ratio}) + val_ratio ({self.val_ratio}) + test_ratio ({self.test_ratio}) must sum to 1.0 (got {total:.3f})")
        return self


class PreprocessConfig(BaseModel):
    """Configuration for spectral preprocessing and dimensionality reduction."""
    pca_components: int = Field(default=20, ge=2, le=128, description="Number of principal components to extract")
    normalize_reflectance: bool = Field(default=True, description="Scale raw digital numbers or radiances to [0.0, 1.0]")
    remove_water_absorption_bands: bool = Field(
        default=False, 
        description="Whether to strip known atmospheric water absorption regions (requires band wavelength metadata)"
    )
    bad_band_ranges_nm: List[Tuple[float, float]] = Field(
        default=[(1350.0, 1420.0), (1800.0, 1960.0)],
        description="Default atmospheric water vapor absorption windows in nanometers"
    )


class ModelConfig(BaseModel):
    """Configuration for model architecture and checkpoints."""
    architecture: Literal["pca_baseline", "3d_cnn", "hybrid_vit"] = Field(
        default="pca_baseline",
        description="Target model backbone architecture"
    )
    num_classes: int = Field(default=16, ge=2, description="Number of target pathology or land cover classes")
    in_channels: int = Field(default=20, ge=1, description="Input spectral channels (e.g. after PCA or band selection)")
    dropout: float = Field(default=0.2, ge=0.0, le=0.9, description="Dropout rate for regularized training")
    checkpoint_dir: Path = Field(default=Path("models/checkpoints"), description="Path to save PyTorch training weights")
    onnx_dir: Path = Field(default=Path("models/onnx"), description="Path to export ONNX inference artifacts")


class TrainingConfig(BaseModel):
    """Configuration for model training execution."""
    batch_size: int = Field(default=8, ge=1, le=256, description="Mini-batch size (kept small for VRAM safety)")
    learning_rate: float = Field(default=1e-3, gt=0.0, description="Optimizer learning rate")
    num_epochs: int = Field(default=30, ge=1, le=1000, description="Total training epochs")
    device: Literal["cpu", "cuda", "directml"] = Field(
        default="cpu",
        description="Target execution device (CPU is default safe fallback)"
    )
    mixed_precision: bool = Field(default=False, description="Enable FP16 automatic mixed precision")


class APIConfig(BaseModel):
    """Configuration for FastAPI service."""
    host: str = Field(default="127.0.0.1", description="API bind host")
    port: int = Field(default=8000, ge=1024, le=65535, description="API bind port")
    title: str = Field(default="TerraSpectra Geospatial Inference API", description="OpenAPI title")
    version: str = Field(default="0.1.0", description="API version")
    max_upload_size_bytes: int = Field(default=524_288_000, description="Maximum upload size (500 MB)")
    allowed_origins: List[str] = Field(
        default=["http://localhost:5173", "http://127.0.0.1:5173"],
        description="CORS allowed origins for frontend"
    )


class AppConfig(BaseModel):
    """Root configuration holding all sub-system configs."""
    data: DataConfig = Field(default_factory=DataConfig)
    preprocess: PreprocessConfig = Field(default_factory=PreprocessConfig)
    model: ModelConfig = Field(default_factory=ModelConfig)
    training: TrainingConfig = Field(default_factory=TrainingConfig)
    api: APIConfig = Field(default_factory=APIConfig)

    @classmethod
    def from_yaml(cls, yaml_path: Path | str) -> AppConfig:
        """Load configuration from a YAML file."""
        path = Path(yaml_path)
        if not path.is_file():
            raise FileNotFoundError(f"Configuration file not found: {path}")
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        return cls(**data)

    def to_yaml(self, yaml_path: Path | str) -> None:
        """Save configuration to a YAML file."""
        path = Path(yaml_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        # Convert models to dict with serializable types
        data = self.model_dump(mode="json")
        with open(path, "w", encoding="utf-8") as f:
            yaml.safe_dump(data, f, default_flow_style=False, sort_keys=False)
