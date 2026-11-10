"""
Core module for TerraSpectra: types, configurations, exceptions, logging, diagnostics, and validation.
"""

from terraspectra.core.config import (
    AppConfig,
    DataConfig,
    PreprocessConfig,
    ModelConfig,
    TrainingConfig,
    APIConfig,
)
from terraspectra.core.types import (
    HyperspectralCube,
    SpectralSignature,
    PredictionOutput,
)
from terraspectra.core.exceptions import (
    TerraSpectraError,
    InvalidCubeError,
    IncompatibleBandsError,
    SpatialLeakageError,
    ModelInferenceError,
    DatasetDownloadError,
)
from terraspectra.core.utils import get_project_root, set_seed, ensure_dir
from terraspectra.core.logging import setup_logger
from terraspectra.core.diagnostics import run_environment_diagnostics, format_diagnostics_report
from terraspectra.core.validation import (
    check_nan_inf,
    validate_hyperspectral_cube,
    validate_wavelengths,
    validate_patch_dimensions,
)

__all__ = [
    "AppConfig",
    "DataConfig",
    "PreprocessConfig",
    "ModelConfig",
    "TrainingConfig",
    "APIConfig",
    "HyperspectralCube",
    "SpectralSignature",
    "PredictionOutput",
    "TerraSpectraError",
    "InvalidCubeError",
    "IncompatibleBandsError",
    "SpatialLeakageError",
    "ModelInferenceError",
    "DatasetDownloadError",
    "get_project_root",
    "set_seed",
    "ensure_dir",
    "setup_logger",
    "run_environment_diagnostics",
    "format_diagnostics_report",
    "check_nan_inf",
    "validate_hyperspectral_cube",
    "validate_wavelengths",
    "validate_patch_dimensions",
]
