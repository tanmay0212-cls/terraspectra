"""
Custom exception definitions for the TerraSpectra system.
"""


class TerraSpectraError(Exception):
    """Base exception class for all TerraSpectra errors."""
    pass


class InvalidCubeError(TerraSpectraError):
    """Raised when an input hyperspectral cube is malformed, has invalid dimensions, or contains NaN/Inf values."""
    pass


class IncompatibleBandsError(TerraSpectraError):
    """Raised when the number of spectral bands does not match expected calibration or model input dimensions."""
    pass


class SpatialLeakageError(TerraSpectraError):
    """Raised when a train/test split violation or spatial overlap is detected."""
    pass


class ModelInferenceError(TerraSpectraError):
    """Raised when a model fails during preprocessing, forward pass, or post-processing."""
    pass


class DatasetDownloadError(TerraSpectraError):
    """Raised when an automated dataset fetch or checksum verification fails."""
    pass
