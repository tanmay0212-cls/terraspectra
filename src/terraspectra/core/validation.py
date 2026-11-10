"""
Array and tensor validation utilities for hyperspectral datacubes.
"""

from __future__ import annotations
from typing import Optional, Tuple
import numpy as np
from terraspectra.core.exceptions import InvalidCubeError, IncompatibleBandsError


def check_nan_inf(arr: np.ndarray, name: str = "array", raise_on_error: bool = True) -> Tuple[int, int]:
    """
    Count NaN and infinite values in a numpy array.
    
    Args:
        arr: Target numpy array.
        name: Name of array for error reporting.
        raise_on_error: If True, raises InvalidCubeError when any NaN or Inf is found.
        
    Returns:
        Tuple of (nan_count, inf_count).
    """
    if not isinstance(arr, np.ndarray):
        raise TypeError(f"Expected numpy.ndarray for '{name}', got {type(arr).__name__}")

    nan_count = int(np.isnan(arr).sum())
    inf_count = int(np.isinf(arr).sum())

    if raise_on_error and (nan_count > 0 or inf_count > 0):
        raise InvalidCubeError(
            f"Array '{name}' failed sanity check: found {nan_count} NaN values and {inf_count} Inf values."
        )

    return nan_count, inf_count


def validate_hyperspectral_cube(
    arr: np.ndarray,
    expected_bands: Optional[int] = None,
    allow_nan: bool = False,
    min_val: Optional[float] = None,
    max_val: Optional[float] = None,
) -> None:
    """
    Validate dimensionality, dtypes, and value range of a hyperspectral datacube.
    
    Expected format: 3D array of shape (height, width, bands).
    """
    if not isinstance(arr, np.ndarray):
        raise TypeError(f"Hyperspectral cube must be a numpy.ndarray, got {type(arr).__name__}")

    if arr.ndim != 3:
        raise InvalidCubeError(f"Expected 3D datacube (height, width, bands), but array has ndim={arr.ndim} and shape={arr.shape}")

    h, w, b = arr.shape
    if h < 1 or w < 1 or b < 1:
        raise InvalidCubeError(f"Invalid cube dimensions: {arr.shape}. All dimensions must be >= 1.")

    if not np.issubdtype(arr.dtype, np.number):
        raise InvalidCubeError(f"Cube dtype must be numeric, got {arr.dtype}")

    if not allow_nan:
        check_nan_inf(arr, name="hyperspectral_cube", raise_on_error=True)

    if expected_bands is not None and b != expected_bands:
        raise IncompatibleBandsError(
            f"Spectral band count mismatch: cube has {b} bands, expected {expected_bands}."
        )

    if min_val is not None:
        actual_min = float(np.nanmin(arr))
        if actual_min < min_val:
            raise InvalidCubeError(
                f"Array values below permissible minimum {min_val}: minimum observed was {actual_min:.4f}"
            )

    if max_val is not None:
        actual_max = float(np.nanmax(arr))
        if actual_max > max_val:
            raise InvalidCubeError(
                f"Array values above permissible maximum {max_val}: maximum observed was {actual_max:.4f}"
            )


def validate_wavelengths(wavelengths: np.ndarray, expected_bands: int) -> None:
    """Validate 1D wavelength sequence in nanometers."""
    if not isinstance(wavelengths, np.ndarray):
        raise TypeError(f"Wavelengths must be a numpy.ndarray, got {type(wavelengths).__name__}")

    if wavelengths.ndim != 1:
        raise InvalidCubeError(f"Wavelengths must be a 1D array, got ndim={wavelengths.ndim}")

    if len(wavelengths) != expected_bands:
        raise IncompatibleBandsError(
            f"Wavelengths length ({len(wavelengths)}) does not match expected band count ({expected_bands})"
        )

    # Check monotonicity
    if not np.all(np.diff(wavelengths) > 0):
        raise InvalidCubeError("Wavelengths must be strictly monotonically increasing")

    # Range check: hyperspectral VNIR-SWIR typically falls in 350 nm - 2600 nm
    if wavelengths[0] < 300.0 or wavelengths[-1] > 3000.0:
        raise InvalidCubeError(
            f"Wavelengths range [{wavelengths[0]:.1f}, {wavelengths[-1]:.1f}] nm is outside valid optical range [300, 3000] nm"
        )


def validate_patch_dimensions(
    patch: np.ndarray,
    expected_size: int,
    expected_channels: Optional[int] = None
) -> None:
    """Validate a spatial patch of shape (patch_size, patch_size, channels)."""
    if patch.ndim != 3:
        raise InvalidCubeError(f"Patch must be 3D (H, W, C), got ndim={patch.ndim}")

    h, w, c = patch.shape
    if h != expected_size or w != expected_size:
        raise InvalidCubeError(f"Patch spatial dimensions ({h}, {w}) do not match expected size ({expected_size}, {expected_size})")

    if expected_channels is not None and c != expected_channels:
        raise IncompatibleBandsError(f"Patch channel count {c} does not match expected {expected_channels}")
