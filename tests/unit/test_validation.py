"""
Test hyperspectral array validation, NaN/Inf checks, and dimension assertions.
"""

import numpy as np
import pytest
from terraspectra.core.exceptions import InvalidCubeError, IncompatibleBandsError
from terraspectra.core.validation import (
    check_nan_inf,
    validate_hyperspectral_cube,
    validate_wavelengths,
    validate_patch_dimensions,
)


def test_valid_datacube():
    # Valid synthetic cube: (H=20, W=20, B=50) with values in [0.0, 1.0]
    cube = np.random.uniform(0.1, 0.9, size=(20, 20, 50)).astype(np.float32)
    validate_hyperspectral_cube(cube, expected_bands=50, min_val=0.0, max_val=1.0)


def test_datacube_wrong_dimensions():
    # 2D array instead of 3D
    arr_2d = np.ones((10, 10), dtype=np.float32)
    with pytest.raises(InvalidCubeError):
        validate_hyperspectral_cube(arr_2d)


def test_datacube_nan_detection():
    cube = np.ones((10, 10, 10), dtype=np.float32)
    cube[2, 3, 4] = np.nan

    nan_count, inf_count = check_nan_inf(cube, raise_on_error=False)
    assert nan_count == 1
    assert inf_count == 0

    with pytest.raises(InvalidCubeError):
        validate_hyperspectral_cube(cube, allow_nan=False)


def test_datacube_inf_detection():
    cube = np.ones((10, 10, 10), dtype=np.float32)
    cube[1, 1, 1] = np.inf

    nan_count, inf_count = check_nan_inf(cube, raise_on_error=False)
    assert nan_count == 0
    assert inf_count == 1

    with pytest.raises(InvalidCubeError):
        validate_hyperspectral_cube(cube)


def test_datacube_band_mismatch():
    cube = np.ones((10, 10, 30), dtype=np.float32)
    with pytest.raises(IncompatibleBandsError):
        validate_hyperspectral_cube(cube, expected_bands=50)


def test_valid_wavelengths():
    waves = np.linspace(400.0, 900.0, 50)
    validate_wavelengths(waves, expected_bands=50)


def test_non_monotonic_wavelengths():
    waves = np.array([400.0, 450.0, 430.0, 500.0])
    with pytest.raises(InvalidCubeError):
        validate_wavelengths(waves, expected_bands=4)


def test_patch_dimension_validation():
    patch = np.zeros((16, 16, 20), dtype=np.float32)
    validate_patch_dimensions(patch, expected_size=16, expected_channels=20)

    with pytest.raises(InvalidCubeError):
        validate_patch_dimensions(patch, expected_size=32)
