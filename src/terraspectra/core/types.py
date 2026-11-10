"""
Core data types and contracts for TerraSpectra.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
import numpy as np


@dataclass
class HyperspectralCube:
    """
    Represents a hyperspectral datacube with associated spectral and geospatial metadata.
    
    Attributes:
        data: 3D numpy array of shape (height, width, bands), normalized reflectance [0.0, 1.0].
        wavelengths: 1D array of center wavelengths in nanometers (nm).
        sensor_name: Identifier of the imaging spectrometer (e.g. 'AVIRIS', 'ROSIS', 'Headwall-VNIR').
        metadata: Arbitrary geospatial or acquisition metadata.
        mask: Optional binary mask of valid pixels (e.g. canopy mask vs background soil).
    """
    data: np.ndarray
    wavelengths: np.ndarray
    sensor_name: str = "Unknown"
    metadata: Dict[str, Any] = field(default_factory=dict)
    mask: Optional[np.ndarray] = None

    def __post_init__(self):
        if self.data.ndim != 3:
            raise ValueError(f"Expected 3D array (H, W, B), got shape {self.data.shape}")
        if len(self.wavelengths) != self.data.shape[2]:
            raise ValueError(
                f"Wavelengths length ({len(self.wavelengths)}) does not match bands ({self.data.shape[2]})"
            )

    @property
    def height(self) -> int:
        return self.data.shape[0]

    @property
    def width(self) -> int:
        return self.data.shape[1]

    @property
    def num_bands(self) -> int:
        return self.data.shape[2]

    @property
    def shape(self) -> Tuple[int, int, int]:
        return self.data.shape

    def get_rgb_composite(self, red_nm: float = 660.0, green_nm: float = 560.0, blue_nm: float = 480.0) -> np.ndarray:
        """
        Synthesize a false-color or true-color RGB image by finding closest matching spectral bands.
        
        Returns:
            RGB array of shape (H, W, 3) scaled to uint8 [0, 255].
        """
        r_idx = int(np.argmin(np.abs(self.wavelengths - red_nm)))
        g_idx = int(np.argmin(np.abs(self.wavelengths - green_nm)))
        b_idx = int(np.argmin(np.abs(self.wavelengths - blue_nm)))

        rgb = np.stack([self.data[:, :, r_idx], self.data[:, :, g_idx], self.data[:, :, b_idx]], axis=-1)
        # Normalize per-channel to [0, 255]
        rgb_norm = np.zeros_like(rgb, dtype=np.uint8)
        for i in range(3):
            c = rgb[:, :, i]
            c_min, c_max = np.percentile(c, 2), np.percentile(c, 98)
            if c_max > c_min:
                scaled = np.clip((c - c_min) / (c_max - c_min), 0.0, 1.0) * 255.0
            else:
                scaled = np.zeros_like(c)
            rgb_norm[:, :, i] = scaled.astype(np.uint8)
        return rgb_norm

    def get_spectrum(self, y: int, x: int) -> np.ndarray:
        """Extract continuous spectral reflectance curve for a specific (y, x) pixel."""
        return self.data[y, x, :]


@dataclass
class SpectralSignature:
    """Represents a continuous spectral reflectance profile for an individual pixel or region."""
    wavelengths: np.ndarray
    reflectance: np.ndarray
    label: Optional[str] = None
    confidence: Optional[float] = None


@dataclass
class PredictionOutput:
    """Output contract for spatial disease inference."""
    class_map: np.ndarray
    confidence_map: np.ndarray
    class_names: List[str]
    metadata: Dict[str, Any] = field(default_factory=dict)
