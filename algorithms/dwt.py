"""Multi-level two-dimensional Haar wavelet transform implemented manually."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .common import retain_largest_coefficients

SQRT_TWO = np.sqrt(2.0)


@dataclass(frozen=True)
class DWTResult:
    reconstructed: np.ndarray
    coefficients: np.ndarray
    retained_count: int
    total_count: int
    levels: int


def _validate_shape(array: np.ndarray, levels: int) -> None:
    if array.ndim != 2:
        raise ValueError("DWT expects a two-dimensional grayscale image")
    if levels < 1:
        raise ValueError("At least one DWT level is required")
    divisor = 2**levels
    if array.shape[0] % divisor or array.shape[1] % divisor:
        raise ValueError(f"Image dimensions must be divisible by {divisor}")


def _haar_forward_region(region: np.ndarray) -> np.ndarray:
    """Perform one separable 2D Haar step: rows first, columns second."""

    row_low = (region[:, 0::2] + region[:, 1::2]) / SQRT_TWO
    row_high = (region[:, 0::2] - region[:, 1::2]) / SQRT_TWO
    row_transformed = np.concatenate((row_low, row_high), axis=1)

    column_low = (row_transformed[0::2, :] + row_transformed[1::2, :]) / SQRT_TWO
    column_high = (row_transformed[0::2, :] - row_transformed[1::2, :]) / SQRT_TWO
    return np.concatenate((column_low, column_high), axis=0)


def _haar_inverse_region(region: np.ndarray) -> np.ndarray:
    """Undo one separable 2D Haar step in reverse order."""

    height, width = region.shape
    column_low = region[: height // 2, :]
    column_high = region[height // 2 :, :]
    column_restored = np.empty_like(region)
    column_restored[0::2, :] = (column_low + column_high) / SQRT_TWO
    column_restored[1::2, :] = (column_low - column_high) / SQRT_TWO

    row_low = column_restored[:, : width // 2]
    row_high = column_restored[:, width // 2 :]
    restored = np.empty_like(region)
    restored[:, 0::2] = (row_low + row_high) / SQRT_TWO
    restored[:, 1::2] = (row_low - row_high) / SQRT_TWO
    return restored


def forward_haar_2d(image: np.ndarray, levels: int = 3) -> np.ndarray:
    """Apply a multi-level Haar DWT without a wavelet library."""

    _validate_shape(image, levels)
    coefficients = image.astype(np.float64, copy=True)

    for level in range(levels):
        height = image.shape[0] // (2**level)
        width = image.shape[1] // (2**level)
        region = coefficients[:height, :width].copy()
        coefficients[:height, :width] = _haar_forward_region(region)
    return coefficients


def inverse_haar_2d(coefficients: np.ndarray, levels: int = 3) -> np.ndarray:
    """Apply the inverse multi-level Haar transform."""

    _validate_shape(coefficients, levels)
    restored = coefficients.astype(np.float64, copy=True)

    for level in reversed(range(levels)):
        height = coefficients.shape[0] // (2**level)
        width = coefficients.shape[1] // (2**level)
        region = restored[:height, :width].copy()
        restored[:height, :width] = _haar_inverse_region(region)
    return np.clip(restored, 0, 255)


def compress_dwt(
    image: np.ndarray, keep_ratio: float, levels: int = 3
) -> DWTResult:
    """Transform, reduce coefficients and reconstruct an image with Haar DWT."""

    coefficients = forward_haar_2d(image, levels)
    reduced, retained_count = retain_largest_coefficients(coefficients, keep_ratio)
    reconstructed = inverse_haar_2d(reduced, levels)
    return DWTResult(
        reconstructed=reconstructed,
        coefficients=reduced,
        retained_count=retained_count,
        total_count=coefficients.size,
        levels=levels,
    )
