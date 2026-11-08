"""Image quality and coefficient statistics."""

from __future__ import annotations

import math

import numpy as np


def mean_squared_error(original: np.ndarray, restored: np.ndarray) -> float:
    if original.shape != restored.shape:
        raise ValueError("Images must have identical dimensions")
    difference = original.astype(np.float64) - restored.astype(np.float64)
    return float(np.mean(difference**2))


def psnr(original: np.ndarray, restored: np.ndarray, peak: float = 255.0) -> float:
    """Calculate peak signal-to-noise ratio in decibels."""

    mse = mean_squared_error(original, restored)
    if mse == 0:
        return math.inf
    return 10.0 * math.log10((peak**2) / mse)


def coefficient_compression_ratio(total_count: int, retained_count: int) -> float:
    """Return a coefficient-based ratio, not an encoded file size ratio."""

    if retained_count <= 0:
        return math.inf
    return total_count / retained_count


def format_psnr(value: float) -> str:
    return "∞" if math.isinf(value) else f"{value:.2f} dB"
