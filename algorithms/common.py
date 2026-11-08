"""Shared coefficient reduction helpers.

The project intentionally contains no calls to ready-made DCT/DWT routines.
NumPy is used only for arrays, sorting and matrix multiplication.
"""

from __future__ import annotations

import numpy as np


def retain_largest_coefficients(
    coefficients: np.ndarray, keep_ratio: float
) -> tuple[np.ndarray, int]:
    """Keep exactly the requested share of largest-magnitude coefficients.

    Parameters
    ----------
    coefficients:
        Two-dimensional transform coefficient array.
    keep_ratio:
        Fraction from 0 to 1. At least one coefficient is retained.
    """

    if coefficients.ndim != 2:
        raise ValueError("Coefficient array must be two-dimensional")
    if not 0 < keep_ratio <= 1:
        raise ValueError("keep_ratio must be in the interval (0, 1]")

    flat = coefficients.ravel()
    keep_count = max(1, min(flat.size, int(round(flat.size * keep_ratio))))

    if keep_count == flat.size:
        return coefficients.copy(), keep_count

    indices = np.argpartition(np.abs(flat), -keep_count)[-keep_count:]
    reduced = np.zeros_like(flat)
    reduced[indices] = flat[indices]
    return reduced.reshape(coefficients.shape), keep_count
