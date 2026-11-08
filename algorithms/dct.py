"""Block 8x8 discrete cosine transform implemented with a transform matrix."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .common import retain_largest_coefficients

BLOCK_SIZE = 8


@dataclass(frozen=True)
class DCTResult:
    reconstructed: np.ndarray
    coefficients: np.ndarray
    retained_count: int
    total_count: int


def build_dct_matrix(size: int = BLOCK_SIZE) -> np.ndarray:
    """Build an orthonormal DCT-II transform matrix from the definition."""

    matrix = np.empty((size, size), dtype=np.float64)
    for frequency in range(size):
        alpha = np.sqrt(1 / size) if frequency == 0 else np.sqrt(2 / size)
        for position in range(size):
            matrix[frequency, position] = alpha * np.cos(
                np.pi * (2 * position + 1) * frequency / (2 * size)
            )
    return matrix


DCT_MATRIX = build_dct_matrix()


def _validate_image(image: np.ndarray) -> None:
    if image.ndim != 2:
        raise ValueError("DCT expects a two-dimensional grayscale image")
    height, width = image.shape
    if height % BLOCK_SIZE or width % BLOCK_SIZE:
        raise ValueError("Image dimensions must be divisible by 8")


def forward_dct(image: np.ndarray) -> np.ndarray:
    """Apply own matrix DCT to every 8x8 block of a grayscale image."""

    _validate_image(image)
    source = image.astype(np.float64, copy=False) - 128.0
    coefficients = np.empty_like(source)

    for row in range(0, source.shape[0], BLOCK_SIZE):
        for column in range(0, source.shape[1], BLOCK_SIZE):
            block = source[row : row + BLOCK_SIZE, column : column + BLOCK_SIZE]
            coefficients[row : row + BLOCK_SIZE, column : column + BLOCK_SIZE] = (
                DCT_MATRIX @ block @ DCT_MATRIX.T
            )
    return coefficients


def inverse_dct(coefficients: np.ndarray) -> np.ndarray:
    """Apply the inverse matrix DCT to every 8x8 coefficient block."""

    _validate_image(coefficients)
    restored = np.empty_like(coefficients, dtype=np.float64)

    for row in range(0, coefficients.shape[0], BLOCK_SIZE):
        for column in range(0, coefficients.shape[1], BLOCK_SIZE):
            block = coefficients[
                row : row + BLOCK_SIZE, column : column + BLOCK_SIZE
            ]
            restored[row : row + BLOCK_SIZE, column : column + BLOCK_SIZE] = (
                DCT_MATRIX.T @ block @ DCT_MATRIX
            )
    return np.clip(restored + 128.0, 0, 255)


def compress_dct(image: np.ndarray, keep_ratio: float) -> DCTResult:
    """Transform, reduce coefficients and reconstruct an image with DCT."""

    coefficients = forward_dct(image)
    reduced, retained_count = retain_largest_coefficients(coefficients, keep_ratio)
    reconstructed = inverse_dct(reduced)
    return DCTResult(
        reconstructed=reconstructed,
        coefficients=reduced,
        retained_count=retained_count,
        total_count=coefficients.size,
    )
