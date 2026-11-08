"""Loading, normalising and converting grayscale images."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

WORK_SIZE = 512


def load_grayscale_image(path: str | Path, size: int = WORK_SIZE) -> np.ndarray:
    """Load an image, convert it to grayscale and fit it to a square work area."""

    with Image.open(path) as source:
        grayscale = ImageOps.grayscale(source)
        prepared = ImageOps.fit(
            grayscale,
            (size, size),
            method=Image.Resampling.LANCZOS,
        )
    return np.asarray(prepared, dtype=np.float64)


def array_to_image(array: np.ndarray) -> Image.Image:
    pixels = np.clip(np.rint(array), 0, 255).astype(np.uint8)
    return Image.fromarray(pixels, mode="L")
