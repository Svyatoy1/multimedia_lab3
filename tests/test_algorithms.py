from __future__ import annotations

import math
import unittest

import numpy as np

from algorithms import (
    compress_dct,
    compress_dwt,
    forward_dct,
    forward_haar_2d,
    inverse_dct,
    inverse_haar_2d,
)
from metrics import psnr


class TransformTests(unittest.TestCase):
    def setUp(self) -> None:
        rng = np.random.default_rng(42)
        self.image = rng.integers(0, 256, size=(64, 64)).astype(np.float64)

    def test_dct_round_trip_without_reduction(self) -> None:
        restored = inverse_dct(forward_dct(self.image))
        self.assertLess(float(np.max(np.abs(restored - self.image))), 1e-9)

    def test_haar_round_trip_without_reduction(self) -> None:
        restored = inverse_haar_2d(forward_haar_2d(self.image, 3), 3)
        self.assertLess(float(np.max(np.abs(restored - self.image))), 1e-9)

    def test_requested_coefficient_count_is_retained(self) -> None:
        dct_result = compress_dct(self.image, 0.20)
        dwt_result = compress_dwt(self.image, 0.20, 3)
        expected = round(self.image.size * 0.20)
        self.assertEqual(dct_result.retained_count, expected)
        self.assertEqual(dwt_result.retained_count, expected)

    def test_psnr_for_identical_images_is_infinite(self) -> None:
        self.assertTrue(math.isinf(psnr(self.image, self.image.copy())))

    def test_compressed_images_keep_shape_and_range(self) -> None:
        for result in (compress_dct(self.image, 0.05), compress_dwt(self.image, 0.05, 3)):
            self.assertEqual(result.reconstructed.shape, self.image.shape)
            self.assertGreaterEqual(float(result.reconstructed.min()), 0)
            self.assertLessEqual(float(result.reconstructed.max()), 255)


if __name__ == "__main__":
    unittest.main()
