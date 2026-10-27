"""Unit tests verifying SSIM re-exports and backwards compatibility."""

from __future__ import annotations

import numpy as np
import pytest

import dataset_curator
import dataset_curator.metrics
from dataset_curator.metrics import ssim
from dataset_curator.metrics.ssim import (
    SimilarityResult,
    check_image_similarity,
    compare_images_ssim,
    compute_ssim,
)


class TestSsimReExports:
    """Verifies that SSIM functions and dataclasses are accessible from metrics.ssim."""

    def test_imports_match(self) -> None:
        """Verify re-exported objects are identical to module attributes."""
        assert ssim.compute_ssim is compute_ssim
        assert ssim.compare_images_ssim is compare_images_ssim
        assert ssim.check_image_similarity is check_image_similarity
        assert ssim.SimilarityResult is SimilarityResult

    def test_package_level_getattr(self) -> None:
        """Verify package-level lazy imports via __getattr__."""
        assert dataset_curator.compute_ssim is compute_ssim
        assert dataset_curator.SimilarityResult is SimilarityResult
        assert dataset_curator.metrics.compute_ssim is compute_ssim

        with pytest.raises(AttributeError):
            _ = dataset_curator.non_existent_symbol

        with pytest.raises(AttributeError):
            _ = dataset_curator.metrics.non_existent_symbol

    def test_basic_computation(self) -> None:
        """Sanity check compute_ssim from re-exported module."""
        img = np.full((32, 32, 3), 128, dtype=np.uint8)
        score = compute_ssim(img, img)
        assert score == pytest.approx(1.0, rel=1e-4)
