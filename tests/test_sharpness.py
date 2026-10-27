"""Unit tests for focus sharpness using Laplacian variance."""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from dataset_curator.metrics.sharpness import (
    SharpnessFilterStage,
    compute_laplacian_variance,
)


class TestSharpnessMetric:
    """Tests for compute_laplacian_variance function."""

    def test_smooth_image_low_variance(self) -> None:
        """Completely flat or heavily smoothed image has minimal Laplacian variance."""
        gray = np.full((100, 100), 128, dtype=np.uint8)
        assert compute_laplacian_variance(gray) == pytest.approx(0.0)

    def test_sharp_edges_high_variance(self) -> None:
        """Checkerboard image with sharp edges has very high Laplacian variance."""
        y, x = np.mgrid[0:100, 0:100]
        checker = (((x // 4) + (y // 4)) % 2) * 200
        gray = checker.astype(np.uint8)

        var = compute_laplacian_variance(gray)
        assert var > 1000.0


class TestSharpnessFilterStage:
    """Tests for SharpnessFilterStage thresholding."""

    def test_rejects_blurred_image(self) -> None:
        """Rejects image whose edges are blurred below the 120.0 threshold."""
        stage = SharpnessFilterStage(min_laplacian_var=120.0)
        # Create subtle ramp and blur it
        y, x = np.mgrid[0:100, 0:100]
        ramp = ((x + y) / 2).astype(np.uint8)
        blurred = cv2.GaussianBlur(ramp, (15, 15), 0)
        rgb = np.stack([blurred] * 3, axis=-1)

        result = stage.process(rgb, blurred)
        assert result.passed is False
        assert result.rejection_reason == "blurry"
        assert result.metric_value < 120.0

    def test_accepts_sharp_image(self) -> None:
        """Accepts image with fine details and sharp edges exceeding 120.0 threshold."""
        stage = SharpnessFilterStage(min_laplacian_var=120.0)
        y, x = np.mgrid[0:100, 0:100]
        pattern = (((x // 8) + (y // 8)) % 2) * 180 + 30
        gray = pattern.astype(np.uint8)
        rgb = np.stack([gray] * 3, axis=-1)

        result = stage.process(rgb, gray)
        assert result.passed is True
        assert result.rejection_reason is None
        assert result.metric_value >= 120.0
