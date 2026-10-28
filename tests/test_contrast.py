"""Unit tests for dynamic range and contrast evaluation."""

from __future__ import annotations

import numpy as np
import pytest

from dataset.curator.metrics import ContrastFilterStage, compute_luminance_std


class TestContrastMetric:
    """Tests for compute_luminance_std function."""

    def test_flat_image_has_zero_std(self) -> None:
        """Constant image has standard deviation 0.0."""
        gray = np.full((100, 100), 128, dtype=np.uint8)
        assert compute_luminance_std(gray) == pytest.approx(0.0)

    def test_high_contrast_image_std(self) -> None:
        """Binary high contrast image (half 0, half 200) has expected std."""
        gray = np.zeros((100, 100), dtype=np.uint8)
        gray[:50, :] = 200
        # std for half 0, half 200 is 100.0
        assert compute_luminance_std(gray) == pytest.approx(100.0)


class TestContrastFilterStage:
    """Tests for ContrastFilterStage thresholding."""

    def test_rejects_low_contrast(self) -> None:
        """Rejects image with standard deviation below 40.0."""
        stage = ContrastFilterStage(min_luminance_std=40.0)
        # Low contrast image around 128 with variance +/- 5
        gray = (
            np.random.RandomState(42)
            .randint(123, 133, size=(100, 100))
            .astype(np.uint8)
        )
        rgb = np.stack([gray] * 3, axis=-1)

        result = stage.process(rgb, gray)
        assert result.passed is False
        assert result.rejection_reason == "low_contrast"
        assert result.metric_value < 40.0

    def test_accepts_high_contrast(self) -> None:
        """Accepts image with standard deviation >= 40.0."""
        stage = ContrastFilterStage(min_luminance_std=40.0)
        # Uniform spread between 30 and 220 has std ~= (220-30)/sqrt(12) ~= 54.8
        gray = np.tile(np.linspace(30, 220, 100, dtype=np.uint8), (100, 1))
        rgb = np.stack([gray] * 3, axis=-1)

        result = stage.process(rgb, gray)
        assert result.passed is True
        assert result.rejection_reason is None
        assert result.metric_value > 40.0
