"""Unit tests for clipping and exposure ratio calculation and filter stage."""

from __future__ import annotations

import numpy as np
import pytest

from dataset_curator.curator.metrics.clipping import ClippingFilterStage, compute_clipping_ratio


class TestClippingMetric:
    """Tests for compute_clipping_ratio function."""

    def test_zero_clipping_in_midrange(self) -> None:
        """Image with pixel intensities between 2 and 253 should have 0.0 clipping ratio."""
        img = np.full((100, 100), 128, dtype=np.uint8)
        ratio = compute_clipping_ratio(img, low_thresh=1, high_thresh=254)
        assert ratio == pytest.approx(0.0)

    def test_pure_underexposed_image(self) -> None:
        """Image filled with 0 (<= 1) should have 1.0 clipping ratio."""
        img = np.zeros((50, 50), dtype=np.uint8)
        ratio = compute_clipping_ratio(img, low_thresh=1, high_thresh=254)
        assert ratio == pytest.approx(1.0)

    def test_pure_overexposed_image(self) -> None:
        """Image filled with 255 (>= 254) should have 1.0 clipping ratio."""
        img = np.full((50, 50), 255, dtype=np.uint8)
        ratio = compute_clipping_ratio(img, low_thresh=1, high_thresh=254)
        assert ratio == pytest.approx(1.0)

    def test_boundary_values(self) -> None:
        """Pixel values of 1 and 254 must be counted as clipped; 2 and 253 must not."""
        img = np.array(
            [
                [0, 1, 2],  # 0 and 1 clipped (under)
                [100, 128, 200],  # none clipped
                [253, 254, 255],  # 254 and 255 clipped (over)
            ],
            dtype=np.uint8,
        )
        # 4 clipped out of 9 pixels
        expected_ratio = 4.0 / 9.0
        ratio = compute_clipping_ratio(img, low_thresh=1, high_thresh=254)
        assert ratio == pytest.approx(expected_ratio)

    def test_exact_one_percent_clipping(self) -> None:
        """Verify accurate fractional clipping computation."""
        # 100x100 = 10,000 pixels. 100 pixels clipped = exactly 1.0% (0.01)
        img = np.full((100, 100), 128, dtype=np.uint8)
        img[0, :] = 0  # 100 pixels set to 0
        ratio = compute_clipping_ratio(img, low_thresh=1, high_thresh=254)
        assert ratio == pytest.approx(0.01)

    def test_three_channel_rgb_clipping(self) -> None:
        """Pixel with any saturated channel counts towards clipped total in 3D mode."""
        img = np.full((10, 10, 3), 128, dtype=np.uint8)
        # Saturate red channel on top-left pixel
        img[0, 0, 0] = 255
        # Saturate blue channel on bottom-right pixel (underexposed)
        img[9, 9, 2] = 0
        ratio = compute_clipping_ratio(img, low_thresh=1, high_thresh=254)
        assert ratio == pytest.approx(2 / 100)


class TestClippingFilterStage:
    """Tests for ClippingFilterStage evaluation logic."""

    def test_accepts_clean_image(self) -> None:
        """Stage accepts an image with clipping below threshold."""
        stage = ClippingFilterStage(max_clipping_ratio=0.01)
        gray = np.full((100, 100), 128, dtype=np.uint8)
        rgb = np.stack([gray] * 3, axis=-1)

        result = stage.process(rgb, gray)
        assert result.passed is True
        assert result.rejection_reason is None
        assert result.metric_value == pytest.approx(0.0)

    def test_rejects_exceeded_clipping(self) -> None:
        """Stage rejects an image when clipping ratio strictly exceeds threshold."""
        stage = ClippingFilterStage(max_clipping_ratio=0.01)
        gray = np.full((100, 100), 128, dtype=np.uint8)
        # 150 pixels clipped = 1.5% > 1.0%
        gray[:1, :100] = 0
        gray[1, :50] = 255
        rgb = np.stack([gray] * 3, axis=-1)

        result = stage.process(rgb, gray)
        assert result.passed is False
        assert result.rejection_reason == "clipping_exceeded"
        assert result.metric_value == pytest.approx(0.015)
