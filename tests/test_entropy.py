"""Unit tests for Shannon entropy evaluation."""

from __future__ import annotations

import numpy as np
import pytest

from dataset_curator.curator.metrics.entropy import EntropyFilterStage, compute_shannon_entropy


class TestEntropyMetric:
    """Tests for compute_shannon_entropy function."""

    def test_constant_image_zero_entropy(self) -> None:
        """Single tone image has zero information entropy."""
        gray = np.full((100, 100), 50, dtype=np.uint8)
        assert compute_shannon_entropy(gray) == pytest.approx(0.0)

    def test_uniform_distribution_maximum_entropy(self) -> None:
        """Uniform distribution across all 256 pixel values has theoretical maximum entropy of 8.0 bits."""
        # 256 * 100 = 25,600 pixels containing equal counts of 0..255
        values = np.arange(256, dtype=np.uint8)
        gray = np.repeat(values, 100).reshape(160, 160)
        assert compute_shannon_entropy(gray) == pytest.approx(8.0, rel=1e-5)


class TestEntropyFilterStage:
    """Tests for EntropyFilterStage thresholding."""

    def test_rejects_low_entropy(self) -> None:
        """Rejects image with entropy below 7.0 bits/pixel."""
        stage = EntropyFilterStage(min_entropy=7.0)
        # Image using only 4 distinct values: max entropy = log2(4) = 2.0 bits
        gray = (
            np.random.RandomState(42)
            .choice([50, 100, 150, 200], size=(100, 100))
            .astype(np.uint8)
        )
        rgb = np.stack([gray] * 3, axis=-1)

        result = stage.process(rgb, gray)
        assert result.passed is False
        assert result.rejection_reason == "low_entropy"
        assert result.metric_value < 7.0

    def test_accepts_high_entropy(self) -> None:
        """Accepts image with entropy >= 7.0 bits/pixel."""
        stage = EntropyFilterStage(min_entropy=7.0)
        # Random uniform across 200 distinct values: log2(200) ~= 7.64 bits
        gray = np.random.RandomState(42).randint(
            20, 220, size=(200, 200), dtype=np.uint8
        )
        rgb = np.stack([gray] * 3, axis=-1)

        result = stage.process(rgb, gray)
        assert result.passed is True
        assert result.rejection_reason is None
        assert result.metric_value >= 7.0
