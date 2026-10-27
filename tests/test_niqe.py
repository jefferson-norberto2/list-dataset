"""Unit tests for the NIQE quality filter stage using PyIQA."""

from __future__ import annotations

import math

import numpy as np

from dataset_curator.metrics.niqe import NIQEFilterStage


class TestNIQEFilterStage:
    """Tests for NIQEFilterStage with PyIQA integration."""

    def test_niqe_device_and_execution(self) -> None:
        """Evaluates NIQE on synthetic high-contrast pattern."""
        stage = NIQEFilterStage(max_niqe=15.0, device="auto")

        # Create smooth gradient pattern (512x512x3)
        y, x = np.mgrid[0:512, 0:512]
        img = (((x + y) / (512 + 512)) * 200 + 20).astype(np.uint8)
        rgb = np.stack([img, img, img], axis=-1)
        gray = img

        result = stage.process(rgb, gray)
        assert not math.isnan(result.metric_value)
        assert result.metric_value > 0.0
        assert result.stage_name == "niqe"

    def test_niqe_rejects_above_threshold(self) -> None:
        """Rejects images whose NIQE exceeds threshold."""
        stage = NIQEFilterStage(max_niqe=1.0, device="cpu")
        passed, reason = stage.evaluate(3.8)
        assert passed is False
        assert reason == "high_niqe"

    def test_niqe_accepts_below_threshold(self) -> None:
        """Accepts images whose NIQE is below threshold."""
        stage = NIQEFilterStage(max_niqe=4.0, device="cpu")
        passed, reason = stage.evaluate(3.1)
        assert passed is True
        assert reason is None

    def test_niqe_rejects_nan(self) -> None:
        """Rejects images returning NaN score."""
        stage = NIQEFilterStage(max_niqe=3.5, device="cpu")
        passed, reason = stage.evaluate(float("nan"))
        assert passed is False
        assert reason == "niqe_nan"
