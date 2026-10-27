"""Example reference unit test for the ColorCastFilterStage."""

from __future__ import annotations

import numpy as np
import pytest

# Import from the example reference
from .new_metric_example import ColorCastFilterStage, compute_color_cast_ratio


class TestColorCastMetric:
    """Tests for compute_color_cast_ratio function."""

    def test_neutral_gray_zero_cast(self) -> None:
        """Neutral grayscale image should have approximately zero chromatic divergence."""
        gray = np.full((100, 100, 3), 128, dtype=np.uint8)
        cast = compute_color_cast_ratio(gray)
        assert cast == pytest.approx(0.0, abs=1.0)

    def test_heavy_tint_detected(self) -> None:
        """Heavily tinted monochromatic image must register high color cast."""
        # Pure intense yellow (R=255, G=255, B=0)
        yellow = np.zeros((100, 100, 3), dtype=np.uint8)
        yellow[:, :, 0] = 255
        yellow[:, :, 1] = 255
        cast = compute_color_cast_ratio(yellow)
        assert cast > 50.0


class TestColorCastFilterStage:
    """Tests for ColorCastFilterStage decision logic."""

    def test_accepts_natural_white_balance(self) -> None:
        """Natural balanced image passes the stage."""
        stage = ColorCastFilterStage(max_cast_dist=15.0)
        neutral = np.full((100, 100, 3), 128, dtype=np.uint8)
        gray = neutral[:, :, 0]

        result = stage.process(neutral, gray)
        assert result.passed is True
        assert result.rejection_reason is None

    def test_rejects_unnatural_tint(self) -> None:
        """Heavily tinted image fails with severe_color_cast."""
        stage = ColorCastFilterStage(max_cast_dist=15.0)
        tinted = np.zeros((100, 100, 3), dtype=np.uint8)
        tinted[:, :, 2] = 255  # Pure saturated blue
        gray = np.full((100, 100), 128, dtype=np.uint8)

        result = stage.process(tinted, gray)
        assert result.passed is False
        assert result.rejection_reason == "severe_color_cast"
