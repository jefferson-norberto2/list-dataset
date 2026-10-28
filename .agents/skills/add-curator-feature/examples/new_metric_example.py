"""Example reference implementation of a new quality gate filter stage.

This example demonstrates how to implement a Color Cast / White Balance filter stage
following the dataset-curator protocol and design patterns.
"""

from __future__ import annotations

import cv2
import numpy as np

from dataset.curator.metrics import FilterResult, QualityFilterStage


def compute_color_cast_ratio(image_rgb: np.ndarray) -> float:
    """Calculates color cast divergence using the Gray-World assumption in CIELAB color space.

    Converts RGB to CIE L*a*b* and computes the Euclidean distance of average (a*, b*)
    chrominance from neutral gray (0, 0).

    Args:
        image_rgb: 3-channel RGB image uint8 array of shape (H, W, 3).

    Returns:
        float: Distance from neutral gray (higher values indicate severe color cast).
    """
    if image_rgb.size == 0:
        return 0.0

    lab = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2LAB)
    # L in [0, 255], a in [0, 255] (centered at 128), b in [0, 255] (centered at 128)
    a_mean = float(np.mean(lab[:, :, 1])) - 128.0
    b_mean = float(np.mean(lab[:, :, 2])) - 128.0
    cast_distance = float(np.sqrt(a_mean**2 + b_mean**2))
    return cast_distance


class ColorCastFilterStage(QualityFilterStage):
    """Quality gate stage evaluating unnatural tint / color cast."""

    name: str = "color_cast"

    def __init__(self, max_cast_dist: float = 15.0) -> None:
        """Initializes the color cast stage.

        Args:
            max_cast_dist: Maximum allowable divergence from neutral gray in LAB space.
        """
        self.max_cast_dist = max_cast_dist

    def compute(self, image_rgb: np.ndarray, gray: np.ndarray) -> float:
        """Computes color cast distance on the RGB array."""
        return compute_color_cast_ratio(image_rgb)

    def evaluate(self, metric_value: float) -> tuple[bool, str | None]:
        """Evaluates if color cast exceeds acceptable threshold."""
        if metric_value > self.max_cast_dist:
            return False, "severe_color_cast"
        return True, None

    def process(self, image_rgb: np.ndarray, gray: np.ndarray) -> FilterResult:
        """Executes computation and quality decision."""
        val = self.compute(image_rgb, gray)
        passed, reason = self.evaluate(val)
        return FilterResult(
            stage_name=self.name,
            metric_value=val,
            passed=passed,
            rejection_reason=reason,
        )
