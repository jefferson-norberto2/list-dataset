"""Dynamic range and contrast evaluation filter stage."""

from __future__ import annotations

import numpy as np

from dataset.curator.metrics.base import FilterResult, QualityFilterStage


def compute_luminance_std(gray: np.ndarray) -> float:
    """Calculates the standard deviation of grayscale luminance.

    Args:
        gray: 2D Grayscale luminance image uint8.

    Returns:
        float: Standard deviation of pixel intensities.
    """
    if gray.size == 0:
        return 0.0
    return float(np.std(gray, dtype=np.float64))


class ContrastFilterStage(QualityFilterStage):
    """Quality gate stage evaluating image dynamic range via luminance standard deviation."""

    name: str = "contrast"

    def __init__(self, min_luminance_std: float = 40.0) -> None:
        """Initializes the contrast filter stage.

        Args:
            min_luminance_std: Minimum acceptable standard deviation of grayscale luminance.
        """
        self.min_luminance_std = min_luminance_std

    def compute(self, image_rgb: np.ndarray, gray: np.ndarray) -> float:
        """Computes the luminance standard deviation."""
        return compute_luminance_std(gray)

    def evaluate(self, metric_value: float) -> tuple[bool, str | None]:
        """Checks if luminance standard deviation meets the minimum contrast threshold."""
        if metric_value < self.min_luminance_std:
            return False, "low_contrast"
        return True, None

    def process(self, image_rgb: np.ndarray, gray: np.ndarray) -> FilterResult:
        """Executes the contrast check."""
        val = self.compute(image_rgb, gray)
        passed, reason = self.evaluate(val)
        return FilterResult(
            stage_name=self.name,
            metric_value=val,
            passed=passed,
            rejection_reason=reason,
        )
