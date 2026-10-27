"""Information density (Shannon entropy) evaluation filter stage."""

from __future__ import annotations

import numpy as np

from dataset_curator.metrics.base import FilterResult, QualityFilterStage


def compute_shannon_entropy(gray: np.ndarray) -> float:
    """Calculates the Shannon entropy of an 8-bit grayscale image in bits/pixel.

    Args:
        gray: 2D Grayscale image uint8 of shape (H, W).

    Returns:
        float: Shannon entropy in range [0.0, 8.0] bits/pixel.
    """
    if gray.size == 0:
        return 0.0

    counts = np.bincount(gray.ravel(), minlength=256)
    total_pixels = gray.size
    probabilities = counts[counts > 0] / total_pixels

    # Shannon entropy: - sum(p * log2(p))
    entropy = -np.sum(probabilities * np.log2(probabilities))
    return float(entropy)


class EntropyFilterStage(QualityFilterStage):
    """Quality gate stage evaluating image information density via Shannon entropy."""

    name: str = "entropy"

    def __init__(self, min_entropy: float = 7.0) -> None:
        """Initializes the entropy filter stage.

        Args:
            min_entropy: Minimum acceptable Shannon entropy in bits/pixel (default: 7.0).
        """
        self.min_entropy = min_entropy

    def compute(self, image_rgb: np.ndarray, gray: np.ndarray) -> float:
        """Computes the Shannon entropy of the grayscale image."""
        return compute_shannon_entropy(gray)

    def evaluate(self, metric_value: float) -> tuple[bool, str | None]:
        """Checks if Shannon entropy meets the minimum threshold."""
        if metric_value < self.min_entropy:
            return False, "low_entropy"
        return True, None

    def process(self, image_rgb: np.ndarray, gray: np.ndarray) -> FilterResult:
        """Executes the entropy check."""
        val = self.compute(image_rgb, gray)
        passed, reason = self.evaluate(val)
        return FilterResult(
            stage_name=self.name,
            metric_value=val,
            passed=passed,
            rejection_reason=reason,
        )
