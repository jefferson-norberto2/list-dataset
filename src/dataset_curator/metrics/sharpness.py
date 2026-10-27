"""Focus sharpness evaluation filter stage using Laplacian variance."""

from __future__ import annotations

import cv2
import numpy as np

from dataset_curator.metrics.base import FilterResult, QualityFilterStage


def compute_laplacian_variance(gray: np.ndarray, ksize: int = 1) -> float:
    """Calculates the focus sharpness using the variance of the Laplacian operator.

    Based on the focus measurement algorithm proposed by Pech-Pacheco et al. (2000).
    Higher values indicate sharp edges and high focus; lower values indicate blur.

    Args:
        gray: 2D Grayscale image uint8 of shape (H, W).
        ksize: Aperture size used to compute the second-derivative filters (default: 1).

    Returns:
        float: Variance of the Laplacian response.
    """
    if gray.size == 0:
        return 0.0

    laplacian = cv2.Laplacian(gray, ddepth=cv2.CV_64F, ksize=ksize)
    return float(laplacian.var())


class SharpnessFilterStage(QualityFilterStage):
    """Quality gate stage evaluating image focus sharpness via Laplacian variance."""

    name: str = "sharpness"

    def __init__(self, min_laplacian_var: float = 120.0, ksize: int = 1) -> None:
        """Initializes the sharpness filter stage.

        Args:
            min_laplacian_var: Minimum acceptable variance of the Laplacian operator (default: 120.0).
            ksize: Kernel aperture size for the Laplacian filter (default: 1).
        """
        self.min_laplacian_var = min_laplacian_var
        self.ksize = ksize

    def compute(self, image_rgb: np.ndarray, gray: np.ndarray) -> float:
        """Computes the Laplacian variance of the grayscale image."""
        return compute_laplacian_variance(gray, ksize=self.ksize)

    def evaluate(self, metric_value: float) -> tuple[bool, str | None]:
        """Checks if Laplacian variance meets the minimum sharpness threshold."""
        if metric_value < self.min_laplacian_var:
            return False, "blurry"
        return True, None

    def process(self, image_rgb: np.ndarray, gray: np.ndarray) -> FilterResult:
        """Executes the sharpness check."""
        val = self.compute(image_rgb, gray)
        passed, reason = self.evaluate(val)
        return FilterResult(
            stage_name=self.name,
            metric_value=val,
            passed=passed,
            rejection_reason=reason,
        )
