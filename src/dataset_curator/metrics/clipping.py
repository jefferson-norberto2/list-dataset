"""Clipping and exposure evaluation filter stage."""

from __future__ import annotations

import numpy as np

from dataset_curator.metrics.base import FilterResult, QualityFilterStage


def compute_clipping_ratio(
    image: np.ndarray,
    low_thresh: int = 1,
    high_thresh: int = 254,
) -> float:
    """Calculates the proportion of under-exposed and over-exposed pixels.

    For 2D grayscale arrays, computes the ratio of pixels where value <= low_thresh
    or value >= high_thresh. For 3D arrays, checks clipping across channels.

    Args:
        image: 2D or 3D uint8 image array.
        low_thresh: Maximum pixel value considered under-exposed (default: 1).
        high_thresh: Minimum pixel value considered over-exposed (default: 254).

    Returns:
        float: Fraction of clipped pixels in range [0.0, 1.0].
    """
    if image.size == 0:
        return 0.0

    if image.ndim == 2:
        clipped = (image <= low_thresh) | (image >= high_thresh)
        return float(np.count_nonzero(clipped) / clipped.size)
    elif image.ndim == 3:
        # A pixel is considered clipped if any color channel saturates
        channel_clipped = (image <= low_thresh) | (image >= high_thresh)
        pixel_clipped = np.any(channel_clipped, axis=-1)
        return float(np.count_nonzero(pixel_clipped) / pixel_clipped.size)
    else:
        raise ValueError(f"Expected 2D or 3D array, got ndim={image.ndim}")


class ClippingFilterStage(QualityFilterStage):
    """Quality gate stage evaluating under/over-exposure pixel clipping."""

    name: str = "clipping"

    def __init__(
        self,
        max_clipping_ratio: float = 0.01,
        low_thresh: int = 1,
        high_thresh: int = 254,
        use_grayscale: bool = True,
    ) -> None:
        """Initializes the clipping filter stage.

        Args:
            max_clipping_ratio: Maximum allowable proportion of clipped pixels (default 0.01 = 1.0%).
            low_thresh: Threshold for under-exposure (default 1).
            high_thresh: Threshold for over-exposure (default 254).
            use_grayscale: If True, evaluates on luminance; otherwise evaluates on 3-channel RGB.
        """
        self.max_clipping_ratio = max_clipping_ratio
        self.low_thresh = low_thresh
        self.high_thresh = high_thresh
        self.use_grayscale = use_grayscale

    def compute(self, image_rgb: np.ndarray, gray: np.ndarray) -> float:
        """Computes the clipping ratio on grayscale luminance or RGB."""
        target = gray if self.use_grayscale else image_rgb
        return compute_clipping_ratio(
            target, low_thresh=self.low_thresh, high_thresh=self.high_thresh
        )

    def evaluate(self, metric_value: float) -> tuple[bool, str | None]:
        """Checks if the clipping ratio is within acceptable limits."""
        if metric_value > self.max_clipping_ratio:
            return False, "clipping_exceeded"
        return True, None

    def process(self, image_rgb: np.ndarray, gray: np.ndarray) -> FilterResult:
        """Executes the clipping check."""
        val = self.compute(image_rgb, gray)
        passed, reason = self.evaluate(val)
        return FilterResult(
            stage_name=self.name,
            metric_value=val,
            passed=passed,
            rejection_reason=reason,
        )
