"""Base interfaces and result containers for quality gate filter stages."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np


@dataclass(frozen=True, slots=True)
class FilterResult:
    """Outcome of a single filter stage evaluation.

    Attributes:
        stage_name: Unique identifier for the filter stage.
        metric_value: Measured numerical value (or NaN if computation failed).
        passed: Whether the image passed this quality check.
        rejection_reason: Human-readable rejection reason if failed, else None.
    """

    stage_name: str
    metric_value: float
    passed: bool
    rejection_reason: str | None = None


@runtime_checkable
class QualityFilterStage(Protocol):
    """Protocol for cascaded quality gate filter stages."""

    @property
    def name(self) -> str:
        """Name of the filter stage."""
        ...

    def compute(self, image_rgb: np.ndarray, gray: np.ndarray) -> float:
        """Calculates the scalar quality metric for the given image.

        Args:
            image_rgb: 3-channel RGB image uint8 of shape (H, W, 3).
            gray: 1-channel Grayscale luminance uint8 of shape (H, W).

        Returns:
            float: Computed metric value.
        """
        ...

    def evaluate(self, metric_value: float) -> tuple[bool, str | None]:
        """Evaluates whether the metric value passes the quality threshold.

        Args:
            metric_value: The computed metric value.

        Returns:
            tuple[bool, str | None]: (True, None) if accepted, or (False, rejection_reason) if rejected.
        """
        ...

    def process(self, image_rgb: np.ndarray, gray: np.ndarray) -> FilterResult:
        """Executes computation and threshold evaluation for the stage.

        Args:
            image_rgb: RGB image.
            gray: Grayscale image.

        Returns:
            FilterResult: Result containing the metric value and pass/fail status.
        """
        ...
