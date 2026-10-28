"""NIQE (Natural Image Quality Evaluator) filter stage using pyiqa."""

from __future__ import annotations

import logging
import math
from typing import Literal

import numpy as np

from dataset.curator.metrics.base import FilterResult, QualityFilterStage

logger = logging.getLogger(__name__)


class NIQEFilterStage(QualityFilterStage):
    """Quality gate stage evaluating Natural Image Quality Evaluator (NIQE) score.

    Uses pyiqa with automatic CUDA device detection and graceful CPU fallback.
    """

    name: str = "niqe"

    def __init__(
        self,
        max_niqe: float = 3.5,
        device: Literal["auto", "cuda", "cpu"] = "auto",
    ) -> None:
        """Initializes the NIQE filter stage.

        Args:
            max_niqe: Maximum allowable NIQE score (lower indicates higher quality). Default: 3.5.
            device: Device preference for tensor calculations ('auto', 'cuda', 'cpu').
        """
        self.max_niqe = max_niqe
        self.device_pref = device
        self._device: str | None = None
        self._metric = None

    def _get_target_device(self) -> str:
        """Determines the target computation device with automatic CUDA detection and fallback."""
        if self._device is not None:
            return self._device

        import torch

        if self.device_pref == "cuda":
            if torch.cuda.is_available():
                self._device = "cuda"
            else:
                logger.warning(
                    "CUDA requested for NIQE but unavailable. Falling back to CPU."
                )
                self._device = "cpu"
        elif self.device_pref == "cpu":
            self._device = "cpu"
        else:  # "auto"
            self._device = "cuda" if torch.cuda.is_available() else "cpu"

        logger.debug("NIQE metric configured on device: %s", self._device)
        return self._device

    def _ensure_metric(self):
        """Lazily instantiates the PyIQA NIQE metric on the target device."""
        if self._metric is None:
            import pyiqa
            import torch

            target_device = self._get_target_device()
            try:
                self._metric = pyiqa.create_metric(
                    "niqe", device=torch.device(target_device)
                )
            except Exception as exc:
                if target_device == "cuda":
                    logger.warning(
                        "Failed to initialize NIQE on CUDA (%s). Retrying on CPU.", exc
                    )
                    self._device = "cpu"
                    self._metric = pyiqa.create_metric(
                        "niqe", device=torch.device("cpu")
                    )
                else:
                    raise
        return self._metric

    def compute(self, image_rgb: np.ndarray, gray: np.ndarray) -> float:
        """Computes the NIQE score using pyiqa.

        Args:
            image_rgb: RGB image uint8 array of shape (H, W, 3).
            gray: Grayscale image uint8 array of shape (H, W).

        Returns:
            float: NIQE score (lower is better, or NaN if calculation failed).
        """
        import torch

        try:
            metric = self._ensure_metric()
            target_device = self._get_target_device()
            # Convert RGB uint8 to normalized float tensor (1, 3, H, W) in [0.0, 1.0]
            tensor = (
                torch.from_numpy(image_rgb)
                .permute(2, 0, 1)
                .unsqueeze(0)
                .float()
                .div(255.0)
                .to(target_device)
            )

            with torch.no_grad():
                score = metric(tensor)
                if isinstance(score, torch.Tensor):
                    val = float(score.detach().cpu().item())
                else:
                    val = float(score)
                return val
        except Exception as exc:  # noqa: BLE001 - return NaN if PyIQA metric evaluation fails
            logger.error("Error computing NIQE metric: %s", exc)
            return float("nan")

    def evaluate(self, metric_value: float) -> tuple[bool, str | None]:
        """Checks if NIQE score is non-NaN and does not exceed max_niqe threshold."""
        if math.isnan(metric_value):
            return False, "niqe_nan"
        if metric_value > self.max_niqe:
            return False, "high_niqe"
        return True, None

    def process(self, image_rgb: np.ndarray, gray: np.ndarray) -> FilterResult:
        """Executes the NIQE check."""
        val = self.compute(image_rgb, gray)
        passed, reason = self.evaluate(val)
        return FilterResult(
            stage_name=self.name,
            metric_value=val,
            passed=passed,
            rejection_reason=reason,
        )
