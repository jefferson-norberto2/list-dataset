"""Cascaded quality gate pipeline with early-exit strategy."""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass, field

import cv2
import numpy as np

from dataset.curator.config import CurateConfig
from dataset.curator.metrics.base import FilterResult, QualityFilterStage
from dataset.curator.metrics.clipping import ClippingFilterStage
from dataset.curator.metrics.contrast import ContrastFilterStage
from dataset.curator.metrics.csv_score import CSVScoreFilterStage, load_csv_scores
from dataset.curator.metrics.entropy import EntropyFilterStage
from dataset.curator.metrics.niqe import NIQEFilterStage
from dataset.curator.metrics.sharpness import SharpnessFilterStage

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class CurationAssessment:
    """Aggregated assessment result from the cascaded quality gate.

    Attributes:
        is_pristine: Whether the image passed all cascaded quality filters.
        rejection_reason: Reason for rejection, or 'accepted' if all filters passed.
        csv_score: Quality score from external CSV (None if skipped or not enabled).
        clipping_ratio: Proportion of clipped pixels (None if skipped or failed earlier).
        luminance_std: Grayscale luminance standard deviation (None if skipped).
        entropy: Shannon entropy in bits/pixel (None if skipped).
        laplacian_var: Variance of Laplacian operator (None if skipped).
        niqe_score: NIQE score from pyiqa (None if skipped).
        stage_results: Mapping of stage names to their individual FilterResult.
    """

    is_pristine: bool
    rejection_reason: str
    csv_score: float | None = None
    clipping_ratio: float | None = None
    luminance_std: float | None = None
    entropy: float | None = None
    laplacian_var: float | None = None
    niqe_score: float | None = None
    stage_results: dict[str, FilterResult] = field(default_factory=dict)


class CascadedQualityGate:
    """Cascaded quality gate executing quality filters ordered by computational cost.

    Executes filters sequentially:
      1. Clipping / Exposure Check (Lowest cost)
      2. Dynamic Range / Contrast Check (Low cost)
      3. Information Density / Shannon Entropy (Low-Medium cost)
      4. Focus Sharpness / Laplacian Variance (Medium cost)
      5. No-Reference IQA / NIQE (High cost)

    When configured with external CSV quality scores, an initial CSV score filter
    stage is prepended before image processing filters ($O(1)$ lookup cost).

    Implements an early-exit strategy: immediately stops and reports failure upon
    the first failing filter, skipping subsequent expensive calculations.
    """

    def __init__(
        self,
        stages: Sequence[QualityFilterStage] | None = None,
        config: CurateConfig | None = None,
    ) -> None:
        """Initializes the cascaded quality gate.

        Args:
            stages: Explicit list of filter stages. If None, builds default pipeline from config.
            config: Configuration settings. Used when stages is None.
        """
        if stages is not None:
            self.stages = list(stages)
        else:
            cfg = config or CurateConfig(input_dir=None, output_dir=None)  # type: ignore[arg-type]
            configured_stages: list[QualityFilterStage] = []
            if cfg.is_csv_score_enabled:
                assert cfg.csv_score_path is not None
                assert cfg.csv_score_column is not None
                scores = load_csv_scores(
                    csv_path=cfg.csv_score_path,
                    score_column=cfg.csv_score_column,
                )
                configured_stages.append(
                    CSVScoreFilterStage(min_score=cfg.min_csv_score, scores=scores)
                )

            configured_stages.extend(
                [
                    ClippingFilterStage(max_clipping_ratio=cfg.max_clipping_ratio),
                    ContrastFilterStage(min_luminance_std=cfg.min_luminance_std),
                    EntropyFilterStage(min_entropy=cfg.min_entropy),
                    SharpnessFilterStage(min_laplacian_var=cfg.min_laplacian_var),
                    NIQEFilterStage(max_niqe=cfg.max_niqe, device=cfg.device),
                ]
            )
            self.stages = configured_stages

    def evaluate(
        self,
        image_rgb: np.ndarray,
        filename: str | None = None,
        csv_score: float | None = None,
    ) -> CurationAssessment:
        """Evaluates an image against the cascade of quality filters.

        Args:
            image_rgb: 3-channel RGB image uint8 array of shape (H, W, 3).
            filename: Optional image filename for contextual stages (e.g. CSV score).
            csv_score: Optional pre-loaded CSV score for the candidate image.

        Returns:
            CurationAssessment: Assessment record detailing pass/fail status and measured metrics.
        """
        for stage in self.stages:
            if hasattr(stage, "set_current_image"):
                stage.set_current_image(filename=filename, score=csv_score)

        # Precompute grayscale luminance once to share across grayscale-based stages
        gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)

        results: dict[str, FilterResult] = {}
        metric_values: dict[str, float | None] = {
            "csv_score": csv_score,
            "clipping": None,
            "contrast": None,
            "entropy": None,
            "sharpness": None,
            "niqe": None,
        }

        for stage in self.stages:
            result = stage.process(image_rgb, gray)
            results[stage.name] = result
            metric_values[stage.name] = result.metric_value

            if not result.passed:
                reason = result.rejection_reason or f"failed_{stage.name}"
                logger.debug(
                    "Image rejected at stage '%s' (value=%.4f, reason=%s)",
                    stage.name,
                    result.metric_value,
                    reason,
                )
                return CurationAssessment(
                    is_pristine=False,
                    rejection_reason=reason,
                    csv_score=metric_values.get("csv_score"),
                    clipping_ratio=metric_values.get("clipping"),
                    luminance_std=metric_values.get("contrast"),
                    entropy=metric_values.get("entropy"),
                    laplacian_var=metric_values.get("sharpness"),
                    niqe_score=metric_values.get("niqe"),
                    stage_results=results,
                )

        # All stages passed
        return CurationAssessment(
            is_pristine=True,
            rejection_reason="accepted",
            csv_score=metric_values.get("csv_score"),
            clipping_ratio=metric_values.get("clipping"),
            luminance_std=metric_values.get("contrast"),
            entropy=metric_values.get("entropy"),
            laplacian_var=metric_values.get("sharpness"),
            niqe_score=metric_values.get("niqe"),
            stage_results=results,
        )
