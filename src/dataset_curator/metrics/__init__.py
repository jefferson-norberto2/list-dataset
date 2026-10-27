"""Quality metrics package for dataset curation."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from dataset_curator.metrics.base import FilterResult, QualityFilterStage
from dataset_curator.metrics.clipping import ClippingFilterStage, compute_clipping_ratio
from dataset_curator.metrics.contrast import ContrastFilterStage, compute_luminance_std
from dataset_curator.metrics.csv_score import (
    CSVScoreFilterStage,
    evaluate_csv_score,
    load_csv_scores,
)
from dataset_curator.metrics.entropy import EntropyFilterStage, compute_shannon_entropy
from dataset_curator.metrics.niqe import NIQEFilterStage
from dataset_curator.metrics.sharpness import (
    SharpnessFilterStage,
    compute_laplacian_variance,
)

if TYPE_CHECKING:
    from dataset_curator.similarity import (
        SimilarityResult,
        check_image_similarity,
        compare_images_ssim,
        compute_ssim,
    )

__all__ = [
    "CSVScoreFilterStage",
    "ClippingFilterStage",
    "ContrastFilterStage",
    "EntropyFilterStage",
    "FilterResult",
    "NIQEFilterStage",
    "QualityFilterStage",
    "SharpnessFilterStage",
    "SimilarityResult",
    "check_image_similarity",
    "compare_images_ssim",
    "compute_clipping_ratio",
    "compute_laplacian_variance",
    "compute_luminance_std",
    "compute_shannon_entropy",
    "compute_ssim",
    "evaluate_csv_score",
    "load_csv_scores",
]


def __getattr__(name: str) -> Any:
    if name in {
        "compute_ssim",
        "compare_images_ssim",
        "check_image_similarity",
        "SimilarityResult",
    }:
        import dataset_curator.similarity as sim

        return getattr(sim, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
