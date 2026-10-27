"""Modular reference dataset curation package."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from dataset_curator.audit import AuditEntry, AuditLogger
from dataset_curator.config import CurateConfig
from dataset_curator.curator import DatasetCurator
from dataset_curator.metrics import (
    ClippingFilterStage,
    ContrastFilterStage,
    CSVScoreFilterStage,
    EntropyFilterStage,
    FilterResult,
    NIQEFilterStage,
    QualityFilterStage,
    SharpnessFilterStage,
    compute_clipping_ratio,
    compute_laplacian_variance,
    compute_luminance_std,
    compute_shannon_entropy,
    evaluate_csv_score,
    load_csv_scores,
)
from dataset_curator.pipeline import CascadedQualityGate, CurationAssessment
from dataset_curator.preprocessing import (
    ImageLoadingError,
    ImageResolutionError,
    center_crop,
    downsample_shorter_edge,
    find_supported_images,
    load_image,
    preprocess_image,
    save_image_lossless_png,
)

if TYPE_CHECKING:
    from dataset_curator.similarity import (
        SimilarityResult,
        check_image_similarity,
        compare_images_ssim,
        compute_ssim,
    )

__version__ = "0.1.0"

__all__ = [
    "AuditEntry",
    "AuditLogger",
    "CSVScoreFilterStage",
    "CascadedQualityGate",
    "ClippingFilterStage",
    "ContrastFilterStage",
    "CurateConfig",
    "CurationAssessment",
    "DatasetCurator",
    "EntropyFilterStage",
    "FilterResult",
    "ImageLoadingError",
    "ImageResolutionError",
    "NIQEFilterStage",
    "QualityFilterStage",
    "SharpnessFilterStage",
    "SimilarityResult",
    "center_crop",
    "check_image_similarity",
    "compare_images_ssim",
    "compute_clipping_ratio",
    "compute_laplacian_variance",
    "compute_luminance_std",
    "compute_shannon_entropy",
    "compute_ssim",
    "downsample_shorter_edge",
    "evaluate_csv_score",
    "find_supported_images",
    "load_csv_scores",
    "load_image",
    "preprocess_image",
    "save_image_lossless_png",
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
