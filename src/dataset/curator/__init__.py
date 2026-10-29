"""Modular reference dataset curation package."""

from __future__ import annotations

from dataset.curator.audit import AuditEntry, AuditLogger
from dataset.curator.config import CurateConfig
from dataset.curator.curator import DatasetCurator
from dataset.curator.metrics import (
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
from dataset.curator.pipeline import CascadedQualityGate, CurationAssessment
from dataset.curator.preprocessing import (
    ImageLoadingError,
    ImageResolutionError,
    center_crop,
    downsample_shorter_edge,
    find_supported_images,
    load_image,
    preprocess_image,
    save_image_lossless_png,
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
    "compute_clipping_ratio",
    "compute_laplacian_variance",
    "compute_luminance_std",
    "compute_shannon_entropy",
    "downsample_shorter_edge",
    "evaluate_csv_score",
    "find_supported_images",
    "load_csv_scores",
    "load_image",
    "preprocess_image",
    "save_image_lossless_png",
]
