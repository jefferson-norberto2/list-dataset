"""Structural Similarity Index (SSIM) metric functions and re-exports."""

from __future__ import annotations

from dataset_curator.similarity import (
    SimilarityResult,
    check_image_similarity,
    compare_images_ssim,
    compute_ssim,
)

__all__ = [
    "SimilarityResult",
    "check_image_similarity",
    "compare_images_ssim",
    "compute_ssim",
]
