"""Configuration dataclass and models for the dataset curation pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal


@dataclass(frozen=True, slots=True)
class CurateConfig:
    """Configuration settings and quality thresholds for reference image dataset curation.

    Attributes:
        input_dir: Directory containing candidate input images.
        output_dir: Directory where curated pristine crops will be saved.
        audit_csv_path: File path for the structured CSV audit log.
        target_dim: Target resolution for the shorter image dimension during downsampling.
        crop_size: Fixed (height, width) dimensions for center cropping.
        max_clipping_ratio: Maximum allowable proportion of under- or over-exposed pixels (1.0% = 0.01).
        min_luminance_std: Minimum standard deviation of grayscale luminance.
        min_entropy: Minimum Shannon entropy in bits/pixel.
        min_laplacian_var: Minimum variance of the Laplacian operator (sharpness).
        max_niqe: Maximum allowable Natural Image Quality Evaluator (NIQE) score.
        device: Device to use for tensor-based metrics ('auto', 'cuda', 'cpu').
        png_compression: PNG compression level (0 is lossless uncompressed, 9 is maximum compression).
        filename_pattern: Format string for pristine output filenames.
        supported_formats: Tuple of supported file extensions (case-insensitive).
        csv_score_path: Optional path to an external CSV file containing image quality scores.
        csv_score_column: Optional column name in the external CSV containing quality scores.
        min_csv_score: Minimum allowable quality score from external CSV (default: 0.7).
    """

    input_dir: Path
    output_dir: Path
    audit_csv_path: Path = field(default_factory=lambda: Path("curation_audit.csv"))
    target_dim: int = 512
    crop_size: tuple[int, int] = (512, 512)
    max_clipping_ratio: float = 0.01
    min_luminance_std: float = 40.0
    min_entropy: float = 7.0
    min_laplacian_var: float = 120.0
    max_niqe: float = 3.5
    device: Literal["auto", "cuda", "cpu"] = "auto"
    png_compression: int = 0
    filename_pattern: str = "{stem}_pristine_{target_dim}.png"
    supported_formats: tuple[str, ...] = (
        ".png",
        ".jpg",
        ".jpeg",
        ".bmp",
        ".tiff",
        ".tif",
    )
    csv_score_path: Path | None = None
    csv_score_column: str | None = None
    min_csv_score: float = 0.7

    @property
    def is_csv_score_enabled(self) -> bool:
        """Indicates whether external CSV score filtering is enabled."""
        return self.csv_score_path is not None and bool(
            self.csv_score_column and self.csv_score_column.strip()
        )

    def __post_init__(self) -> None:
        """Validate configuration settings and thresholds."""
        if self.target_dim <= 0:
            raise ValueError(f"target_dim must be positive, got {self.target_dim}")
        if self.crop_size[0] <= 0 or self.crop_size[1] <= 0:
            raise ValueError(
                f"crop_size dimensions must be positive, got {self.crop_size}"
            )
        if not (0.0 <= self.max_clipping_ratio <= 1.0):
            raise ValueError(
                f"max_clipping_ratio must be between 0.0 and 1.0, got {self.max_clipping_ratio}"
            )
        if self.min_luminance_std < 0.0:
            raise ValueError(
                f"min_luminance_std must be non-negative, got {self.min_luminance_std}"
            )
        if self.min_entropy < 0.0:
            raise ValueError(
                f"min_entropy must be non-negative, got {self.min_entropy}"
            )
        if self.min_laplacian_var < 0.0:
            raise ValueError(
                f"min_laplacian_var must be non-negative, got {self.min_laplacian_var}"
            )
        if self.max_niqe < 0.0:
            raise ValueError(f"max_niqe must be non-negative, got {self.max_niqe}")
        if self.min_csv_score < 0.0:
            raise ValueError(
                f"min_csv_score must be non-negative, got {self.min_csv_score}"
            )
        if not (0 <= self.png_compression <= 9):
            raise ValueError(
                f"png_compression must be between 0 and 9, got {self.png_compression}"
            )
        if self.device not in ("auto", "cuda", "cpu"):
            raise ValueError(
                f"device must be one of 'auto', 'cuda', 'cpu', got {self.device}"
            )
