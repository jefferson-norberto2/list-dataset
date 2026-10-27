"""Unit tests for configuration validation."""

from __future__ import annotations

from pathlib import Path

import pytest

from dataset_curator.config import CurateConfig


class TestCurateConfigValidation:
    """Tests verifying CurateConfig parameter validation."""

    def test_default_config_valid(self, tmp_path: Path) -> None:
        """Default parameters initialize successfully."""
        cfg = CurateConfig(input_dir=tmp_path / "in", output_dir=tmp_path / "out")
        assert cfg.target_dim == 512
        assert cfg.crop_size == (512, 512)
        assert cfg.max_clipping_ratio == 0.01
        assert cfg.min_luminance_std == 40.0
        assert cfg.min_entropy == 7.0
        assert cfg.min_laplacian_var == 120.0
        assert cfg.max_niqe == 3.5

    def test_invalid_target_dim(self, tmp_path: Path) -> None:
        """Non-positive target_dim raises ValueError."""
        with pytest.raises(ValueError, match="target_dim must be positive"):
            CurateConfig(input_dir=tmp_path, output_dir=tmp_path, target_dim=0)

    def test_invalid_clipping_ratio(self, tmp_path: Path) -> None:
        """Clipping ratio outside [0, 1] raises ValueError."""
        with pytest.raises(ValueError, match="max_clipping_ratio must be between"):
            CurateConfig(
                input_dir=tmp_path, output_dir=tmp_path, max_clipping_ratio=1.5
            )

    def test_invalid_device(self, tmp_path: Path) -> None:
        """Unsupported device name raises ValueError."""
        with pytest.raises(ValueError, match="device must be one of"):
            CurateConfig(input_dir=tmp_path, output_dir=tmp_path, device="tpu")  # type: ignore[arg-type]

    def test_invalid_png_compression(self, tmp_path: Path) -> None:
        """Compression level outside 0-9 raises ValueError."""
        with pytest.raises(ValueError, match="png_compression must be between 0 and 9"):
            CurateConfig(input_dir=tmp_path, output_dir=tmp_path, png_compression=10)

    def test_negative_thresholds(self, tmp_path: Path) -> None:
        """Negative metric thresholds raise ValueError."""
        with pytest.raises(ValueError, match="min_luminance_std must be non-negative"):
            CurateConfig(
                input_dir=tmp_path, output_dir=tmp_path, min_luminance_std=-1.0
            )
        with pytest.raises(ValueError, match="min_entropy must be non-negative"):
            CurateConfig(input_dir=tmp_path, output_dir=tmp_path, min_entropy=-1.0)
        with pytest.raises(ValueError, match="min_laplacian_var must be non-negative"):
            CurateConfig(
                input_dir=tmp_path, output_dir=tmp_path, min_laplacian_var=-1.0
            )
        with pytest.raises(ValueError, match="max_niqe must be non-negative"):
            CurateConfig(input_dir=tmp_path, output_dir=tmp_path, max_niqe=-1.0)
