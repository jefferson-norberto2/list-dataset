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
        assert cfg.csv_score_path is None
        assert cfg.csv_score_column is None
        assert cfg.min_csv_score == 0.7
        assert cfg.is_csv_score_enabled is False

    def test_csv_score_enabled_logic(self, tmp_path: Path) -> None:
        """Verifies is_csv_score_enabled property logic."""
        # Neither provided
        cfg0 = CurateConfig(input_dir=tmp_path, output_dir=tmp_path)
        assert cfg0.is_csv_score_enabled is False
        assert cfg0.min_csv_score == 0.7

        # Only path provided
        cfg1 = CurateConfig(
            input_dir=tmp_path,
            output_dir=tmp_path,
            csv_score_path=tmp_path / "scores.csv",
        )
        assert cfg1.is_csv_score_enabled is False

        # Only column provided
        cfg2 = CurateConfig(
            input_dir=tmp_path,
            output_dir=tmp_path,
            csv_score_column="score",
        )
        assert cfg2.is_csv_score_enabled is False

        # Path provided but empty or whitespace column
        cfg_empty = CurateConfig(
            input_dir=tmp_path,
            output_dir=tmp_path,
            csv_score_path=tmp_path / "scores.csv",
            csv_score_column="   ",
        )
        assert cfg_empty.is_csv_score_enabled is False

        # Both provided
        cfg3 = CurateConfig(
            input_dir=tmp_path,
            output_dir=tmp_path,
            csv_score_path=tmp_path / "scores.csv",
            csv_score_column="score",
        )
        assert cfg3.is_csv_score_enabled is True

    def test_invalid_crop_size(self, tmp_path: Path) -> None:
        """Non-positive crop dimensions raise ValueError."""
        with pytest.raises(ValueError, match="crop_size dimensions must be positive"):
            CurateConfig(input_dir=tmp_path, output_dir=tmp_path, crop_size=(0, 512))
        with pytest.raises(ValueError, match="crop_size dimensions must be positive"):
            CurateConfig(input_dir=tmp_path, output_dir=tmp_path, crop_size=(512, -1))

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
        with pytest.raises(ValueError, match="min_csv_score must be non-negative"):
            CurateConfig(input_dir=tmp_path, output_dir=tmp_path, min_csv_score=-0.1)
