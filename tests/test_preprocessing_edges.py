"""Additional edge case tests for image preprocessing."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from dataset.curator import (
    ImageLoadingError,
    downsample_shorter_edge,
    load_image,
)


class TestPreprocessingEdgeCases:
    """Tests for diverse image formats and channel configurations."""

    def test_load_16bit_image(self, tmp_path: Path) -> None:
        """16-bit image is normalized properly to 8-bit uint8."""
        path = tmp_path / "img16.png"
        img16 = np.full((100, 100), 32768, dtype=np.uint16)
        cv2.imwrite(str(path), img16)

        loaded = load_image(path)
        assert loaded.dtype == np.uint8
        assert loaded.shape == (100, 100, 3)
        assert loaded[0, 0, 0] == 128

    def test_load_grayscale_png(self, tmp_path: Path) -> None:
        """Grayscale 2D image is converted to 3-channel RGB."""
        path = tmp_path / "gray.png"
        gray = np.full((100, 100), 75, dtype=np.uint8)
        cv2.imwrite(str(path), gray)

        loaded = load_image(path)
        assert loaded.shape == (100, 100, 3)
        assert np.all(loaded == 75)

    def test_load_bgra_transparent_png(self, tmp_path: Path) -> None:
        """4-channel BGRA image is converted to 3-channel RGB."""
        path = tmp_path / "alpha.png"
        bgra = np.full((100, 100, 4), 200, dtype=np.uint8)
        cv2.imwrite(str(path), bgra)

        loaded = load_image(path)
        assert loaded.shape == (100, 100, 3)

    def test_load_empty_file_error(self, tmp_path: Path) -> None:
        """Empty 0-byte file raises ImageLoadingError."""
        path = tmp_path / "zero.png"
        path.touch()

        with pytest.raises(ImageLoadingError, match="empty"):
            load_image(path)

    def test_load_nonexistent_file(self, tmp_path: Path) -> None:
        """Non-existent path raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            load_image(tmp_path / "nonexistent.png")

    def test_downsample_invalid_dim(self) -> None:
        """target_dim <= 0 raises ValueError."""
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        with pytest.raises(ValueError, match="target_dim must be positive"):
            downsample_shorter_edge(img, target_dim=0)

    def test_downsample_invalid_ndim(self) -> None:
        """4D array raises ValueError."""
        img = np.zeros((1, 100, 100, 3), dtype=np.uint8)
        with pytest.raises(ValueError, match="Expected 2D or 3D"):
            downsample_shorter_edge(img, target_dim=50)  # type: ignore[arg-type]
