"""Unit tests for image preprocessing, downsampling, cropping, and lossless I/O."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

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


class TestDownsampleAndCrop:
    """Tests for proportional downsampling and center crop extraction."""

    def test_landscape_downsampling(self) -> None:
        """Landscape image (H=600, W=800) shorter edge H should become exactly 512."""
        img = np.zeros((600, 800, 3), dtype=np.uint8)
        resized = downsample_shorter_edge(img, target_dim=512)

        h, w = resized.shape[:2]
        assert h == 512
        # Aspect ratio: 800 * (512 / 600) = 682.667 -> 683
        assert w == 683

    def test_portrait_downsampling(self) -> None:
        """Portrait image (H=1200, W=900) shorter edge W should become exactly 512."""
        img = np.zeros((1200, 900, 3), dtype=np.uint8)
        resized = downsample_shorter_edge(img, target_dim=512)

        h, w = resized.shape[:2]
        assert w == 512
        # Aspect ratio: 1200 * (512 / 900) = 682.667 -> 683
        assert h == 683

    def test_square_downsampling(self) -> None:
        """Square image (1024x1024) becomes exactly 512x512."""
        img = np.zeros((1024, 1024, 3), dtype=np.uint8)
        resized = downsample_shorter_edge(img, target_dim=512)
        assert resized.shape == (512, 512, 3)

    def test_rejection_of_undersized_image(self) -> None:
        """Images with shorter edge smaller than target_dim raise ImageResolutionError."""
        small = np.zeros((256, 400, 3), dtype=np.uint8)
        with pytest.raises(
            ImageResolutionError, match="smaller than target resolution"
        ):
            downsample_shorter_edge(small, target_dim=512)

    def test_center_crop_dimensions(self) -> None:
        """Center crop extracts exact requested dimensions from the spatial center."""
        img = np.zeros((600, 800, 3), dtype=np.uint8)
        # Mark center pixel
        img[300, 400, :] = 255

        crop = center_crop(img, crop_size=(512, 512))
        assert crop.shape == (512, 512, 3)
        # Center of 600x800 crop starts at y=(600-512)//2 = 44, x=(800-512)//2 = 144
        # Original (300, 400) maps to (300-44, 400-144) = (256, 256) inside crop
        assert np.array_equal(crop[256, 256, :], [255, 255, 255])

    def test_full_preprocessing_pipeline(self) -> None:
        """preprocess_image performs both Lanczos4 resize and center crop."""
        img = np.zeros((1080, 1920, 3), dtype=np.uint8)
        result = preprocess_image(img, target_dim=512, crop_size=(512, 512))
        assert result.shape == (512, 512, 3)


class TestLosslessIO:
    """Tests for lossless image encoding, file discovery, and decoding integrity."""

    def test_lossless_png_roundtrip(self, tmp_path: Path) -> None:
        """Lossless PNG save and reload must preserve every single pixel value exactly."""
        rng = np.random.RandomState(42)
        original = rng.randint(0, 256, size=(512, 512, 3), dtype=np.uint8)

        dest = tmp_path / "test_lossless.png"
        saved = save_image_lossless_png(original, dest, compression_level=0)
        assert saved.is_file()

        reloaded = load_image(saved)
        assert np.array_equal(original, reloaded)

    def test_corrupted_image_handling(self, tmp_path: Path) -> None:
        """Corrupted or invalid image files must raise ImageLoadingError gracefully."""
        corrupt_file = tmp_path / "broken.jpg"
        corrupt_file.write_bytes(b"not a valid jpeg header data")

        with pytest.raises(ImageLoadingError):
            load_image(corrupt_file)

    def test_find_supported_images(self, tmp_path: Path) -> None:
        """find_supported_images discovers supported image extensions recursively."""
        (tmp_path / "img1.png").touch()
        (tmp_path / "img2.JPG").touch()
        (tmp_path / "img3.tiff").touch()
        (tmp_path / "sub").mkdir()
        (tmp_path / "sub" / "img4.bmp").touch()
        (tmp_path / "ignore.txt").touch()

        found = find_supported_images(tmp_path)
        found_names = [p.name.lower() for p in found]
        assert len(found) == 4
        assert "img1.png" in found_names
        assert "img2.jpg" in found_names
        assert "img3.tiff" in found_names
        assert "img4.bmp" in found_names
        assert "ignore.txt" not in found_names
