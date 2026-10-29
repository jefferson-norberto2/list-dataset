"""Unit tests for DistortionGenerator and synthetic degradation methods."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from dataset.distortion import DistortionGenerator


@pytest.fixture
def sample_image_path(tmp_path: Path) -> Path:
    """Creates a temporary sample RGB test image on disk."""
    img_dir = tmp_path / "raw"
    img_dir.mkdir(parents=True, exist_ok=True)
    img_path = img_dir / "sample.png"

    # Create deterministic test image with gradients
    data = np.zeros((100, 100, 3), dtype=np.uint8)
    data[:, :, 0] = np.linspace(0, 255, 100, dtype=np.uint8)
    data[:, :, 1] = 128
    data[:, :, 2] = np.linspace(255, 0, 100, dtype=np.uint8)

    cv2.imwrite(str(img_path), data)
    return img_path


class TestDistortionGenerator:
    """Tests covering DistortionGenerator operations and safeguards."""

    def test_uninitialized_raises_value_error(self) -> None:
        """Calling distortion methods before loading an image must raise ValueError."""
        gen = DistortionGenerator()
        with pytest.raises(ValueError, match="No image loaded"):
            gen.add_gaussian_blur()

        with pytest.raises(ValueError, match="No image loaded"):
            gen.add_poisson_gaussian_noise()

        with pytest.raises(ValueError, match="No image loaded"):
            gen.add_jpeg_compression()

        with pytest.raises(ValueError, match="No image loaded"):
            gen.add_jpeg2000_compression()

        with pytest.raises(ValueError, match="No image loaded"):
            gen.change_contrast()

        with pytest.raises(ValueError, match="No image loaded"):
            gen.add_pink_noise()

        with pytest.raises(ValueError, match="No image loaded"):
            gen.save_output(np.zeros((10, 10, 3), dtype=np.float32), "blur", 1)

    def test_change_image_missing_file_raises_value_error(self, tmp_path: Path) -> None:
        """Loading a non-existent file path must raise ValueError."""
        gen = DistortionGenerator()
        missing = tmp_path / "non_existent.png"
        with pytest.raises(ValueError, match="Image not found at path"):
            gen.change_image(str(missing))

    def test_load_and_image_metadata(self, sample_image_path: Path) -> None:
        """Successful load initializes image properties and dimensions."""
        gen = DistortionGenerator()
        gen.change_image(str(sample_image_path))

        assert gen.img is not None
        assert gen.img_uint8 is not None
        assert gen.height == 100
        assert gen.width == 100
        assert gen.channels == 3
        assert gen.img_name == "sample"
        assert gen.img.dtype == np.float32

    def test_add_gaussian_blur(self, sample_image_path: Path) -> None:
        """Verifies Gaussian blur with default and forced kernel."""
        gen = DistortionGenerator()
        gen.change_image(str(sample_image_path))

        blurred_default = gen.add_gaussian_blur(sigma=2.0)
        assert blurred_default.shape == (100, 100, 3)

        blurred_kernel = gen.add_gaussian_blur(sigma=2.0, force_kernel=5)
        assert blurred_kernel.shape == (100, 100, 3)

        blurred_even_kernel = gen.add_gaussian_blur(sigma=2.0, force_kernel=4)
        assert blurred_even_kernel.shape == (100, 100, 3)

    def test_add_poisson_gaussian_noise(self, sample_image_path: Path) -> None:
        """Verifies Poisson-Gaussian noise output dimensions and alteration."""
        gen = DistortionGenerator()
        gen.change_image(str(sample_image_path))

        noisy = gen.add_poisson_gaussian_noise(shot_noise=0.02, read_noise=0.005)
        assert noisy.shape == (100, 100, 3)
        assert not np.allclose(noisy, gen.img)

    def test_add_jpeg_compression(self, sample_image_path: Path) -> None:
        """Verifies JPEG compression output range and format."""
        gen = DistortionGenerator()
        gen.change_image(str(sample_image_path))

        compressed = gen.add_jpeg_compression(quality=20)
        assert compressed.shape == (100, 100, 3)
        assert compressed.min() >= 0.0
        assert compressed.max() <= 1.0

    def test_add_jpeg2000_compression(self, sample_image_path: Path) -> None:
        """Verifies JPEG 2000 compression and temp file handling."""
        gen = DistortionGenerator()
        gen.change_image(str(sample_image_path))

        compressed = gen.add_jpeg2000_compression(compression_ratio=10)
        assert compressed.shape == (100, 100, 3)
        assert compressed.min() >= 0.0
        assert compressed.max() <= 1.0

    def test_change_contrast(self, sample_image_path: Path) -> None:
        """Verifies contrast decrement reduces variance toward neutral gray."""
        gen = DistortionGenerator()
        gen.change_image(str(sample_image_path))

        low_contrast = gen.change_contrast(alpha=0.3)
        assert low_contrast.shape == (100, 100, 3)
        assert np.std(low_contrast) < np.std(gen.img)

    def test_add_pink_noise(self, sample_image_path: Path) -> None:
        """Verifies 1/f pink noise generation."""
        gen = DistortionGenerator()
        gen.change_image(str(sample_image_path))

        pink = gen.add_pink_noise(intensity=0.1, spatial_scale=1.0)
        assert pink.shape == (100, 100, 3)
        assert not np.allclose(pink, gen.img)

    def test_save_output(self, sample_image_path: Path) -> None:
        """Verifies save_output generates valid PNG file at destination."""
        gen = DistortionGenerator()
        gen.change_image(str(sample_image_path))

        blurred = gen.add_gaussian_blur(sigma=1.5)
        gen.save_output(blurred, "blur", 1)

        expected_file = sample_image_path.parent.parent / "blur" / "dist_1_sample.png"
        assert expected_file.exists()
        saved = cv2.imread(str(expected_file))
        assert saved is not None
        assert saved.shape == (100, 100, 3)
