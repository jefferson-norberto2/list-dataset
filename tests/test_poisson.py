"""Unit tests for PoissonDistortion and sequential noise application."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from dataset.distortion.algorithms.poisson import PoissonDistortion


@pytest.fixture
def sample_rgb_image() -> np.ndarray:
    """Creates a deterministic 64x64 RGB test image in [0, 255]."""
    rng = np.random.default_rng(42)
    return rng.integers(0, 256, size=(64, 64, 3), dtype=np.uint8)


@pytest.fixture
def sample_gray_image() -> np.ndarray:
    """Creates a deterministic 64x64 Grayscale test image in [0, 255]."""
    rng = np.random.default_rng(42)
    return rng.integers(0, 256, size=(64, 64), dtype=np.uint8)


@pytest.fixture
def sample_image_file(tmp_path: Path, sample_rgb_image: np.ndarray) -> Path:
    """Writes a sample RGB image to a temporary file and returns its path."""
    file_path = tmp_path / "test_input.png"
    cv2.imwrite(str(file_path), cv2.cvtColor(sample_rgb_image, cv2.COLOR_RGB2BGR))
    return file_path


class TestPoissonDistortionInitialization:
    """Tests for PoissonDistortion initialization, image loading, and validation."""

    def test_init_empty(self) -> None:
        """Uninitialized engine has None image attributes and raises if checked."""
        dist = PoissonDistortion()
        assert dist.img is None
        assert dist.current_sequence_step == 0
        with pytest.raises(ValueError, match="No image loaded"):
            dist.apply()

    def test_init_with_numpy_array(self, sample_rgb_image: np.ndarray) -> None:
        """Initializing directly with an array normalizes it to float32 [0, 1]."""
        dist = PoissonDistortion(image=sample_rgb_image)
        assert dist.img is not None
        assert dist.img.dtype == np.float32
        assert dist.height == 64
        assert dist.width == 64
        assert dist.channels == 3

    def test_init_with_file_path(self, sample_image_file: Path) -> None:
        """Initializing with a Path loads and converts the image properly."""
        dist = PoissonDistortion(image=sample_image_file)
        assert dist.img is not None
        assert dist.img_name == "test_input"
        assert dist.channels == 3

    def test_init_with_invalid_type(self) -> None:
        """Passing an unsupported type raises TypeError."""
        with pytest.raises(TypeError, match="Unsupported image type"):
            PoissonDistortion(image=12345)  # type: ignore[arg-type]

    def test_change_image_missing_file_raises(self, tmp_path: Path) -> None:
        """Attempting to load a non-existent file raises ValueError."""
        dist = PoissonDistortion()
        with pytest.raises(ValueError, match="Image not found at path"):
            dist.change_image(tmp_path / "missing.png")

    def test_set_image_validation(self) -> None:
        """Invalid arrays raise ValueError."""
        dist = PoissonDistortion()

        with pytest.raises(ValueError, match="Invalid image array"):
            dist.set_image(np.array([]))

        with pytest.raises(ValueError, match="Expected 2D or 3D"):
            dist.set_image(np.zeros((10, 10, 3, 2), dtype=np.float32))

    def test_set_image_float_clipping(self) -> None:
        """Float arrays are clipped to [0, 1]."""
        dist = PoissonDistortion()
        data = np.array([[-0.5, 1.5], [0.2, 0.8]], dtype=np.float32)
        dist.set_image(data)
        assert dist.img is not None
        assert dist.img.min() == 0.0
        assert dist.img.max() == 1.0


class TestPoissonDistortionSequentialApply:
    """Tests verifying the sequential round-robin behavior of the apply() method."""

    def test_sequential_round_robin(self, sample_rgb_image: np.ndarray) -> None:
        """Calling apply() repeatedly must cycle through all 6 distortion algorithms."""
        dist = PoissonDistortion(image=sample_rgb_image)

        expected_methods = [
            "add_poisson_gaussian_noise",
            "add_radial_vignette_noise",
            "add_spatially_correlated_noise",
            "add_fixed_pattern_noise",
            "add_channel_dependent_noise",
            "add_true_poisson_gaussian_noise",
        ]

        # First cycle through all 6 methods
        for i, expected_name in enumerate(expected_methods):
            assert dist.current_sequence_step == i
            noisy = dist.apply(shot_noise=0.01, read_noise=0.001)
            assert isinstance(noisy, np.ndarray)
            assert noisy.shape == sample_rgb_image.shape
            assert dist.last_applied_method == expected_name

        # 7th invocation must wrap around back to the 1st method (index 0)
        assert dist.current_sequence_step == 0
        noisy_seventh = dist.apply(shot_noise=0.01, read_noise=0.001)
        assert dist.last_applied_method == "add_poisson_gaussian_noise"
        assert noisy_seventh.shape == sample_rgb_image.shape

    def test_reset_sequence(self, sample_rgb_image: np.ndarray) -> None:
        """reset_sequence() resets the sequence step counter to 0."""
        dist = PoissonDistortion(image=sample_rgb_image)
        dist.apply()
        dist.apply()
        assert dist.current_sequence_step == 2

        dist.reset_sequence()
        assert dist.current_sequence_step == 0
        assert dist.last_applied_method is None

    def test_apply_with_explicit_method(self, sample_rgb_image: np.ndarray) -> None:
        """Specifying method explicitly executes that method directly."""
        dist = PoissonDistortion(image=sample_rgb_image)
        res = dist.apply(method="add_radial_vignette_noise", edge_factor=3.0)
        assert dist.last_applied_method == "add_radial_vignette_noise"
        assert res.shape == (64, 64, 3)

        # Invalid method name raises ValueError
        with pytest.raises(ValueError, match="Unknown distortion method"):
            dist.apply(method="non_existent_method")

    def test_apply_with_explicit_step(self, sample_rgb_image: np.ndarray) -> None:
        """Specifying step explicitly executes that step."""
        dist = PoissonDistortion(image=sample_rgb_image)
        res = dist.apply(step=3)
        assert dist.last_applied_method == "add_fixed_pattern_noise"
        assert res.shape == (64, 64, 3)

    def test_apply_passes_new_image(
        self, sample_rgb_image: np.ndarray, sample_gray_image: np.ndarray
    ) -> None:
        """Passing an image argument to apply updates the working image."""
        dist = PoissonDistortion()
        res = dist.apply(image=sample_rgb_image, shot_noise=0.02)
        assert dist.img is not None
        assert res.shape == (64, 64, 3)


class TestIndividualNoiseAlgorithms:
    """Tests validating mathematical behavior of each noise method."""

    def test_add_poisson_gaussian_noise(self, sample_rgb_image: np.ndarray) -> None:
        """Standard heteroscedastic noise modifies image pixels."""
        dist = PoissonDistortion(image=sample_rgb_image)
        noisy = dist.add_poisson_gaussian_noise(shot_noise=0.02, read_noise=0.005)
        assert not np.allclose(noisy, dist.img)
        assert noisy.shape == dist.img.shape

    def test_add_radial_vignette_noise_rgb_and_gray(
        self, sample_rgb_image: np.ndarray, sample_gray_image: np.ndarray
    ) -> None:
        """Vignette noise works on both RGB and 2D grayscale."""
        # RGB
        dist_rgb = PoissonDistortion(image=sample_rgb_image)
        noisy_rgb = dist_rgb.add_radial_vignette_noise(edge_factor=3.0)
        assert noisy_rgb.shape == (64, 64, 3)

        # Grayscale
        dist_gray = PoissonDistortion(image=sample_gray_image)
        noisy_gray = dist_gray.add_radial_vignette_noise(edge_factor=3.0)
        assert noisy_gray.shape == (64, 64)

    def test_add_spatially_correlated_noise(
        self, sample_rgb_image: np.ndarray, sample_gray_image: np.ndarray
    ) -> None:
        """Spatially correlated noise applies Gaussian smoothing across channels."""
        # RGB
        dist_rgb = PoissonDistortion(image=sample_rgb_image)
        noisy_rgb = dist_rgb.add_spatially_correlated_noise(correlation_radius=1.2)
        assert noisy_rgb.shape == (64, 64, 3)

        # Grayscale
        dist_gray = PoissonDistortion(image=sample_gray_image)
        noisy_gray = dist_gray.add_spatially_correlated_noise(correlation_radius=1.2)
        assert noisy_gray.shape == (64, 64)

    def test_add_fixed_pattern_noise(
        self, sample_rgb_image: np.ndarray, sample_gray_image: np.ndarray
    ) -> None:
        """Column FPN introduces vertical stripe variation."""
        # RGB
        dist_rgb = PoissonDistortion(image=sample_rgb_image)
        noisy_rgb = dist_rgb.add_fixed_pattern_noise(column_noise_std=0.05)
        assert noisy_rgb.shape == (64, 64, 3)

        # Grayscale
        dist_gray = PoissonDistortion(image=sample_gray_image)
        noisy_gray = dist_gray.add_fixed_pattern_noise(column_noise_std=0.05)
        assert noisy_gray.shape == (64, 64)

    def test_add_channel_dependent_noise(
        self, sample_rgb_image: np.ndarray, sample_gray_image: np.ndarray
    ) -> None:
        """Channel-dependent noise applies multipliers to RGB, rejects Grayscale."""
        dist_rgb = PoissonDistortion(image=sample_rgb_image)
        noisy_rgb = dist_rgb.add_channel_dependent_noise(
            shot_noise=0.02, channel_multipliers=(2.0, 1.0, 3.0)
        )
        assert noisy_rgb.shape == (64, 64, 3)

        dist_gray = PoissonDistortion(image=sample_gray_image)
        with pytest.raises(ValueError, match="RGB image required"):
            dist_gray.add_channel_dependent_noise()

    def test_add_true_poisson_gaussian_noise(
        self, sample_rgb_image: np.ndarray
    ) -> None:
        """Discrete Poisson sampling runs properly with direct or derived parameters."""
        dist = PoissonDistortion(image=sample_rgb_image)

        # Using direct photon_scale
        noisy1 = dist.add_true_poisson_gaussian_noise(
            photon_scale=500.0, read_noise_std=0.02
        )
        assert noisy1.shape == (64, 64, 3)

        # Derived from shot_noise and read_noise
        noisy2 = dist.add_true_poisson_gaussian_noise(shot_noise=0.02, read_noise=0.001)
        assert noisy2.shape == (64, 64, 3)

    def test_save_output(self, sample_rgb_image: np.ndarray, tmp_path: Path) -> None:
        """save_output writes valid lossless PNG image to disk."""
        dist = PoissonDistortion(image=sample_rgb_image)
        distorted = dist.add_poisson_gaussian_noise()

        out_path = tmp_path / "artifacts" / "noisy.png"
        saved_file = dist.save_output(distorted, out_path)

        assert saved_file.exists()
        loaded = cv2.imread(str(saved_file))
        assert loaded is not None
        assert loaded.shape == (64, 64, 3)
