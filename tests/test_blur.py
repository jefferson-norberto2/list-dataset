"""Unit tests for BlurDistortion and sequential blur application."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from dataset.distortion.algorithms.blur import BlurDistortion


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


class TestBlurDistortionInitialization:
    """Tests for BlurDistortion initialization, image loading, and validation."""

    def test_init_empty(self) -> None:
        """Uninitialized engine has None image attributes and raises if applied."""
        dist = BlurDistortion()
        assert dist.img is None
        assert dist.current_sequence_step == 0
        with pytest.raises(ValueError, match="No image loaded"):
            dist.apply()

    def test_init_with_numpy_array(self, sample_rgb_image: np.ndarray) -> None:
        """Initializing with an array normalizes it to float32 [0.0, 1.0]."""
        dist = BlurDistortion(image=sample_rgb_image)
        assert dist.img is not None
        assert dist.img.dtype == np.float32
        assert dist.height == 64
        assert dist.width == 64
        assert dist.channels == 3

    def test_init_with_file_path(self, sample_image_file: Path) -> None:
        """Initializing with a Path loads and converts the image properly."""
        dist = BlurDistortion(image=sample_image_file)
        assert dist.img is not None
        assert dist.img_name == "test_input"
        assert dist.channels == 3

    def test_init_with_invalid_type(self) -> None:
        """Passing an unsupported type raises TypeError."""
        with pytest.raises(TypeError, match="Unsupported image type"):
            BlurDistortion(image=12345)  # type: ignore[arg-type]

    def test_change_image_missing_file_raises(self, tmp_path: Path) -> None:
        """Attempting to load a non-existent file raises ValueError."""
        dist = BlurDistortion()
        with pytest.raises(ValueError, match="Image not found at path"):
            dist.change_image(tmp_path / "missing.png")

    def test_set_image_validation(self) -> None:
        """Empty or invalid dimension arrays raise ValueError."""
        dist = BlurDistortion()

        with pytest.raises(ValueError, match="Invalid image array"):
            dist.set_image(np.array([]))

        with pytest.raises(ValueError, match="Expected 2D or 3D"):
            dist.set_image(np.zeros((10, 10, 3, 2), dtype=np.float32))

    def test_set_image_float_clipping(self) -> None:
        """Float arrays are clipped to [0.0, 1.0]."""
        dist = BlurDistortion()
        data = np.array([[-0.5, 1.5], [0.2, 0.8]], dtype=np.float32)
        dist.set_image(data)
        assert dist.img is not None
        assert dist.img.min() == 0.0
        assert dist.img.max() == 1.0


class TestBlurDistortionSequentialApply:
    """Tests verifying the sequential round-robin behavior of the apply() method."""

    def test_sequential_round_robin(self, sample_rgb_image: np.ndarray) -> None:
        """Calling apply() repeatedly cycles across all 4 blur algorithms."""
        dist = BlurDistortion(image=sample_rgb_image)
        expected_sequence = [
            "add_gaussian_blur",
            "add_directional_gaussian_blur",
            "add_radial_spatially_varying_blur",
            "add_chromatic_gaussian_blur",
        ]

        # First cycle (steps 0 to 3)
        for i, expected_name in enumerate(expected_sequence):
            assert dist.current_sequence_step == i
            blurred = dist.apply()
            assert dist.last_applied_method == expected_name
            assert blurred.shape == (64, 64, 3)
            assert np.isfinite(blurred).all()

        # Second cycle wraps around
        for expected_name in expected_sequence:
            blurred = dist.apply()
            assert dist.last_applied_method == expected_name

    def test_explicit_method_override(self, sample_rgb_image: np.ndarray) -> None:
        """Specifying method executes it directly without updating the sequence index."""
        dist = BlurDistortion(image=sample_rgb_image)
        assert dist.current_sequence_step == 0

        blurred = dist.apply(method="add_directional_gaussian_blur")
        assert dist.last_applied_method == "add_directional_gaussian_blur"
        assert dist.current_sequence_step == 0  # Counter untouched
        assert blurred.shape == (64, 64, 3)

    def test_explicit_step_override(self, sample_rgb_image: np.ndarray) -> None:
        """Specifying step executes the targeted index without advancing counter."""
        dist = BlurDistortion(image=sample_rgb_image)
        blurred = dist.apply(step=2)
        assert dist.last_applied_method == "add_radial_spatially_varying_blur"
        assert dist.current_sequence_step == 0
        assert blurred.shape == (64, 64, 3)

    def test_unknown_method_raises(self, sample_rgb_image: np.ndarray) -> None:
        """Passing an unknown method name raises ValueError."""
        dist = BlurDistortion(image=sample_rgb_image)
        with pytest.raises(ValueError, match="Unknown distortion method"):
            dist.apply(method="non_existent_blur")

    def test_apply_with_image_argument(
        self, sample_rgb_image: np.ndarray, sample_gray_image: np.ndarray
    ) -> None:
        """Supplying image directly to apply() updates the active working image."""
        dist = BlurDistortion(image=sample_rgb_image)
        assert dist.channels == 3

        # Apply on 2D image directly with method that supports 2D
        blurred = dist.apply(image=sample_gray_image, method="add_gaussian_blur")
        assert dist.channels == 1
        assert blurred.shape == (64, 64)

    def test_reset_sequence(self, sample_rgb_image: np.ndarray) -> None:
        """reset_sequence resets pointer and last_applied_method."""
        dist = BlurDistortion(image=sample_rgb_image)
        dist.apply()
        dist.apply()
        assert dist.current_sequence_step == 2
        assert dist.last_applied_method is not None

        dist.reset_sequence()
        assert dist.current_sequence_step == 0
        assert dist.last_applied_method is None


class TestArgumentResolutionStrategy:
    """Tests for the parameter adaptation and routing strategy."""

    def test_generic_sigma_scaling(self) -> None:
        """Generic sigma parameter maps appropriately to each specific blur algorithm."""
        # Gaussian blur
        res_gauss = BlurDistortion.resolve_blur_arguments(
            "add_gaussian_blur", sigma=4.0
        )
        assert res_gauss["sigma_x"] == 4.0
        assert res_gauss["sigma_y"] is None

        # Directional blur
        res_dir = BlurDistortion.resolve_blur_arguments(
            "add_directional_gaussian_blur", sigma=4.0
        )
        assert res_dir["sigma_x"] == 4.0
        assert res_dir["sigma_y"] == pytest.approx(1.0)
        assert res_dir["angle_deg"] == 45.0

        # Radial spatially varying blur
        res_rad = BlurDistortion.resolve_blur_arguments(
            "add_radial_spatially_varying_blur", sigma=4.0
        )
        assert res_rad["center_sigma"] == pytest.approx(1.0)
        assert res_rad["edge_sigma"] == pytest.approx(7.0)

        # Chromatic blur
        res_chrom = BlurDistortion.resolve_blur_arguments(
            "add_chromatic_gaussian_blur", sigma=2.0
        )
        assert res_chrom["sigmas"] == (
            pytest.approx(2.2),
            pytest.approx(1.8),
            pytest.approx(2.8),
        )

    def test_explicit_kwargs_precedence(self) -> None:
        """Explicit parameters override defaults and generic sigma."""
        res_dir = BlurDistortion.resolve_blur_arguments(
            "add_directional_gaussian_blur",
            sigma=2.0,
            sigma_x=5.0,
            sigma_y=0.5,
            angle_deg=90.0,
        )
        assert res_dir["sigma_x"] == 5.0
        assert res_dir["sigma_y"] == 0.5
        assert res_dir["angle_deg"] == 90.0

        res_rad = BlurDistortion.resolve_blur_arguments(
            "add_radial_spatially_varying_blur",
            sigma=2.0,
            center_sigma=0.2,
            edge_sigma=4.5,
        )
        assert res_rad["center_sigma"] == 0.2
        assert res_rad["edge_sigma"] == 4.5

    def test_extra_unrelated_kwargs_absorbed(
        self, sample_rgb_image: np.ndarray
    ) -> None:
        """Extra unrelated kwargs do not cause errors in apply or individual methods."""
        dist = BlurDistortion(image=sample_rgb_image)
        # Pass parameters from noise or other pipelines
        blurred = dist.apply(
            sigma=2.0,
            shot_noise=0.01,
            read_noise=0.001,
            compression_ratio=20,
        )
        assert np.isfinite(blurred).all()


class TestIndividualBlurMethods:
    """Tests for individual blur algorithm implementations."""

    def test_add_gaussian_blur(self, sample_rgb_image: np.ndarray) -> None:
        """Gaussian blur computes correct blur with and without force_kernel."""
        dist = BlurDistortion(image=sample_rgb_image)

        b1 = dist.add_gaussian_blur(sigma_x=2.0)
        assert b1.shape == (64, 64, 3)

        b2 = dist.add_gaussian_blur(sigma_x=2.0, sigma_y=1.0)
        assert b2.shape == (64, 64, 3)

        # Kernel force with scalar and tuple
        b3 = dist.add_gaussian_blur(sigma_x=2.0, force_kernel=5)
        assert b3.shape == (64, 64, 3)

        b4 = dist.add_gaussian_blur(sigma_x=2.0, force_kernel=(5, 7))
        assert b4.shape == (64, 64, 3)

    def test_add_directional_gaussian_blur(self, sample_rgb_image: np.ndarray) -> None:
        """Directional blur works across various rotation angles."""
        dist = BlurDistortion(image=sample_rgb_image)

        for angle in [0.0, 45.0, 90.0, 135.0, 180.0]:
            blurred = dist.add_directional_gaussian_blur(
                sigma_x=3.0, sigma_y=0.5, angle_deg=angle
            )
            assert blurred.shape == (64, 64, 3)
            assert np.isfinite(blurred).all()

    def test_add_radial_spatially_varying_blur(
        self, sample_rgb_image: np.ndarray, sample_gray_image: np.ndarray
    ) -> None:
        """Radial blur works on both 3D RGB and 2D Grayscale images."""
        dist = BlurDistortion(image=sample_rgb_image)
        blurred_rgb = dist.add_radial_spatially_varying_blur(
            center_sigma=0.2, edge_sigma=4.0
        )
        assert blurred_rgb.shape == (64, 64, 3)

        dist.set_image(sample_gray_image)
        blurred_gray = dist.add_radial_spatially_varying_blur(
            center_sigma=0.2, edge_sigma=4.0
        )
        assert blurred_gray.shape == (64, 64)

    def test_add_chromatic_gaussian_blur(self, sample_rgb_image: np.ndarray) -> None:
        """Chromatic blur blurs RGB channels with different sigmas."""
        dist = BlurDistortion(image=sample_rgb_image)
        blurred = dist.add_chromatic_gaussian_blur(sigmas=(3.0, 1.0, 2.0))
        assert blurred.shape == (64, 64, 3)

    def test_add_chromatic_gaussian_blur_validation(
        self, sample_gray_image: np.ndarray, sample_rgb_image: np.ndarray
    ) -> None:
        """Chromatic blur raises ValueError on non-RGB image or invalid sigmas length."""
        dist = BlurDistortion(image=sample_gray_image)
        with pytest.raises(ValueError, match="RGB image required"):
            dist.add_chromatic_gaussian_blur()

        dist.set_image(sample_rgb_image)
        with pytest.raises(ValueError, match="Expected 3 sigma values"):
            dist.add_chromatic_gaussian_blur(sigmas=(1.0, 2.0))  # type: ignore[arg-type]


class TestSaveOutput:
    """Tests for save_output lossless image export."""

    def test_save_output(self, sample_rgb_image: np.ndarray, tmp_path: Path) -> None:
        """save_output writes valid PNG file to disk."""
        dist = BlurDistortion(image=sample_rgb_image)
        blurred = dist.apply()
        out_path = tmp_path / "subfolder" / "blurred.png"

        saved_path = dist.save_output(blurred, out_path)
        assert saved_path.is_file()
        assert saved_path.stat().st_size > 0

        # Read back and verify dimensions
        loaded = cv2.imread(str(saved_path))
        assert loaded.shape == (64, 64, 3)

    def test_save_output_grayscale(
        self, sample_gray_image: np.ndarray, tmp_path: Path
    ) -> None:
        """save_output handles 2D grayscale images cleanly."""
        dist = BlurDistortion(image=sample_gray_image)
        blurred = dist.apply(method="add_gaussian_blur")
        out_path = tmp_path / "blurred_gray.png"

        saved_path = dist.save_output(blurred, out_path)
        assert saved_path.is_file()
        loaded = cv2.imread(str(saved_path), cv2.IMREAD_GRAYSCALE)
        assert loaded.shape == (64, 64)
