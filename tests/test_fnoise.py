"""Unit tests for FnoiseDistortion and sequential 1/f pink noise application."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from dataset.distortion.algorithms.fnoise import FnoiseDistortion


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


class TestFnoiseDistortionInitialization:
    """Tests for FnoiseDistortion initialization, image loading, and validation."""

    def test_init_empty(self) -> None:
        """Uninitialized engine has None image attributes and raises if applied."""
        dist = FnoiseDistortion()
        assert dist.img is None
        assert dist.current_sequence_step == 0
        with pytest.raises(ValueError, match="No image loaded"):
            dist.apply()

    def test_init_with_numpy_array(self, sample_rgb_image: np.ndarray) -> None:
        """Initializing with an array normalizes it to float32 [0.0, 1.0]."""
        dist = FnoiseDistortion(image=sample_rgb_image)
        assert dist.img is not None
        assert dist.img.dtype == np.float32
        assert dist.height == 64
        assert dist.width == 64
        assert dist.channels == 3

    def test_init_with_file_path(self, sample_image_file: Path) -> None:
        """Initializing with a Path loads and converts the image properly."""
        dist = FnoiseDistortion(image=sample_image_file)
        assert dist.img is not None
        assert dist.img_name == "test_input"
        assert dist.channels == 3

    def test_init_with_invalid_type(self) -> None:
        """Passing an unsupported type raises TypeError."""
        with pytest.raises(TypeError, match="Unsupported image type"):
            FnoiseDistortion(image=12345)  # type: ignore[arg-type]

    def test_change_image_missing_file_raises(self, tmp_path: Path) -> None:
        """Attempting to load a non-existent file raises ValueError."""
        dist = FnoiseDistortion()
        with pytest.raises(ValueError, match="Image not found at path"):
            dist.change_image(tmp_path / "missing.png")

    def test_set_image_validation(self) -> None:
        """Empty or invalid dimension arrays raise ValueError."""
        dist = FnoiseDistortion()

        with pytest.raises(ValueError, match="Invalid image array"):
            dist.set_image(np.array([]))

        with pytest.raises(ValueError, match="Expected 2D or 3D"):
            dist.set_image(np.zeros((10, 10, 3, 2), dtype=np.float32))

    def test_set_image_float_clipping(self) -> None:
        """Float arrays are clipped to [0.0, 1.0]."""
        dist = FnoiseDistortion()
        data = np.array([[-0.5, 1.5], [0.2, 0.8]], dtype=np.float32)
        dist.set_image(data)
        assert dist.img is not None
        assert dist.img.min() == 0.0
        assert dist.img.max() == 1.0


class TestFnoiseDistortionSequentialApply:
    """Tests verifying sequential round-robin behavior of apply()."""

    def test_sequential_round_robin(self, sample_rgb_image: np.ndarray) -> None:
        """Calling apply() repeatedly cycles across all 4 pink noise algorithms."""
        dist = FnoiseDistortion(image=sample_rgb_image)
        expected_sequence = [
            "add_correlated_pink_noise",
            "add_anisotropic_pink_noise",
            "add_bandlimited_pink_noise",
            "add_multiplicative_pink_noise",
        ]

        # First cycle (steps 0 to 3)
        for i, expected_name in enumerate(expected_sequence):
            assert dist.current_sequence_step == i
            modified = dist.apply()
            assert dist.last_applied_method == expected_name
            assert modified.shape == (64, 64, 3)
            assert np.isfinite(modified).all()
            assert 0.0 <= modified.min() <= modified.max() <= 1.0

        # Second cycle wraps around
        for expected_name in expected_sequence:
            modified = dist.apply()
            assert dist.last_applied_method == expected_name

    def test_explicit_method_override(self, sample_rgb_image: np.ndarray) -> None:
        """Specifying method executes it directly without updating sequence index."""
        dist = FnoiseDistortion(image=sample_rgb_image)
        assert dist.current_sequence_step == 0

        modified = dist.apply(method="add_anisotropic_pink_noise")
        assert dist.last_applied_method == "add_anisotropic_pink_noise"
        assert dist.current_sequence_step == 0
        assert modified.shape == (64, 64, 3)

    def test_explicit_step_override(self, sample_rgb_image: np.ndarray) -> None:
        """Specifying step executes the targeted index without advancing counter."""
        dist = FnoiseDistortion(image=sample_rgb_image)
        modified = dist.apply(step=2)
        assert dist.last_applied_method == "add_bandlimited_pink_noise"
        assert dist.current_sequence_step == 0
        assert modified.shape == (64, 64, 3)

    def test_unknown_method_raises(self, sample_rgb_image: np.ndarray) -> None:
        """Passing an unknown method name raises ValueError."""
        dist = FnoiseDistortion(image=sample_rgb_image)
        with pytest.raises(ValueError, match="Unknown distortion method"):
            dist.apply(method="non_existent_fnoise")

    def test_apply_with_image_argument(
        self, sample_rgb_image: np.ndarray, sample_gray_image: np.ndarray
    ) -> None:
        """Supplying image directly to apply() updates the active working image."""
        dist = FnoiseDistortion(image=sample_rgb_image)
        assert dist.channels == 3

        modified = dist.apply(
            image=sample_gray_image, method="add_correlated_pink_noise"
        )
        assert dist.channels == 1
        assert modified.shape == (64, 64)

    def test_reset_sequence(self, sample_rgb_image: np.ndarray) -> None:
        """reset_sequence resets pointer and last_applied_method."""
        dist = FnoiseDistortion(image=sample_rgb_image)
        dist.apply()
        dist.apply()
        assert dist.current_sequence_step == 2
        assert dist.last_applied_method is not None

        dist.reset_sequence()
        assert dist.current_sequence_step == 0
        assert dist.last_applied_method is None


class TestArgumentResolutionStrategy:
    """Tests for parameter adaptation and routing strategy."""

    def test_generic_intensity_and_noise_scaling(self) -> None:
        """Generic intensity / noise / strength maps appropriately to each algorithm."""
        # Correlated pink noise
        res_corr = FnoiseDistortion.resolve_fnoise_arguments(
            "add_correlated_pink_noise", intensity=0.08
        )
        assert res_corr["intensity"] == 0.08
        assert res_corr["spatial_scale"] == 1.0
        assert res_corr["color_correlation"] == 0.85

        # Anisotropic pink noise
        res_aniso = FnoiseDistortion.resolve_fnoise_arguments(
            "add_anisotropic_pink_noise", noise=0.08, scale=2.0
        )
        assert res_aniso["intensity"] == 0.08
        assert res_aniso["scale_u"] == pytest.approx(3.0)
        assert res_aniso["scale_v"] == pytest.approx(1.0)
        assert res_aniso["angle_deg"] == 0.0

        # Bandlimited pink noise
        res_band = FnoiseDistortion.resolve_fnoise_arguments(
            "add_bandlimited_pink_noise", strength=0.08
        )
        assert res_band["intensity"] == 0.08
        assert res_band["f_min"] == 0.02
        assert res_band["f_max"] == 0.4

        # Multiplicative pink noise
        res_mult = FnoiseDistortion.resolve_fnoise_arguments(
            "add_multiplicative_pink_noise", intensity=0.08
        )
        assert res_mult["intensity"] == pytest.approx(0.16)
        assert res_mult["spatial_scale"] == 1.0

    def test_explicit_kwargs_precedence(self) -> None:
        """Explicit parameters override defaults and generic parameters."""
        res_aniso = FnoiseDistortion.resolve_fnoise_arguments(
            "add_anisotropic_pink_noise",
            intensity=0.05,
            scale_u=2.5,
            scale_v=0.8,
            angle_deg=45.0,
        )
        assert res_aniso["scale_u"] == 2.5
        assert res_aniso["scale_v"] == 0.8
        assert res_aniso["angle_deg"] == 45.0

        res_band = FnoiseDistortion.resolve_fnoise_arguments(
            "add_bandlimited_pink_noise",
            intensity=0.05,
            f_min=0.05,
            f_max=0.35,
        )
        assert res_band["f_min"] == 0.05
        assert res_band["f_max"] == 0.35

    def test_extra_unrelated_kwargs_absorbed(
        self, sample_rgb_image: np.ndarray
    ) -> None:
        """Extra unrelated kwargs do not cause errors in apply or individual methods."""
        dist = FnoiseDistortion(image=sample_rgb_image)
        modified = dist.apply(
            intensity=0.05,
            sigma=2.0,
            alpha=0.5,
            shot_noise=0.01,
            read_noise=0.001,
        )
        assert np.isfinite(modified).all()


class TestIndividualFnoiseMethods:
    """Tests for individual pink noise algorithm implementations."""

    def test_add_correlated_pink_noise(
        self, sample_rgb_image: np.ndarray, sample_gray_image: np.ndarray
    ) -> None:
        """Correlated pink noise works on 3D RGB and 2D Grayscale images."""
        dist = FnoiseDistortion(image=sample_rgb_image)
        noisy_rgb = dist.add_correlated_pink_noise(
            intensity=0.1, spatial_scale=1.2, color_correlation=0.9
        )
        assert noisy_rgb.shape == (64, 64, 3)
        assert 0.0 <= noisy_rgb.min() <= noisy_rgb.max() <= 1.0

        dist.set_image(sample_gray_image)
        noisy_gray = dist.add_correlated_pink_noise(intensity=0.1)
        assert noisy_gray.shape == (64, 64)
        assert 0.0 <= noisy_gray.min() <= noisy_gray.max() <= 1.0

    def test_add_anisotropic_pink_noise(
        self, sample_rgb_image: np.ndarray, sample_gray_image: np.ndarray
    ) -> None:
        """Anisotropic pink noise functions on 3D and 2D across orientation angles."""
        dist = FnoiseDistortion(image=sample_rgb_image)
        for angle in [0.0, 45.0, 90.0]:
            noisy = dist.add_anisotropic_pink_noise(
                intensity=0.08, scale_u=2.0, scale_v=0.5, angle_deg=angle
            )
            assert noisy.shape == (64, 64, 3)
            assert np.isfinite(noisy).all()

        dist.set_image(sample_gray_image)
        noisy_gray = dist.add_anisotropic_pink_noise(intensity=0.08)
        assert noisy_gray.shape == (64, 64)

    def test_add_bandlimited_pink_noise(
        self, sample_rgb_image: np.ndarray, sample_gray_image: np.ndarray
    ) -> None:
        """Bandlimited pink noise filters frequencies to the designated octave."""
        dist = FnoiseDistortion(image=sample_rgb_image)
        noisy = dist.add_bandlimited_pink_noise(intensity=0.1, f_min=0.05, f_max=0.3)
        assert noisy.shape == (64, 64, 3)
        assert 0.0 <= noisy.min() <= noisy.max() <= 1.0

        dist.set_image(sample_gray_image)
        noisy_gray = dist.add_bandlimited_pink_noise(
            intensity=0.1, f_min=0.3, f_max=0.05
        )
        assert noisy_gray.shape == (64, 64)

    def test_add_multiplicative_pink_noise(
        self, sample_rgb_image: np.ndarray, sample_gray_image: np.ndarray
    ) -> None:
        """Multiplicative pink noise modulates image intensity."""
        dist = FnoiseDistortion(image=sample_rgb_image)
        noisy = dist.add_multiplicative_pink_noise(intensity=0.15)
        assert noisy.shape == (64, 64, 3)
        assert 0.0 <= noisy.min() <= noisy.max() <= 1.0

        dist.set_image(sample_gray_image)
        noisy_gray = dist.add_multiplicative_pink_noise(intensity=0.15)
        assert noisy_gray.shape == (64, 64)


class TestSaveOutput:
    """Tests for save_output lossless image export."""

    def test_save_output(self, sample_rgb_image: np.ndarray, tmp_path: Path) -> None:
        """save_output writes valid PNG file to disk."""
        dist = FnoiseDistortion(image=sample_rgb_image)
        modified = dist.apply()
        out_path = tmp_path / "subfolder" / "fnoise.png"

        saved_path = dist.save_output(modified, out_path)
        assert saved_path.is_file()
        assert saved_path.stat().st_size > 0

        loaded = cv2.imread(str(saved_path))
        assert loaded.shape == (64, 64, 3)

    def test_save_output_grayscale(
        self, sample_gray_image: np.ndarray, tmp_path: Path
    ) -> None:
        """save_output handles 2D grayscale images cleanly."""
        dist = FnoiseDistortion(image=sample_gray_image)
        modified = dist.apply(method="add_correlated_pink_noise")
        out_path = tmp_path / "fnoise_gray.png"

        saved_path = dist.save_output(modified, out_path)
        assert saved_path.is_file()
        loaded = cv2.imread(str(saved_path), cv2.IMREAD_GRAYSCALE)
        assert loaded.shape == (64, 64)
