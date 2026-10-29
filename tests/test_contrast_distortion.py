"""Unit tests for ContrastDistortion and sequential contrast application."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from dataset.distortion.algorithms.contrast import ContrastDistortion


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


class TestContrastDistortionInitialization:
    """Tests for ContrastDistortion initialization, image loading, and validation."""

    def test_init_empty(self) -> None:
        """Uninitialized engine has None image attributes and raises if applied."""
        dist = ContrastDistortion()
        assert dist.img is None
        assert dist.current_sequence_step == 0
        with pytest.raises(ValueError, match="No image loaded"):
            dist.apply()

    def test_init_with_numpy_array(self, sample_rgb_image: np.ndarray) -> None:
        """Initializing with an array normalizes it to float32 [0.0, 1.0]."""
        dist = ContrastDistortion(image=sample_rgb_image)
        assert dist.img is not None
        assert dist.img.dtype == np.float32
        assert dist.height == 64
        assert dist.width == 64
        assert dist.channels == 3

    def test_init_with_file_path(self, sample_image_file: Path) -> None:
        """Initializing with a Path loads and converts the image properly."""
        dist = ContrastDistortion(image=sample_image_file)
        assert dist.img is not None
        assert dist.img_name == "test_input"
        assert dist.channels == 3

    def test_init_with_invalid_type(self) -> None:
        """Passing an unsupported type raises TypeError."""
        with pytest.raises(TypeError, match="Unsupported image type"):
            ContrastDistortion(image=12345)  # type: ignore[arg-type]

    def test_change_image_missing_file_raises(self, tmp_path: Path) -> None:
        """Attempting to load a non-existent file raises ValueError."""
        dist = ContrastDistortion()
        with pytest.raises(ValueError, match="Image not found at path"):
            dist.change_image(tmp_path / "missing.png")

    def test_set_image_validation(self) -> None:
        """Empty or invalid dimension arrays raise ValueError."""
        dist = ContrastDistortion()

        with pytest.raises(ValueError, match="Invalid image array"):
            dist.set_image(np.array([]))

        with pytest.raises(ValueError, match="Expected 2D or 3D"):
            dist.set_image(np.zeros((10, 10, 3, 2), dtype=np.float32))

    def test_set_image_float_clipping(self) -> None:
        """Float arrays are clipped to [0.0, 1.0]."""
        dist = ContrastDistortion()
        data = np.array([[-0.5, 1.5], [0.2, 0.8]], dtype=np.float32)
        dist.set_image(data)
        assert dist.img is not None
        assert dist.img.min() == 0.0
        assert dist.img.max() == 1.0


class TestContrastDistortionSequentialApply:
    """Tests verifying sequential round-robin behavior of apply()."""

    def test_sequential_round_robin(self, sample_rgb_image: np.ndarray) -> None:
        """Calling apply() repeatedly cycles across all 4 contrast algorithms."""
        dist = ContrastDistortion(image=sample_rgb_image)
        expected_sequence = [
            "change_contrast_mean_adaptive",
            "change_contrast_sigmoidal",
            "change_contrast_luminance",
            "change_contrast_range_scale",
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
        """Specifying method executes it directly without updating the sequence index."""
        dist = ContrastDistortion(image=sample_rgb_image)
        assert dist.current_sequence_step == 0

        modified = dist.apply(method="change_contrast_sigmoidal")
        assert dist.last_applied_method == "change_contrast_sigmoidal"
        assert dist.current_sequence_step == 0  # Counter untouched
        assert modified.shape == (64, 64, 3)

    def test_explicit_step_override(self, sample_rgb_image: np.ndarray) -> None:
        """Specifying step executes the targeted index without advancing counter."""
        dist = ContrastDistortion(image=sample_rgb_image)
        modified = dist.apply(step=3)
        assert dist.last_applied_method == "change_contrast_range_scale"
        assert dist.current_sequence_step == 0
        assert modified.shape == (64, 64, 3)

    def test_unknown_method_raises(self, sample_rgb_image: np.ndarray) -> None:
        """Passing an unknown method name raises ValueError."""
        dist = ContrastDistortion(image=sample_rgb_image)
        with pytest.raises(ValueError, match="Unknown distortion method"):
            dist.apply(method="non_existent_contrast")

    def test_apply_with_image_argument(
        self, sample_rgb_image: np.ndarray, sample_gray_image: np.ndarray
    ) -> None:
        """Supplying image directly to apply() updates the active working image."""
        dist = ContrastDistortion(image=sample_rgb_image)
        assert dist.channels == 3

        # Apply on 2D image directly with method that supports 2D
        modified = dist.apply(
            image=sample_gray_image, method="change_contrast_mean_adaptive"
        )
        assert dist.channels == 1
        assert modified.shape == (64, 64)

    def test_reset_sequence(self, sample_rgb_image: np.ndarray) -> None:
        """reset_sequence resets pointer and last_applied_method."""
        dist = ContrastDistortion(image=sample_rgb_image)
        dist.apply()
        dist.apply()
        assert dist.current_sequence_step == 2
        assert dist.last_applied_method is not None

        dist.reset_sequence()
        assert dist.current_sequence_step == 0
        assert dist.last_applied_method is None


class TestArgumentResolutionStrategy:
    """Tests for parameter adaptation and routing strategy."""

    def test_generic_alpha_and_factor_scaling(self) -> None:
        """Generic alpha / factor / contrast maps appropriately to each algorithm."""
        # Mean adaptive
        res_mean = ContrastDistortion.resolve_contrast_arguments(
            "change_contrast_mean_adaptive", alpha=0.4
        )
        assert res_mean["alpha"] == 0.4
        assert res_mean["per_channel"] is False

        # Sigmoidal
        res_sig = ContrastDistortion.resolve_contrast_arguments(
            "change_contrast_sigmoidal", alpha=0.4
        )
        assert res_sig["slope"] == pytest.approx(4.0)
        assert res_sig["midpoint"] == 0.5

        # Luminance
        res_lum = ContrastDistortion.resolve_contrast_arguments(
            "change_contrast_luminance", factor=0.4
        )
        assert res_lum["alpha"] == 0.4

        # Range scale
        res_rng = ContrastDistortion.resolve_contrast_arguments(
            "change_contrast_range_scale", contrast=0.4
        )
        assert res_rng["low_bound"] == pytest.approx(0.3)
        assert res_rng["high_bound"] == pytest.approx(0.7)

    def test_explicit_kwargs_precedence(self) -> None:
        """Explicit parameters override defaults and generic alpha."""
        res_sig = ContrastDistortion.resolve_contrast_arguments(
            "change_contrast_sigmoidal",
            alpha=0.5,
            slope=15.0,
            midpoint=0.4,
        )
        assert res_sig["slope"] == 15.0
        assert res_sig["midpoint"] == 0.4

        res_rng = ContrastDistortion.resolve_contrast_arguments(
            "change_contrast_range_scale",
            alpha=0.5,
            low_bound=0.1,
            high_bound=0.9,
        )
        assert res_rng["low_bound"] == 0.1
        assert res_rng["high_bound"] == 0.9

    def test_extra_unrelated_kwargs_absorbed(
        self, sample_rgb_image: np.ndarray
    ) -> None:
        """Extra unrelated kwargs do not cause errors in apply or individual methods."""
        dist = ContrastDistortion(image=sample_rgb_image)
        modified = dist.apply(
            alpha=0.5,
            sigma=2.0,
            shot_noise=0.01,
            read_noise=0.001,
        )
        assert np.isfinite(modified).all()


class TestIndividualContrastMethods:
    """Tests for individual contrast algorithm implementations."""

    def test_change_contrast_mean_adaptive(
        self, sample_rgb_image: np.ndarray, sample_gray_image: np.ndarray
    ) -> None:
        """Mean adaptive contrast scales variance around global or per-channel mean."""
        dist = ContrastDistortion(image=sample_rgb_image)

        # Global scaling
        c_global = dist.change_contrast_mean_adaptive(alpha=0.2, per_channel=False)
        assert c_global.shape == (64, 64, 3)
        # Low alpha compresses contrast towards mean, reducing std dev
        assert np.std(c_global) < np.std(dist.img)

        # Per channel scaling
        c_chan = dist.change_contrast_mean_adaptive(alpha=0.2, per_channel=True)
        assert c_chan.shape == (64, 64, 3)

        # 2D Grayscale
        dist.set_image(sample_gray_image)
        c_gray = dist.change_contrast_mean_adaptive(alpha=0.5)
        assert c_gray.shape == (64, 64)

    def test_change_contrast_sigmoidal(self, sample_rgb_image: np.ndarray) -> None:
        """Sigmoidal contrast maps dynamic range through an S-curve."""
        dist = ContrastDistortion(image=sample_rgb_image)

        # Typical slope
        c_sig = dist.change_contrast_sigmoidal(slope=8.0, midpoint=0.5)
        assert c_sig.shape == (64, 64, 3)
        assert 0.0 <= c_sig.min() <= c_sig.max() <= 1.0

        # Extreme slope values (guarding numerical stability)
        c_extreme = dist.change_contrast_sigmoidal(slope=100.0, midpoint=0.5)
        assert np.isfinite(c_extreme).all()

        # Zero slope handles zero division cleanly
        c_zero = dist.change_contrast_sigmoidal(slope=0.0, midpoint=0.5)
        assert np.isfinite(c_zero).all()

    def test_change_contrast_luminance(
        self, sample_rgb_image: np.ndarray, sample_gray_image: np.ndarray
    ) -> None:
        """Luminance contrast adjusts L* channel in CIELAB space."""
        dist = ContrastDistortion(image=sample_rgb_image)
        c_lum = dist.change_contrast_luminance(alpha=0.4)
        assert c_lum.shape == (64, 64, 3)
        assert np.isfinite(c_lum).all()

        # Grayscale image raises ValueError
        dist.set_image(sample_gray_image)
        with pytest.raises(ValueError, match="RGB image required"):
            dist.change_contrast_luminance()

    def test_change_contrast_range_scale(self, sample_rgb_image: np.ndarray) -> None:
        """Range scale restricts values to specified bounds."""
        dist = ContrastDistortion(image=sample_rgb_image)

        c_range = dist.change_contrast_range_scale(low_bound=0.2, high_bound=0.8)
        assert c_range.shape == (64, 64, 3)
        assert float(c_range.min()) >= 0.2 - 1e-4
        assert float(c_range.max()) <= 0.8 + 1e-4

        # Inverted bounds get automatically sorted
        c_inv = dist.change_contrast_range_scale(low_bound=0.9, high_bound=0.1)
        assert float(c_inv.min()) >= 0.1 - 1e-4
        assert float(c_inv.max()) <= 0.9 + 1e-4


class TestSaveOutput:
    """Tests for save_output lossless image export."""

    def test_save_output(self, sample_rgb_image: np.ndarray, tmp_path: Path) -> None:
        """save_output writes valid PNG file to disk."""
        dist = ContrastDistortion(image=sample_rgb_image)
        modified = dist.apply()
        out_path = tmp_path / "subfolder" / "contrast.png"

        saved_path = dist.save_output(modified, out_path)
        assert saved_path.is_file()
        assert saved_path.stat().st_size > 0

        # Read back and verify dimensions
        loaded = cv2.imread(str(saved_path))
        assert loaded.shape == (64, 64, 3)

    def test_save_output_grayscale(
        self, sample_gray_image: np.ndarray, tmp_path: Path
    ) -> None:
        """save_output handles 2D grayscale images cleanly."""
        dist = ContrastDistortion(image=sample_gray_image)
        modified = dist.apply(method="change_contrast_mean_adaptive")
        out_path = tmp_path / "contrast_gray.png"

        saved_path = dist.save_output(modified, out_path)
        assert saved_path.is_file()
        loaded = cv2.imread(str(saved_path), cv2.IMREAD_GRAYSCALE)
        assert loaded.shape == (64, 64)
