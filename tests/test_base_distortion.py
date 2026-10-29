"""Unit tests for BaseDistortion abstract class and common lifecycle methods."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pytest

from dataset.distortion.algorithms.base import BaseDistortion


class DummyDistortion(BaseDistortion):
    """Concrete subclass for testing BaseDistortion behavior."""

    DISTORTION_METHODS = ("dummy_method_a", "dummy_method_b")

    def dummy_method_a(self, scale: float = 1.0) -> np.ndarray:
        self._check_image_loaded()
        assert self.img is not None
        return np.clip(self.img * scale, 0.0, 1.0)

    def dummy_method_b(self, offset: float = 0.1) -> np.ndarray:
        self._check_image_loaded()
        assert self.img is not None
        return np.clip(self.img + offset, 0.0, 1.0)

    def apply(
        self,
        *args: Any,
        image: np.ndarray | None = None,
        method: str | None = None,
        step: int | None = None,
        **kwargs: Any,
    ) -> np.ndarray:
        if image is not None:
            self.set_image(image)
        self._check_image_loaded()
        _, method_fn = self._select_method(method=method, step=step)
        return method_fn(**kwargs)


@pytest.fixture
def sample_rgb_image() -> np.ndarray:
    """Deterministic 32x32 RGB uint8 image."""
    rng = np.random.default_rng(123)
    return rng.integers(0, 256, size=(32, 32, 3), dtype=np.uint8)


@pytest.fixture
def sample_gray_image() -> np.ndarray:
    """Deterministic 32x32 grayscale uint8 image."""
    rng = np.random.default_rng(123)
    return rng.integers(0, 256, size=(32, 32), dtype=np.uint8)


@pytest.fixture
def sample_image_file(tmp_path: Path, sample_rgb_image: np.ndarray) -> Path:
    """Writes a sample RGB image to file."""
    file_path = tmp_path / "sample_img.png"
    cv2.imwrite(str(file_path), cv2.cvtColor(sample_rgb_image, cv2.COLOR_RGB2BGR))
    return file_path


class TestBaseDistortionAbstractInterface:
    """Tests ensuring BaseDistortion enforces abstract contract."""

    def test_cannot_instantiate_base_class_directly(self) -> None:
        """BaseDistortion is abstract and cannot be directly instantiated."""
        with pytest.raises(
            TypeError, match="Can't instantiate abstract class BaseDistortion"
        ):
            BaseDistortion()  # type: ignore[abstract]

    def test_unimplemented_apply_raises(self) -> None:
        """Subclass missing apply() cannot be instantiated."""

        class IncompleteDistortion(BaseDistortion):
            pass

        with pytest.raises(
            TypeError, match="Can't instantiate abstract class IncompleteDistortion"
        ):
            IncompleteDistortion()  # type: ignore[abstract]


class TestBaseDistortionLifecycle:
    """Tests lifecycle, image loading, validation, and properties."""

    def test_init_empty(self) -> None:
        dist = DummyDistortion()
        assert dist.img is None
        assert dist.img_uint8 is None
        assert dist.height is None
        assert dist.width is None
        assert dist.channels is None
        assert dist.img_name == "image"
        assert dist.current_sequence_step == 0
        assert dist.last_applied_method is None

    def test_init_with_unsupported_type(self) -> None:
        with pytest.raises(TypeError, match="Unsupported image type: int"):
            DummyDistortion(image=123)  # type: ignore[arg-type]

    def test_init_with_file_path(self, sample_image_file: Path) -> None:
        dist = DummyDistortion(image=sample_image_file)
        assert dist.img is not None
        assert dist.img_name == "sample_img"
        assert dist.height == 32
        assert dist.width == 32
        assert dist.channels == 3

    def test_init_with_str_path(self, sample_image_file: Path) -> None:
        dist = DummyDistortion(image=str(sample_image_file))
        assert dist.img is not None
        assert dist.img_name == "sample_img"

    def test_change_image_missing_file(self, tmp_path: Path) -> None:
        dist = DummyDistortion()
        non_existent = tmp_path / "does_not_exist.png"
        with pytest.raises(ValueError, match="Image not found at path"):
            dist.change_image(non_existent)

    def test_set_image_validation(self) -> None:
        dist = DummyDistortion()
        with pytest.raises(ValueError, match="Invalid image array provided"):
            dist.set_image(np.array([]))

        with pytest.raises(ValueError, match="Expected 2D or 3D image array"):
            dist.set_image(np.zeros((10,), dtype=np.uint8))

        with pytest.raises(ValueError, match="Expected 2D or 3D image array"):
            dist.set_image(np.zeros((10, 10, 3, 2), dtype=np.uint8))

    def test_set_image_gray(self, sample_gray_image: np.ndarray) -> None:
        dist = DummyDistortion()
        dist.set_image(sample_gray_image)
        assert dist.img is not None
        assert dist.img.shape == (32, 32)
        assert dist.channels == 1
        assert dist.height == 32
        assert dist.width == 32
        assert dist.img_uint8 is not None
        assert dist.img_uint8.dtype == np.uint8

    def test_set_image_float(self) -> None:
        dist = DummyDistortion()
        float_img = np.array([[0.0, 0.5], [1.0, 1.5]], dtype=np.float32)
        dist.set_image(float_img)
        assert dist.img is not None
        assert dist.img.max() <= 1.0
        assert dist.img_uint8 is not None
        assert dist.img_uint8.max() <= 255

    def test_check_image_loaded_raises_when_none(self) -> None:
        dist = DummyDistortion()
        with pytest.raises(ValueError, match="No image loaded"):
            dist._check_image_loaded()


class TestBaseDistortionMethodSelection:
    """Tests round-robin selection and method overriding."""

    def test_round_robin_progression(self, sample_rgb_image: np.ndarray) -> None:
        dist = DummyDistortion(sample_rgb_image)
        assert dist.current_sequence_step == 0

        # First call -> dummy_method_a
        out1 = dist.apply()
        assert dist.last_applied_method == "dummy_method_a"
        assert dist.current_sequence_step == 1
        assert out1.shape == (32, 32, 3)

        # Second call -> dummy_method_b
        out2 = dist.apply()
        assert dist.last_applied_method == "dummy_method_b"
        assert dist.current_sequence_step == 0  # wrapped around
        assert out2.shape == (32, 32, 3)

    def test_explicit_method(self, sample_rgb_image: np.ndarray) -> None:
        dist = DummyDistortion(sample_rgb_image)
        dist.apply(method="dummy_method_b")
        assert dist.last_applied_method == "dummy_method_b"
        # sequence step counter should not advance when method is explicit
        assert dist.current_sequence_step == 0

    def test_unknown_method_raises(self, sample_rgb_image: np.ndarray) -> None:
        dist = DummyDistortion(sample_rgb_image)
        with pytest.raises(
            ValueError, match="Unknown distortion method 'invalid_method'"
        ):
            dist.apply(method="invalid_method")

    def test_explicit_step(self, sample_rgb_image: np.ndarray) -> None:
        dist = DummyDistortion(sample_rgb_image)
        dist.apply(step=1)
        assert dist.last_applied_method == "dummy_method_b"
        assert dist.current_sequence_step == 0  # sequence counter untouched

    def test_reset_sequence(self, sample_rgb_image: np.ndarray) -> None:
        dist = DummyDistortion(sample_rgb_image)
        dist.apply()
        assert dist.current_sequence_step == 1
        assert dist.last_applied_method == "dummy_method_a"

        dist.reset_sequence()
        assert dist.current_sequence_step == 0
        assert dist.last_applied_method is None


class TestBaseDistortionSaveOutput:
    """Tests artifact saving and lossless PNG output."""

    def test_save_output_rgb(
        self, tmp_path: Path, sample_rgb_image: np.ndarray
    ) -> None:
        dist = DummyDistortion(sample_rgb_image)
        out = dist.apply(method="dummy_method_a")
        save_file = tmp_path / "sub" / "output_rgb.png"

        saved_path = dist.save_output(out, save_file)
        assert saved_path.exists()
        assert saved_path == save_file.resolve()

        # Read back with OpenCV
        read_bgr = cv2.imread(str(saved_path))
        assert read_bgr is not None
        assert read_bgr.shape == (32, 32, 3)

    def test_save_output_grayscale(
        self, tmp_path: Path, sample_gray_image: np.ndarray
    ) -> None:
        dist = DummyDistortion(sample_gray_image)
        out = dist.apply(method="dummy_method_a")
        save_file = tmp_path / "output_gray.png"

        saved_path = dist.save_output(out, save_file)
        assert saved_path.exists()

        read_gray = cv2.imread(str(saved_path), cv2.IMREAD_GRAYSCALE)
        assert read_gray is not None
        assert read_gray.shape == (32, 32)
