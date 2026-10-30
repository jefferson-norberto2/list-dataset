"""Unit tests for Jpeg2000Distortion and sequential JPEG 2000 compression application."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from dataset.distortion.algorithms.jpeg2000 import Jpeg2000Distortion


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
    file_path = tmp_path / "test_jp2_input.png"
    cv2.imwrite(str(file_path), cv2.cvtColor(sample_rgb_image, cv2.COLOR_RGB2BGR))
    return file_path


class TestJpeg2000DistortionInitialization:
    """Tests for Jpeg2000Distortion initialization, image loading, and validation."""

    def test_init_empty(self) -> None:
        """Uninitialized engine has None image attributes and raises if applied."""
        dist = Jpeg2000Distortion()
        assert dist.img is None
        assert dist.current_sequence_step == 0
        with pytest.raises(ValueError, match="No image loaded"):
            dist.apply()

    def test_init_with_numpy_array(self, sample_rgb_image: np.ndarray) -> None:
        """Initializing with an array normalizes it to float32 [0.0, 1.0]."""
        dist = Jpeg2000Distortion(image=sample_rgb_image)
        assert dist.img is not None
        assert dist.img.dtype == np.float32
        assert dist.height == 64
        assert dist.width == 64
        assert dist.channels == 3

    def test_init_with_file_path(self, sample_image_file: Path) -> None:
        """Initializing with a Path loads and converts the image properly."""
        dist = Jpeg2000Distortion(image=sample_image_file)
        assert dist.img is not None
        assert dist.img_name == "test_jp2_input"
        assert dist.channels == 3

    def test_init_with_invalid_type(self) -> None:
        """Passing an unsupported type raises TypeError."""
        with pytest.raises(TypeError, match="Unsupported image type"):
            Jpeg2000Distortion(image=12345)  # type: ignore[arg-type]

    def test_change_image_missing_file_raises(self, tmp_path: Path) -> None:
        """Attempting to load a non-existent file raises ValueError."""
        dist = Jpeg2000Distortion()
        with pytest.raises(ValueError, match="Image not found at path"):
            dist.change_image(tmp_path / "missing.png")

    def test_set_image_validation(self) -> None:
        """Empty or invalid dimension arrays raise ValueError."""
        dist = Jpeg2000Distortion()

        with pytest.raises(ValueError, match="Invalid image array"):
            dist.set_image(np.array([]))

        with pytest.raises(ValueError, match="Expected 2D or 3D"):
            dist.set_image(np.zeros((10, 10, 3, 2), dtype=np.float32))

    def test_set_image_float_clipping(self) -> None:
        """Float arrays are clipped to [0.0, 1.0]."""
        dist = Jpeg2000Distortion()
        data = np.array([[-0.5, 1.5], [0.2, 0.8]], dtype=np.float32)
        dist.set_image(data)
        assert dist.img is not None
        assert dist.img.min() == 0.0
        assert dist.img.max() == 1.0

    def test_set_image_gray(self, sample_gray_image: np.ndarray) -> None:
        """Grayscale arrays set channels=1 and 2D shape."""
        dist = Jpeg2000Distortion(image=sample_gray_image)
        assert dist.img is not None
        assert dist.channels == 1
        assert dist.img.shape == (64, 64)


class TestJpeg2000DistortionSequentialApply:
    """Tests verifying the sequential round-robin behavior of the apply() method."""

    def test_sequential_round_robin(self, sample_rgb_image: np.ndarray) -> None:
        """Calling apply() repeatedly cycles across all 5 JPEG 2000 algorithms."""
        dist = Jpeg2000Distortion(image=sample_rgb_image)
        expected_sequence = [
            "add_jpeg2000_opencv_memory",
            "add_jpeg2000_tiled",
            "add_jpeg2000_resolutions",
            "add_jpeg2000_target_psnr",
            "add_wavelet_quantization_distortion",
        ]

        for expected_method in expected_sequence:
            out = dist.apply(compression_rate=20.0)
            assert dist.last_applied_method == expected_method
            assert isinstance(out, np.ndarray)
            assert out.shape == sample_rgb_image.shape
            assert out.dtype == np.float32

        # After 5 calls, counter should wrap back to 0
        assert dist.current_sequence_step == 0
        dist.apply(compression_rate=20.0)
        assert dist.last_applied_method == "add_jpeg2000_opencv_memory"
        assert dist.current_sequence_step == 1

    def test_explicit_method_selection(self, sample_rgb_image: np.ndarray) -> None:
        """Specifying method explicitly executes that method without advancing sequence pointer."""
        dist = Jpeg2000Distortion(image=sample_rgb_image)
        assert dist.current_sequence_step == 0

        out = dist.apply(method="add_jpeg2000_target_psnr", compression_rate=25.0)
        assert dist.last_applied_method == "add_jpeg2000_target_psnr"
        assert dist.current_sequence_step == 0
        assert out.shape == sample_rgb_image.shape

    def test_unknown_method_raises(self, sample_rgb_image: np.ndarray) -> None:
        """Requesting an unknown distortion method raises ValueError."""
        dist = Jpeg2000Distortion(image=sample_rgb_image)
        with pytest.raises(ValueError, match="Unknown distortion method"):
            dist.apply(method="non_existent_method")

    def test_explicit_step_selection(self, sample_rgb_image: np.ndarray) -> None:
        """Specifying step explicitly executes corresponding method by index."""
        dist = Jpeg2000Distortion(image=sample_rgb_image)

        dist.apply(step=1)
        assert dist.last_applied_method == "add_jpeg2000_tiled"
        assert dist.current_sequence_step == 0

        dist.apply(step=4)
        assert dist.last_applied_method == "add_wavelet_quantization_distortion"
        assert dist.current_sequence_step == 0

    def test_apply_with_transient_image(self, sample_rgb_image: np.ndarray) -> None:
        """Passing image argument to apply() updates internal image before execution."""
        dist = Jpeg2000Distortion()
        assert dist.img is None

        out = dist.apply(image=sample_rgb_image, compression_rate=15.0)
        assert dist.img is not None
        assert out.shape == sample_rgb_image.shape

    def test_reset_sequence(self, sample_rgb_image: np.ndarray) -> None:
        """Calling reset_sequence resets the step index back to 0."""
        dist = Jpeg2000Distortion(image=sample_rgb_image)
        dist.apply()
        dist.apply()
        assert dist.current_sequence_step == 2
        assert dist.last_applied_method is not None

        dist.reset_sequence()
        assert dist.current_sequence_step == 0
        assert dist.last_applied_method is None


class TestJpeg2000ArgumentResolution:
    """Tests for Jpeg2000Distortion.resolve_jpeg2000_arguments static method."""

    def test_resolve_arguments_defaults(self) -> None:
        """Default arguments properly populated for each method."""
        # 1. opencv memory
        args1 = Jpeg2000Distortion.resolve_jpeg2000_arguments(
            "add_jpeg2000_opencv_memory", compression_rate=25.0
        )
        assert args1["compression_ratio"] == pytest.approx(25.0)

        # 2. tiled
        args2 = Jpeg2000Distortion.resolve_jpeg2000_arguments(
            "add_jpeg2000_tiled", compression_rate=20.0
        )
        assert args2["compression_rate"] == pytest.approx(20.0)
        assert args2["tile_size"] == (128, 128)

        # 3. resolutions
        args3 = Jpeg2000Distortion.resolve_jpeg2000_arguments(
            "add_jpeg2000_resolutions", compression_rate=15.0
        )
        assert args3["compression_rate"] == pytest.approx(15.0)
        assert args3["num_resolutions"] == 3

        # 4. target psnr
        args4 = Jpeg2000Distortion.resolve_jpeg2000_arguments(
            "add_jpeg2000_target_psnr", compression_rate=20.0
        )
        assert args4["target_psnr_db"] == pytest.approx(30.0)

        # 5. wavelet quantization
        args5 = Jpeg2000Distortion.resolve_jpeg2000_arguments(
            "add_wavelet_quantization_distortion", compression_rate=20.0
        )
        assert args5["wavelet"] == "bior4.4"
        assert args5["level"] == 3
        assert args5["quant_step"] == pytest.approx(0.15)

    def test_resolve_arguments_custom_overrides(self) -> None:
        """Explicit keyword arguments override automatic mappings."""
        args1 = Jpeg2000Distortion.resolve_jpeg2000_arguments(
            "add_jpeg2000_opencv_memory",
            compression_ratio=50.0,
        )
        assert args1["compression_ratio"] == pytest.approx(50.0)

        args2 = Jpeg2000Distortion.resolve_jpeg2000_arguments(
            "add_jpeg2000_tiled",
            tile_size=(64, 64),
            compression_rate=10.0,
        )
        assert args2["tile_size"] == (64, 64)
        assert args2["compression_rate"] == pytest.approx(10.0)

        args3 = Jpeg2000Distortion.resolve_jpeg2000_arguments(
            "add_jpeg2000_resolutions",
            num_resolutions=2,
        )
        assert args3["num_resolutions"] == 2

        args4 = Jpeg2000Distortion.resolve_jpeg2000_arguments(
            "add_jpeg2000_target_psnr",
            target_psnr=42.0,
        )
        assert args4["target_psnr_db"] == pytest.approx(42.0)

        args5 = Jpeg2000Distortion.resolve_jpeg2000_arguments(
            "add_wavelet_quantization_distortion",
            wavelet="db4",
            level=2,
            quant_step=0.08,
        )
        assert args5["wavelet"] == "db4"
        assert args5["level"] == 2
        assert args5["quant_step"] == pytest.approx(0.08)

    def test_resolve_rate_mappings(self) -> None:
        """Rate variations map correctly to target PSNR and wavelet quant steps."""
        # Low compression (higher quality)
        args_low = Jpeg2000Distortion.resolve_jpeg2000_arguments(
            "add_jpeg2000_target_psnr", compression_rate=10.0
        )
        assert args_low["target_psnr_db"] == pytest.approx(37.5)

        # High compression (lower quality)
        args_high = Jpeg2000Distortion.resolve_jpeg2000_arguments(
            "add_jpeg2000_target_psnr", compression_rate=40.0
        )
        assert args_high["target_psnr_db"] == pytest.approx(15.0)


class TestJpeg2000DistortionMethods:
    """Direct tests for individual JPEG 2000 distortion methods on RGB and Grayscale."""

    def test_add_jpeg2000_opencv_memory_rgb(self, sample_rgb_image: np.ndarray) -> None:
        dist = Jpeg2000Distortion(image=sample_rgb_image)
        out = dist.add_jpeg2000_opencv_memory(compression_ratio=20.0)
        assert out.shape == sample_rgb_image.shape
        assert out.dtype == np.float32
        assert out.min() >= 0.0
        assert out.max() <= 1.0

    def test_add_jpeg2000_opencv_memory_gray(
        self, sample_gray_image: np.ndarray
    ) -> None:
        dist = Jpeg2000Distortion(image=sample_gray_image)
        out = dist.add_jpeg2000_opencv_memory(compression_ratio=15.0)
        assert out.shape == sample_gray_image.shape
        assert out.dtype == np.float32

    def test_add_jpeg2000_opencv_memory_encode_failure(
        self, sample_rgb_image: np.ndarray, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        dist = Jpeg2000Distortion(image=sample_rgb_image)
        monkeypatch.setattr(cv2, "imencode", lambda *args, **kwargs: (False, None))
        out = dist.add_jpeg2000_opencv_memory()
        assert np.array_equal(out, dist.img)

    def test_add_jpeg2000_opencv_memory_decode_failure(
        self, sample_rgb_image: np.ndarray, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        dist = Jpeg2000Distortion(image=sample_rgb_image)
        monkeypatch.setattr(
            cv2, "imencode", lambda *args, **kwargs: (True, np.array([1, 2]))
        )
        monkeypatch.setattr(cv2, "imdecode", lambda *args, **kwargs: None)
        out = dist.add_jpeg2000_opencv_memory()
        assert np.array_equal(out, dist.img)

    def test_add_jpeg2000_tiled_rgb(self, sample_rgb_image: np.ndarray) -> None:
        dist = Jpeg2000Distortion(image=sample_rgb_image)
        out = dist.add_jpeg2000_tiled(compression_rate=15.0, tile_size=(32, 32))
        assert out.shape == sample_rgb_image.shape
        assert out.dtype == np.float32
        assert out.min() >= 0.0
        assert out.max() <= 1.0

    def test_add_jpeg2000_tiled_gray(self, sample_gray_image: np.ndarray) -> None:
        dist = Jpeg2000Distortion(image=sample_gray_image)
        out = dist.add_jpeg2000_tiled(compression_rate=20.0, tile_size=(32, 32))
        assert out.shape == sample_gray_image.shape
        assert out.dtype == np.float32

    def test_add_jpeg2000_resolutions_rgb(self, sample_rgb_image: np.ndarray) -> None:
        dist = Jpeg2000Distortion(image=sample_rgb_image)
        out = dist.add_jpeg2000_resolutions(compression_rate=20.0, num_resolutions=2)
        assert out.shape == sample_rgb_image.shape
        assert out.dtype == np.float32
        assert out.min() >= 0.0
        assert out.max() <= 1.0

    def test_add_jpeg2000_resolutions_gray(self, sample_gray_image: np.ndarray) -> None:
        dist = Jpeg2000Distortion(image=sample_gray_image)
        out = dist.add_jpeg2000_resolutions(compression_rate=20.0, num_resolutions=2)
        assert out.shape == sample_gray_image.shape
        assert out.dtype == np.float32

    def test_add_jpeg2000_target_psnr_rgb(self, sample_rgb_image: np.ndarray) -> None:
        dist = Jpeg2000Distortion(image=sample_rgb_image)
        out = dist.add_jpeg2000_target_psnr(target_psnr_db=32.0)
        assert out.shape == sample_rgb_image.shape
        assert out.dtype == np.float32
        assert out.min() >= 0.0
        assert out.max() <= 1.0

    def test_add_jpeg2000_target_psnr_gray(self, sample_gray_image: np.ndarray) -> None:
        dist = Jpeg2000Distortion(image=sample_gray_image)
        out = dist.add_jpeg2000_target_psnr(target_psnr_db=35.0)
        assert out.shape == sample_gray_image.shape
        assert out.dtype == np.float32

    def test_add_wavelet_quantization_distortion_rgb(
        self, sample_rgb_image: np.ndarray
    ) -> None:
        dist = Jpeg2000Distortion(image=sample_rgb_image)
        out = dist.add_wavelet_quantization_distortion(
            wavelet="bior4.4", level=2, quant_step=0.2
        )
        assert out.shape == sample_rgb_image.shape
        assert out.dtype == np.float32
        assert out.min() >= 0.0
        assert out.max() <= 1.0

    def test_add_wavelet_quantization_distortion_gray(
        self, sample_gray_image: np.ndarray
    ) -> None:
        dist = Jpeg2000Distortion(image=sample_gray_image)
        out = dist.add_wavelet_quantization_distortion(
            wavelet="bior4.4", level=2, quant_step=0.2
        )
        assert out.shape == sample_gray_image.shape
        assert out.dtype == np.float32

    def test_methods_raise_if_no_image_loaded(self) -> None:
        """Every distortion method raises ValueError when called without loaded image."""
        dist = Jpeg2000Distortion()
        for method_name in Jpeg2000Distortion.DISTORTION_METHODS:
            fn = getattr(dist, method_name)
            with pytest.raises(ValueError, match="No image loaded"):
                fn()


class TestJpeg2000DistortionSaveOutput:
    """Tests artifact saving and lossless PNG output."""

    def test_save_output_rgb(
        self, tmp_path: Path, sample_rgb_image: np.ndarray
    ) -> None:
        dist = Jpeg2000Distortion(image=sample_rgb_image)
        out = dist.apply(compression_rate=20.0)
        save_file = tmp_path / "jp2_compressed_output.png"

        saved_path = dist.save_output(out, save_file)
        assert saved_path.exists()
        assert saved_path == save_file.resolve()

        read_bgr = cv2.imread(str(saved_path))
        assert read_bgr is not None
        assert read_bgr.shape == (64, 64, 3)

    def test_save_output_gray(
        self, tmp_path: Path, sample_gray_image: np.ndarray
    ) -> None:
        dist = Jpeg2000Distortion(image=sample_gray_image)
        out = dist.apply(compression_rate=20.0)
        save_file = tmp_path / "jp2_compressed_gray.png"

        saved_path = dist.save_output(out, save_file)
        assert saved_path.exists()

        read_gray = cv2.imread(str(saved_path), cv2.IMREAD_GRAYSCALE)
        assert read_gray is not None
        assert read_gray.shape == (64, 64)
