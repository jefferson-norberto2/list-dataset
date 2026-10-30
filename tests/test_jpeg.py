"""Unit tests for JpegDistortion and sequential JPEG compression application."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from dataset.distortion.algorithms.jpeg import JpegDistortion


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


class TestJpegDistortionInitialization:
    """Tests for JpegDistortion initialization, image loading, and validation."""

    def test_init_empty(self) -> None:
        """Uninitialized engine has None image attributes and raises if applied."""
        dist = JpegDistortion()
        assert dist.img is None
        assert dist.current_sequence_step == 0
        with pytest.raises(ValueError, match="No image loaded"):
            dist.apply()

    def test_init_with_numpy_array(self, sample_rgb_image: np.ndarray) -> None:
        """Initializing with an array normalizes it to float32 [0.0, 1.0]."""
        dist = JpegDistortion(image=sample_rgb_image)
        assert dist.img is not None
        assert dist.img.dtype == np.float32
        assert dist.height == 64
        assert dist.width == 64
        assert dist.channels == 3

    def test_init_with_file_path(self, sample_image_file: Path) -> None:
        """Initializing with a Path loads and converts the image properly."""
        dist = JpegDistortion(image=sample_image_file)
        assert dist.img is not None
        assert dist.img_name == "test_input"
        assert dist.channels == 3

    def test_init_with_invalid_type(self) -> None:
        """Passing an unsupported type raises TypeError."""
        with pytest.raises(TypeError, match="Unsupported image type"):
            JpegDistortion(image=12345)  # type: ignore[arg-type]

    def test_change_image_missing_file_raises(self, tmp_path: Path) -> None:
        """Attempting to load a non-existent file raises ValueError."""
        dist = JpegDistortion()
        with pytest.raises(ValueError, match="Image not found at path"):
            dist.change_image(tmp_path / "missing.png")

    def test_set_image_validation(self) -> None:
        """Empty or invalid dimension arrays raise ValueError."""
        dist = JpegDistortion()

        with pytest.raises(ValueError, match="Invalid image array"):
            dist.set_image(np.array([]))

        with pytest.raises(ValueError, match="Expected 2D or 3D"):
            dist.set_image(np.zeros((10, 10, 3, 2), dtype=np.float32))

    def test_set_image_float_clipping(self) -> None:
        """Float arrays are clipped to [0.0, 1.0]."""
        dist = JpegDistortion()
        data = np.array([[-0.5, 1.5], [0.2, 0.8]], dtype=np.float32)
        dist.set_image(data)
        assert dist.img is not None
        assert dist.img.min() == 0.0
        assert dist.img.max() == 1.0

    def test_set_image_gray(self, sample_gray_image: np.ndarray) -> None:
        """Grayscale arrays set channels=1 and 2D shape."""
        dist = JpegDistortion(image=sample_gray_image)
        assert dist.img is not None
        assert dist.channels == 1
        assert dist.img.shape == (64, 64)


class TestJpegDistortionSequentialApply:
    """Tests verifying the sequential round-robin behavior of the apply() method."""

    def test_sequential_round_robin(self, sample_rgb_image: np.ndarray) -> None:
        """Calling apply() repeatedly cycles across all 5 JPEG algorithms."""
        dist = JpegDistortion(image=sample_rgb_image)
        expected_sequence = [
            "add_jpeg_with_subsampling",
            "add_jpeg_opencv",
            "add_jpeg_grid_shift",
            "add_double_jpeg_compression",
            "add_custom_quantized_jpeg",
        ]

        for expected_method in expected_sequence:
            out = dist.apply(quality=15)
            assert dist.last_applied_method == expected_method
            assert isinstance(out, np.ndarray)
            assert out.shape == sample_rgb_image.shape
            assert out.dtype == np.float32

        # After 5 calls, counter should have wrapped back to 0
        assert dist.current_sequence_step == 0
        dist.apply(quality=15)
        assert dist.last_applied_method == "add_jpeg_with_subsampling"
        assert dist.current_sequence_step == 1

    def test_explicit_method_selection(self, sample_rgb_image: np.ndarray) -> None:
        """Specifying method explicitly executes that method without advancing sequence pointer."""
        dist = JpegDistortion(image=sample_rgb_image)
        assert dist.current_sequence_step == 0

        out = dist.apply(method="add_jpeg_opencv", quality=20)
        assert dist.last_applied_method == "add_jpeg_opencv"
        assert dist.current_sequence_step == 0
        assert out.shape == sample_rgb_image.shape

    def test_unknown_method_raises(self, sample_rgb_image: np.ndarray) -> None:
        """Requesting an unknown distortion method raises ValueError."""
        dist = JpegDistortion(image=sample_rgb_image)
        with pytest.raises(ValueError, match="Unknown distortion method"):
            dist.apply(method="non_existent_method")

    def test_explicit_step_selection(self, sample_rgb_image: np.ndarray) -> None:
        """Specifying step explicitly executes corresponding method by index."""
        dist = JpegDistortion(image=sample_rgb_image)

        dist.apply(step=2)
        assert dist.last_applied_method == "add_jpeg_grid_shift"
        assert dist.current_sequence_step == 0

        dist.apply(step=4)
        assert dist.last_applied_method == "add_custom_quantized_jpeg"
        assert dist.current_sequence_step == 0

    def test_apply_with_transient_image(self, sample_rgb_image: np.ndarray) -> None:
        """Passing image argument to apply() updates internal image before execution."""
        dist = JpegDistortion()
        assert dist.img is None

        out = dist.apply(image=sample_rgb_image, quality=10)
        assert dist.img is not None
        assert out.shape == sample_rgb_image.shape

    def test_reset_sequence(self, sample_rgb_image: np.ndarray) -> None:
        """Calling reset_sequence resets the step index back to 0."""
        dist = JpegDistortion(image=sample_rgb_image)
        dist.apply()
        dist.apply()
        assert dist.current_sequence_step == 2
        assert dist.last_applied_method is not None

        dist.reset_sequence()
        assert dist.current_sequence_step == 0
        assert dist.last_applied_method is None


class TestJpegArgumentResolution:
    """Tests for JpegDistortion.resolve_jpeg_arguments static method."""

    def test_resolve_arguments_defaults(self) -> None:
        """Default arguments properly populated for each method."""
        # 1. subsampling
        args1 = JpegDistortion.resolve_jpeg_arguments(
            "add_jpeg_with_subsampling", quality=15
        )
        assert args1["quality"] == 15
        assert args1["subsampling"] == 2

        # 2. opencv
        args2 = JpegDistortion.resolve_jpeg_arguments("add_jpeg_opencv", quality=25)
        assert args2["quality"] == 25

        # 3. grid shift
        args3 = JpegDistortion.resolve_jpeg_arguments("add_jpeg_grid_shift", quality=12)
        assert args3["quality"] == 12
        assert args3["shift_x"] == 4
        assert args3["shift_y"] == 4

        # 4. double compression
        args4 = JpegDistortion.resolve_jpeg_arguments(
            "add_double_jpeg_compression", quality=10
        )
        assert args4["quality_2"] == 10
        assert args4["quality_1"] == 20

        # 5. custom quantized
        args5 = JpegDistortion.resolve_jpeg_arguments(
            "add_custom_quantized_jpeg", quality=10
        )
        assert args5["high_freq_quant_factor"] == pytest.approx(8.0)

    def test_resolve_arguments_custom_overrides(self) -> None:
        """Explicit keyword arguments override automatic mappings."""
        args1 = JpegDistortion.resolve_jpeg_arguments(
            "add_jpeg_with_subsampling",
            quality=50,
            subsampling=0,
        )
        assert args1["quality"] == 50
        assert args1["subsampling"] == 0

        args3 = JpegDistortion.resolve_jpeg_arguments(
            "add_jpeg_grid_shift",
            quality=30,
            shift_x=2,
            shift_y=6,
        )
        assert args3["shift_x"] == 2
        assert args3["shift_y"] == 6

        args4 = JpegDistortion.resolve_jpeg_arguments(
            "add_double_jpeg_compression",
            quality=20,
            quality_1=50,
            quality_2=5,
        )
        assert args4["quality_1"] == 50
        assert args4["quality_2"] == 5

        args5 = JpegDistortion.resolve_jpeg_arguments(
            "add_custom_quantized_jpeg",
            factor=12.5,
        )
        assert args5["high_freq_quant_factor"] == pytest.approx(12.5)

    def test_resolve_quant_factor_mapping(self) -> None:
        """Inverse quality mapping scales high_freq_quant_factor sensibly."""
        # High quality -> lower quant factor
        args_high = JpegDistortion.resolve_jpeg_arguments(
            "add_custom_quantized_jpeg", quality=80
        )
        assert args_high["high_freq_quant_factor"] == pytest.approx(1.0)

        # Low quality -> higher quant factor
        args_low = JpegDistortion.resolve_jpeg_arguments(
            "add_custom_quantized_jpeg", quality=20
        )
        assert args_low["high_freq_quant_factor"] == pytest.approx(4.0)


class TestJpegDistortionMethods:
    """Direct tests for individual JPEG compression methods on RGB and Grayscale."""

    @pytest.mark.parametrize("subsampling", [0, 1, 2])
    def test_add_jpeg_with_subsampling_rgb(
        self, sample_rgb_image: np.ndarray, subsampling: int
    ) -> None:
        dist = JpegDistortion(image=sample_rgb_image)
        out = dist.add_jpeg_with_subsampling(quality=15, subsampling=subsampling)
        assert out.shape == sample_rgb_image.shape
        assert out.dtype == np.float32
        assert out.min() >= 0.0
        assert out.max() <= 1.0

    def test_add_jpeg_with_subsampling_gray(
        self, sample_gray_image: np.ndarray
    ) -> None:
        dist = JpegDistortion(image=sample_gray_image)
        out = dist.add_jpeg_with_subsampling(quality=10, subsampling=2)
        assert out.shape == sample_gray_image.shape
        assert out.dtype == np.float32
        assert out.min() >= 0.0
        assert out.max() <= 1.0

    def test_add_jpeg_opencv_rgb(self, sample_rgb_image: np.ndarray) -> None:
        dist = JpegDistortion(image=sample_rgb_image)
        out = dist.add_jpeg_opencv(quality=20)
        assert out.shape == sample_rgb_image.shape
        assert out.dtype == np.float32
        assert out.min() >= 0.0
        assert out.max() <= 1.0

    def test_add_jpeg_opencv_gray(self, sample_gray_image: np.ndarray) -> None:
        dist = JpegDistortion(image=sample_gray_image)
        out = dist.add_jpeg_opencv(quality=20)
        assert out.shape == sample_gray_image.shape
        assert out.dtype == np.float32

    @pytest.mark.parametrize("shift_x,shift_y", [(0, 0), (2, 5), (4, 4), (7, 7)])
    def test_add_jpeg_grid_shift_rgb(
        self, sample_rgb_image: np.ndarray, shift_x: int, shift_y: int
    ) -> None:
        dist = JpegDistortion(image=sample_rgb_image)
        out = dist.add_jpeg_grid_shift(quality=15, shift_x=shift_x, shift_y=shift_y)
        assert out.shape == sample_rgb_image.shape
        assert out.dtype == np.float32
        assert out.min() >= 0.0
        assert out.max() <= 1.0

    def test_add_jpeg_grid_shift_gray(self, sample_gray_image: np.ndarray) -> None:
        dist = JpegDistortion(image=sample_gray_image)
        out = dist.add_jpeg_grid_shift(quality=15, shift_x=3, shift_y=3)
        assert out.shape == sample_gray_image.shape
        assert out.dtype == np.float32

    def test_add_double_jpeg_compression_rgb(
        self, sample_rgb_image: np.ndarray
    ) -> None:
        dist = JpegDistortion(image=sample_rgb_image)
        out = dist.add_double_jpeg_compression(quality_1=30, quality_2=10)
        assert out.shape == sample_rgb_image.shape
        assert out.dtype == np.float32
        assert out.min() >= 0.0
        assert out.max() <= 1.0

    def test_add_double_jpeg_compression_gray(
        self, sample_gray_image: np.ndarray
    ) -> None:
        dist = JpegDistortion(image=sample_gray_image)
        out = dist.add_double_jpeg_compression(quality_1=40, quality_2=15)
        assert out.shape == sample_gray_image.shape
        assert out.dtype == np.float32

    def test_add_custom_quantized_jpeg_rgb(self, sample_rgb_image: np.ndarray) -> None:
        dist = JpegDistortion(image=sample_rgb_image)
        out = dist.add_custom_quantized_jpeg(high_freq_quant_factor=10.0)
        assert out.shape == sample_rgb_image.shape
        assert out.dtype == np.float32
        assert out.min() >= 0.0
        assert out.max() <= 1.0

    def test_add_custom_quantized_jpeg_gray(
        self, sample_gray_image: np.ndarray
    ) -> None:
        dist = JpegDistortion(image=sample_gray_image)
        out = dist.add_custom_quantized_jpeg(high_freq_quant_factor=4.0)
        assert out.shape == sample_gray_image.shape
        assert out.dtype == np.float32

    def test_add_jpeg_opencv_encode_failure(
        self, sample_rgb_image: np.ndarray, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        dist = JpegDistortion(image=sample_rgb_image)
        monkeypatch.setattr(cv2, "imencode", lambda *args, **kwargs: (False, None))
        with pytest.raises(RuntimeError, match="JPEG encoding failed"):
            dist.add_jpeg_opencv()

    def test_add_jpeg_opencv_decode_failure(
        self, sample_rgb_image: np.ndarray, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        dist = JpegDistortion(image=sample_rgb_image)
        monkeypatch.setattr(
            cv2, "imencode", lambda *args, **kwargs: (True, np.array([1, 2, 3]))
        )
        monkeypatch.setattr(cv2, "imdecode", lambda *args, **kwargs: None)
        with pytest.raises(RuntimeError, match="JPEG decoding failed"):
            dist.add_jpeg_opencv()

    def test_methods_raise_if_no_image_loaded(self) -> None:
        """Every distortion method raises ValueError when called without loaded image."""
        dist = JpegDistortion()
        for method_name in JpegDistortion.DISTORTION_METHODS:
            fn = getattr(dist, method_name)
            with pytest.raises(ValueError, match="No image loaded"):
                fn()


class TestJpegDistortionSaveOutput:
    """Tests artifact saving and lossless PNG output."""

    def test_save_output_rgb(
        self, tmp_path: Path, sample_rgb_image: np.ndarray
    ) -> None:
        dist = JpegDistortion(image=sample_rgb_image)
        out = dist.apply(quality=10)
        save_file = tmp_path / "compressed_output.png"

        saved_path = dist.save_output(out, save_file)
        assert saved_path.exists()
        assert saved_path == save_file.resolve()

        read_bgr = cv2.imread(str(saved_path))
        assert read_bgr is not None
        assert read_bgr.shape == (64, 64, 3)

    def test_save_output_gray(
        self, tmp_path: Path, sample_gray_image: np.ndarray
    ) -> None:
        dist = JpegDistortion(image=sample_gray_image)
        out = dist.apply(quality=10)
        save_file = tmp_path / "compressed_gray.png"

        saved_path = dist.save_output(out, save_file)
        assert saved_path.exists()

        read_gray = cv2.imread(str(saved_path), cv2.IMREAD_GRAYSCALE)
        assert read_gray is not None
        assert read_gray.shape == (64, 64)
