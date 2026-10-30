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


@pytest.fixture
def sample_rgb_array() -> np.ndarray:
    """Creates a deterministic 64x64 RGB array."""
    rng = np.random.default_rng(42)
    return rng.integers(0, 256, (64, 64, 3), dtype=np.uint8)


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
            gen.apply_blur()

        with pytest.raises(ValueError, match="No image loaded"):
            gen.apply_poisson()

        with pytest.raises(ValueError, match="No image loaded"):
            gen.apply_jpeg()

        with pytest.raises(ValueError, match="No image loaded"):
            gen.apply_jpeg2000()

        with pytest.raises(ValueError, match="No image loaded"):
            gen.apply_contrast()

        with pytest.raises(ValueError, match="No image loaded"):
            gen.apply_fnoise()

        with pytest.raises(ValueError, match="No image loaded"):
            gen.save_output(np.zeros((10, 10, 3), dtype=np.float32), "blur", 1)

    def test_init_with_image(
        self, sample_image_path: Path, sample_rgb_array: np.ndarray
    ) -> None:
        """Initialization with str, Path, or ndarray properly configures the generator."""
        gen_path = DistortionGenerator(image=sample_image_path)
        assert gen_path.img is not None
        assert gen_path.img_name == "sample"

        gen_arr = DistortionGenerator(image=sample_rgb_array)
        assert gen_arr.img is not None
        assert gen_arr.height == 64
        assert gen_arr.width == 64

        with pytest.raises(TypeError, match="Unsupported image type"):
            DistortionGenerator(image=12345)  # type: ignore[arg-type]

    def test_set_image_validation(self) -> None:
        """Empty or invalid dimension arrays raise ValueError."""
        gen = DistortionGenerator()
        with pytest.raises(ValueError, match="Invalid image array"):
            gen.set_image(np.array([]))

        with pytest.raises(ValueError, match="Expected 2D or 3D"):
            gen.set_image(np.zeros((10, 10, 3, 2), dtype=np.uint8))

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
        """Verifies JPEG 2000 compression and output range."""
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

    def test_apply_pipelines(self, sample_image_path: Path) -> None:
        """Verifies apply_* delegation pipelines."""
        gen = DistortionGenerator(image=sample_image_path)

        b = gen.apply_blur(sigma=1.5)
        assert b.shape == (100, 100, 3)

        p = gen.apply_poisson(shot_noise=0.01)
        assert p.shape == (100, 100, 3)

        j = gen.apply_jpeg(quality=20)
        assert j.shape == (100, 100, 3)

        jp2 = gen.apply_jpeg2000(compression_rate=15.0)
        assert jp2.shape == (100, 100, 3)

        c = gen.apply_contrast(alpha=0.4)
        assert c.shape == (100, 100, 3)

        fn = gen.apply_fnoise(intensity=0.08)
        assert fn.shape == (100, 100, 3)

    def test_save_output_default_and_custom(
        self, sample_image_path: Path, tmp_path: Path
    ) -> None:
        """Verifies save_output generates valid PNG file with default and custom templates."""
        gen = DistortionGenerator()
        gen.change_image(str(sample_image_path))

        blurred = gen.add_gaussian_blur(sigma=1.5)

        # 1. Default legacy template
        gen.save_output(blurred, "blur", 1)
        expected_default = (
            sample_image_path.parent.parent / "blur" / "dist_1_sample.png"
        )
        assert expected_default.exists()

        # 2. Custom directory and template
        custom_dir = tmp_path / "custom_dst"
        custom_saved = gen.save_output(
            blurred,
            "blur",
            2,
            output_dir=custom_dir,
            filename_format="{img_name}.{distortion}.{level}.png",
        )
        assert custom_saved.exists()
        assert custom_saved.name == "sample.blur.2.png"

    def test_generate_all_distortions_for_image(
        self, sample_image_path: Path, tmp_path: Path
    ) -> None:
        """Verifies generate_all_distortions_for_image creates 5 levels for selected distortions."""
        gen = DistortionGenerator()
        out_root = tmp_path / "all_dist_test"

        results = gen.generate_all_distortions_for_image(
            image_path=sample_image_path,
            output_root=out_root,
            distortions=["blur", "jpeg"],
            filename_format="{img_name}.{distortion}.{level}.png",
            use_apply=False,
        )

        assert "blur" in results
        assert "jpeg" in results
        assert len(results["blur"]) == 5
        assert len(results["jpeg"]) == 5

        # Check files exist
        for lvl in range(1, 6):
            expected = out_root / "blur" / f"sample.blur.{lvl}.png"
            assert expected.exists()

    def test_process_folder(self, tmp_path: Path) -> None:
        """Verifies process_folder scans directory, applies limits, and generates subfolders."""
        src_dir = tmp_path / "src_test"
        src_dir.mkdir(parents=True)
        out_root = tmp_path / "dst_test"

        # Create 2 small test images
        img1 = np.ones((32, 32, 3), dtype=np.uint8) * 100
        img2 = np.ones((32, 32, 3), dtype=np.uint8) * 200
        cv2.imwrite(str(src_dir / "img1.png"), img1)
        cv2.imwrite(str(src_dir / "img2.png"), img2)

        gen = DistortionGenerator()

        # Non-existent folder raises
        with pytest.raises(FileNotFoundError, match="Source directory not found"):
            gen.process_folder(tmp_path / "does_not_exist")

        # Process with limit=1 and distortions=['blur', 'contrast']
        counts = gen.process_folder(
            folder_path=src_dir,
            output_root=out_root,
            distortions=["blur", "contrast"],
            limit=1,
            show_progress=False,
        )

        assert counts["blur"] == 5
        assert counts["contrast"] == 5
        assert (out_root / "blur" / "img1.blur.1.png").exists()
        # img2 was excluded by limit=1
        assert not (out_root / "blur" / "img2.blur.1.png").exists()

    def test_main_cli(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """Tests the main() CLI entry point."""
        from dataset.distortion.generator import main

        src_dir = tmp_path / "cli_src"
        src_dir.mkdir()
        out_root = tmp_path / "cli_dst"
        img = np.ones((32, 32, 3), dtype=np.uint8) * 128
        cv2.imwrite(str(src_dir / "cli_sample.png"), img)

        ret = main(
            [
                "--input-dir",
                str(src_dir),
                "--output-dir",
                str(out_root),
                "--distortions",
                "blur",
                "--limit",
                "1",
            ]
        )
        assert ret == 0
        assert (out_root / "blur" / "cli_sample.blur.1.png").exists()

        # Non-existent input dir returns 1
        ret_err = main(["--input-dir", str(tmp_path / "not_there")])
        assert ret_err == 1
