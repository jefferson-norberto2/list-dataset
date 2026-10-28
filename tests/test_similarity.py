"""Unit and integration tests for SSIM similarity module and CLI."""

from __future__ import annotations

import json
import runpy
import subprocess
import sys
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np
import pytest

from dataset_curator.curator.preprocessing import ImageLoadingError
from dataset_curator.analyse.similarity import (
    SimilarityResult,
    check_image_similarity,
    compare_images_ssim,
    compute_ssim,
    main_cli,
)


@pytest.fixture
def sample_rgb_image() -> np.ndarray:
    """Creates a deterministic 64x64 RGB synthetic image."""
    rng = np.random.default_rng(seed=42)
    return rng.integers(0, 256, size=(64, 64, 3), dtype=np.uint8)


@pytest.fixture
def sample_gray_image() -> np.ndarray:
    """Creates a deterministic 64x64 Grayscale synthetic image."""
    rng = np.random.default_rng(seed=42)
    return rng.integers(0, 256, size=(64, 64), dtype=np.uint8)


@pytest.fixture
def saved_image_pair(tmp_path: Path, sample_rgb_image: np.ndarray) -> tuple[Path, Path]:
    """Saves two identical images to disk and returns their paths."""
    p1 = tmp_path / "img1.png"
    p2 = tmp_path / "img2.png"
    cv2.imwrite(str(p1), cv2.cvtColor(sample_rgb_image, cv2.COLOR_RGB2BGR))
    cv2.imwrite(str(p2), cv2.cvtColor(sample_rgb_image, cv2.COLOR_RGB2BGR))
    return p1, p2


class TestComputeSsim:
    """Tests for compute_ssim function."""

    def test_identical_images_return_one(self, sample_rgb_image: np.ndarray) -> None:
        """Identical images must yield an SSIM score of 1.0."""
        score = compute_ssim(sample_rgb_image, sample_rgb_image.copy())
        assert score == pytest.approx(1.0, rel=1e-4)

    def test_slightly_altered_images(self, sample_rgb_image: np.ndarray) -> None:
        """Slightly modified images should have high score strictly between 0 and 1."""
        noisy = sample_rgb_image.copy().astype(np.int16)
        noisy = np.clip(noisy + 5, 0, 255).astype(np.uint8)
        score = compute_ssim(sample_rgb_image, noisy)
        assert 0.8 < score < 1.0

    def test_distinct_images_return_low_score(self) -> None:
        """Distinct images (e.g. constant black vs white) yield low SSIM."""
        black = np.zeros((64, 64, 3), dtype=np.uint8)
        white = np.full((64, 64, 3), 255, dtype=np.uint8)
        score = compute_ssim(black, white)
        assert score < 0.1

    def test_grayscale_2d_arrays(self, sample_gray_image: np.ndarray) -> None:
        """SSIM computation works on 2D grayscale arrays."""
        score_identical = compute_ssim(sample_gray_image, sample_gray_image.copy())
        assert score_identical == pytest.approx(1.0, rel=1e-4)

        noisy = np.clip(sample_gray_image.astype(np.int16) + 10, 0, 255).astype(
            np.uint8
        )
        score_noisy = compute_ssim(sample_gray_image, noisy)
        assert 0.0 < score_noisy < 1.0

    def test_shape_mismatch_without_auto_resize_raises(
        self, sample_rgb_image: np.ndarray
    ) -> None:
        """Shape mismatch without auto_resize raises ValueError."""
        other = cv2.resize(sample_rgb_image, (32, 32))
        with pytest.raises(ValueError, match="Image shapes do not match"):
            compute_ssim(sample_rgb_image, other, auto_resize=False)

    def test_shape_mismatch_with_auto_resize(self) -> None:
        """Shape mismatch with auto_resize resizes and computes score."""
        gradient = np.tile(np.linspace(0, 255, 64, dtype=np.uint8), (64, 1))
        img1 = np.stack([gradient] * 3, axis=-1)
        img2 = cv2.resize(img1, (32, 32))
        score = compute_ssim(img1, img2, auto_resize=True)
        assert isinstance(score, float)
        assert score > 0.9

    def test_shape_mismatch_2d_and_3d_with_auto_resize(
        self, sample_rgb_image: np.ndarray, sample_gray_image: np.ndarray
    ) -> None:
        """Auto resize handles channel conversion between 2D and 3D."""
        score_2d_3d = compute_ssim(
            sample_gray_image, sample_rgb_image, auto_resize=True
        )
        assert isinstance(score_2d_3d, float)
        assert not np.isnan(score_2d_3d)

        score_3d_2d = compute_ssim(
            sample_rgb_image, sample_gray_image, auto_resize=True
        )
        assert isinstance(score_3d_2d, float)
        assert not np.isnan(score_3d_2d)

    def test_images_smaller_than_3x3_return_nan(self) -> None:
        """Images smaller than 3x3 return float('nan')."""
        tiny1 = np.ones((2, 2, 3), dtype=np.uint8)
        tiny2 = np.ones((2, 2, 3), dtype=np.uint8)
        score = compute_ssim(tiny1, tiny2)
        assert np.isnan(score)

        tiny_w = np.ones((10, 2), dtype=np.uint8)
        score_w = compute_ssim(tiny_w, tiny_w)
        assert np.isnan(score_w)

    def test_invalid_input_types_and_dimensions(self) -> None:
        """1D arrays or non-arrays return NaN."""
        one_d = np.ones((10,), dtype=np.uint8)
        assert np.isnan(compute_ssim(one_d, one_d))
        assert np.isnan(compute_ssim("not_an_array", one_d))  # type: ignore[arg-type]

    def test_nan_or_corrupt_data_handled_gracefully(self) -> None:
        """Arrays containing NaN values are handled without crashing."""
        nan_img1 = np.full((32, 32, 3), np.nan, dtype=np.float32)
        nan_img2 = np.ones((32, 32, 3), dtype=np.float32)
        score = compute_ssim(nan_img1, nan_img2)
        assert np.isnan(score)

    def test_floating_point_images(self) -> None:
        """Float images with dynamic range computation."""
        flt1 = np.full((32, 32), 0.5, dtype=np.float32)
        flt2 = np.full((32, 32), 0.5, dtype=np.float32)
        score = compute_ssim(flt1, flt2)
        assert score == pytest.approx(1.0, rel=1e-4)

    def test_win_size_options(self, sample_rgb_image: np.ndarray) -> None:
        """Custom win_size options: odd, even, larger than image, < 3."""
        score_odd = compute_ssim(sample_rgb_image, sample_rgb_image, win_size=5)
        assert score_odd == pytest.approx(1.0, rel=1e-4)

        score_even = compute_ssim(sample_rgb_image, sample_rgb_image, win_size=6)
        assert score_even == pytest.approx(1.0, rel=1e-4)

        score_oversized = compute_ssim(sample_rgb_image, sample_rgb_image, win_size=100)
        assert score_oversized == pytest.approx(1.0, rel=1e-4)

        score_too_small = compute_ssim(sample_rgb_image, sample_rgb_image, win_size=2)
        assert np.isnan(score_too_small)

    def test_even_dimension_default_win_size(self) -> None:
        """When min_dim is even (e.g. 4x4), default win_size is adjusted to odd."""
        img = np.ones((4, 4), dtype=np.uint8)
        score = compute_ssim(img, img)
        assert score == pytest.approx(1.0, rel=1e-4)

    def test_skimage_returns_nan_or_raises(self, sample_rgb_image: np.ndarray) -> None:
        """Handled if structural_similarity returns NaN or raises Exception."""
        with patch(
            "dataset_curator.similarity.structural_similarity",
            return_value=float("nan"),
        ):
            assert np.isnan(compute_ssim(sample_rgb_image, sample_rgb_image))

        with patch(
            "dataset_curator.similarity.structural_similarity",
            side_effect=RuntimeError("internal error"),
        ):
            assert np.isnan(compute_ssim(sample_rgb_image, sample_rgb_image))


class TestCompareImagesSsim:
    """Tests for compare_images_ssim."""

    def test_compare_with_paths(self, saved_image_pair: tuple[Path, Path]) -> None:
        """Paths to valid images compute correct SSIM."""
        p1, p2 = saved_image_pair
        score = compare_images_ssim(p1, p2)
        assert score == pytest.approx(1.0, rel=1e-4)

    def test_compare_with_strings(self, saved_image_pair: tuple[Path, Path]) -> None:
        """String paths compute correct SSIM."""
        p1, p2 = saved_image_pair
        score = compare_images_ssim(str(p1), str(p2))
        assert score == pytest.approx(1.0, rel=1e-4)

    def test_compare_with_numpy_arrays(self, sample_rgb_image: np.ndarray) -> None:
        """Numpy arrays compute correct SSIM."""
        score = compare_images_ssim(sample_rgb_image, sample_rgb_image.copy())
        assert score == pytest.approx(1.0, rel=1e-4)

    def test_compare_mixed_types(
        self, saved_image_pair: tuple[Path, Path], sample_rgb_image: np.ndarray
    ) -> None:
        """Mixing Path and numpy array works."""
        p1, _ = saved_image_pair
        score = compare_images_ssim(p1, sample_rgb_image)
        assert score == pytest.approx(1.0, rel=1e-4)

    def test_invalid_types_raise_type_error(self) -> None:
        """Invalid types for img1 or img2 raise TypeError."""
        with pytest.raises(TypeError, match="Unsupported type for img1"):
            compare_images_ssim(12345, "img2.png")  # type: ignore[arg-type]

        with pytest.raises(TypeError, match="Unsupported type for img2"):
            compare_images_ssim("img1.png", [1, 2, 3])  # type: ignore[arg-type]

    def test_nonexistent_file_raises_filenotfound(self, tmp_path: Path) -> None:
        """Non-existent file path raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            compare_images_ssim(tmp_path / "missing1.png", tmp_path / "missing2.png")

    def test_corrupt_file_raises_imageloadingerror(self, tmp_path: Path) -> None:
        """Empty or corrupt file raises ImageLoadingError."""
        corrupt = tmp_path / "corrupt.png"
        corrupt.write_bytes(b"not an image")
        with pytest.raises(ImageLoadingError):
            compare_images_ssim(corrupt, corrupt)


class TestCheckImageSimilarity:
    """Tests for check_image_similarity and SimilarityResult."""

    def test_similar_images_pass(self, saved_image_pair: tuple[Path, Path]) -> None:
        """Identical/similar images pass similarity threshold."""
        p1, p2 = saved_image_pair
        res = check_image_similarity(p1, p2, threshold=0.8)
        assert isinstance(res, SimilarityResult)
        assert res.is_similar is True
        assert res.score == pytest.approx(1.0, rel=1e-4)
        assert res.threshold == 0.8
        assert res.img1 == str(p1)
        assert res.img2 == str(p2)

    def test_dissimilar_images_fail(self, sample_rgb_image: np.ndarray) -> None:
        """Very different images fail similarity threshold."""
        black = np.zeros_like(sample_rgb_image)
        res = check_image_similarity(sample_rgb_image, black, threshold=0.8)
        assert res.is_similar is False
        assert res.score < 0.5
        assert res.threshold == 0.8

    def test_threshold_edge_conditions(self, sample_rgb_image: np.ndarray) -> None:
        """Edge conditions for threshold comparison."""
        # Score is 1.0; threshold 1.0 should pass
        res_exact = check_image_similarity(
            sample_rgb_image, sample_rgb_image, threshold=1.0
        )
        assert res_exact.is_similar is True

        # Threshold slightly above 1.0 fails
        res_above = check_image_similarity(
            sample_rgb_image, sample_rgb_image, threshold=1.0001
        )
        assert res_above.is_similar is False

    def test_nan_score_marks_dissimilar(self) -> None:
        """If SSIM computation returns NaN, is_similar must be False."""
        tiny = np.zeros((2, 2, 3), dtype=np.uint8)
        res = check_image_similarity(tiny, tiny, threshold=0.5)
        assert np.isnan(res.score)
        assert res.is_similar is False


class TestMainCli:
    """Tests for main_cli entry point."""

    def test_cli_similar_images_exit_0(
        self, saved_image_pair: tuple[Path, Path], capsys: pytest.CaptureFixture[str]
    ) -> None:
        """CLI returns exit code 0 and prints score for similar images."""
        p1, p2 = saved_image_pair
        exit_code = main_cli([str(p1), str(p2), "--threshold", "0.8"])
        captured = capsys.readouterr()
        assert exit_code == 0
        assert "SIMILAR: SSIM score" in captured.out
        assert ">= threshold 0.8000" in captured.out

    def test_cli_dissimilar_images_exit_1(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """CLI returns exit code 1 for dissimilar images."""
        p1 = tmp_path / "black.png"
        p2 = tmp_path / "white.png"
        cv2.imwrite(str(p1), np.zeros((32, 32, 3), dtype=np.uint8))
        cv2.imwrite(str(p2), np.full((32, 32, 3), 255, dtype=np.uint8))

        exit_code = main_cli([str(p1), str(p2), "--threshold", "0.8"])
        captured = capsys.readouterr()
        assert exit_code == 1
        assert "DISSIMILAR: SSIM score" in captured.out

    def test_cli_json_output_similar(
        self, saved_image_pair: tuple[Path, Path], capsys: pytest.CaptureFixture[str]
    ) -> None:
        """CLI outputs structured JSON with expected keys when --json is passed."""
        p1, p2 = saved_image_pair
        exit_code = main_cli([str(p1), str(p2), "--json"])
        captured = capsys.readouterr()
        assert exit_code == 0
        data = json.loads(captured.out)
        assert data["is_similar"] is True
        assert data["score"] == pytest.approx(1.0, rel=1e-4)
        assert data["threshold"] == 0.8
        assert data["image1"] == str(p1)
        assert data["image2"] == str(p2)

    def test_cli_json_output_dissimilar(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """CLI outputs valid JSON with is_similar False for dissimilar images."""
        p1 = tmp_path / "black.png"
        p2 = tmp_path / "white.png"
        cv2.imwrite(str(p1), np.zeros((32, 32, 3), dtype=np.uint8))
        cv2.imwrite(str(p2), np.full((32, 32, 3), 255, dtype=np.uint8))

        exit_code = main_cli([str(p1), str(p2), "--json", "--threshold", "0.9"])
        captured = capsys.readouterr()
        assert exit_code == 1
        data = json.loads(captured.out)
        assert data["is_similar"] is False
        assert data["score"] < 0.2

    def test_cli_auto_resize_flag(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """CLI auto-resize flag resizes image2 to match image1."""
        p1 = tmp_path / "im1.png"
        p2 = tmp_path / "im2.png"
        img1 = np.full((64, 64, 3), 150, dtype=np.uint8)
        img2 = np.full((32, 32, 3), 150, dtype=np.uint8)
        cv2.imwrite(str(p1), img1)
        cv2.imwrite(str(p2), img2)

        exit_code = main_cli([str(p1), str(p2), "--auto-resize"])
        captured = capsys.readouterr()
        assert exit_code == 0
        assert "SIMILAR" in captured.out

    def test_cli_invalid_path_error(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """CLI reports clean error message on stderr and exits 1 for non-existent files."""
        p1 = tmp_path / "nonexistent1.png"
        p2 = tmp_path / "nonexistent2.png"
        exit_code = main_cli([str(p1), str(p2)])
        captured = capsys.readouterr()
        assert exit_code == 1
        assert "Error: Image file does not exist" in captured.err

    def test_cli_invalid_path_json_error(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """CLI outputs valid JSON on error when --json is requested."""
        p1 = tmp_path / "nonexistent1.png"
        p2 = tmp_path / "nonexistent2.png"
        exit_code = main_cli([str(p1), str(p2), "--json"])
        captured = capsys.readouterr()
        assert exit_code == 1
        data = json.loads(captured.out)
        assert "error" in data
        assert data["is_similar"] is False

    def test_cli_nan_score(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """CLI handles NaN scores cleanly and prints DISSIMILAR message."""
        p1 = tmp_path / "tiny1.png"
        p2 = tmp_path / "tiny2.png"
        cv2.imwrite(str(p1), np.ones((2, 2, 3), dtype=np.uint8))
        cv2.imwrite(str(p2), np.ones((2, 2, 3), dtype=np.uint8))

        exit_code = main_cli([str(p1), str(p2)])
        captured = capsys.readouterr()
        assert exit_code == 1
        assert "DISSIMILAR: SSIM could not be computed (NaN)" in captured.out

    def test_module_execution(self, saved_image_pair: tuple[Path, Path]) -> None:
        """Executing dataset_curator.similarity as main module works."""
        p1, p2 = saved_image_pair
        cmd = [
            sys.executable,
            "-m",
            "dataset_curator.similarity",
            str(p1),
            str(p2),
        ]
        result = subprocess.run(cmd, check=False, capture_output=True, text=True)
        assert result.returncode == 0
        assert "SIMILAR" in result.stdout

    def test_run_as_main_module_in_process(
        self, saved_image_pair: tuple[Path, Path], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Directly running __main__ within process triggers sys.exit(0)."""
        p1, p2 = saved_image_pair
        monkeypatch.setattr(
            sys,
            "argv",
            ["dataset_curator.similarity", str(p1), str(p2)],
        )
        monkeypatch.delitem(sys.modules, "dataset_curator.similarity", raising=False)
        with pytest.raises(SystemExit) as exc_info:
            runpy.run_module("dataset_curator.similarity", run_name="__main__")
        assert exc_info.value.code == 0
