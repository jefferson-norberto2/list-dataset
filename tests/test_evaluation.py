"""Unit tests for dataset similarity evaluation module."""

from __future__ import annotations

import csv
import math
from pathlib import Path

import cv2
import numpy as np
import pytest

from dataset.analyzer.evaluation import (
    DatasetSimilarityReport,
    DistortionPair,
    LevelStatistics,
    PairSimilarityResult,
    _evaluate_single_pair,
    compute_distortion_level_statistics,
    compute_level_statistics,
    discover_distortion_pairs,
    evaluate_dataset_similarity,
    parse_distorted_filename,
)


@pytest.fixture
def synthetic_dataset(tmp_path: Path) -> tuple[Path, Path]:
    """Creates a temporary miniature dataset with 2 references and 4 distorted images."""
    ref_dir = tmp_path / "ref"
    dist_dir = tmp_path / "dist"
    ref_dir.mkdir()
    dist_dir.mkdir()

    # Create 2 reference images (64x64)
    img1 = np.full((64, 64, 3), 128, dtype=np.uint8)
    img2 = np.full((64, 64, 3), 200, dtype=np.uint8)
    cv2.imwrite(str(ref_dir / "img1.png"), img1)
    cv2.imwrite(str(ref_dir / "img2.png"), img2)

    # Distorted images
    # img1: AWGN level 1, AWGN level 2
    cv2.imwrite(
        str(dist_dir / "img1.AWGN.1.png"), np.clip(img1 + 5, 0, 255).astype(np.uint8)
    )
    cv2.imwrite(
        str(dist_dir / "img1.AWGN.2.png"), np.clip(img1 + 30, 0, 255).astype(np.uint8)
    )

    # img2: JPEG level 1, JPEG level 2 (inside subfolder)
    sub = dist_dir / "jpeg"
    sub.mkdir()
    cv2.imwrite(
        str(sub / "img2.JPEG.1.png"), np.clip(img2 + 5, 0, 255).astype(np.uint8)
    )
    cv2.imwrite(
        str(sub / "img2.JPEG.2.png"), np.clip(img2 + 30, 0, 255).astype(np.uint8)
    )

    # Add an unparseable file and a non-matching reference file
    cv2.imwrite(str(dist_dir / "unparseable.png"), img1)
    cv2.imwrite(str(dist_dir / "unknown_ref.BLUR.1.png"), img1)
    (dist_dir / "Thumbs.db").write_text("dummy")

    return ref_dir, dist_dir


class TestParseDistortedFilename:
    """Tests for parse_distorted_filename."""

    def test_standard_format(self) -> None:
        result = parse_distorted_filename("1600.JPEG.1.png")
        assert result == ("1600", "JPEG", 1)

    def test_multi_dot_stem(self) -> None:
        result = parse_distorted_filename("butter_flower.AWGN.3.png")
        assert result == ("butter_flower", "AWGN", 3)

        result_complex = parse_distorted_filename("photo.v1.0.contrast.5.png")
        assert result_complex == ("photo.v1.0", "contrast", 5)

    def test_string_level(self) -> None:
        result = parse_distorted_filename("img.blur.high.jpg")
        assert result == ("img", "blur", "high")

    def test_invalid_formats(self) -> None:
        assert parse_distorted_filename("image.png") is None
        assert parse_distorted_filename("Thumbs.db") is None
        assert parse_distorted_filename("justonepart") is None


class TestDiscoverDistortionPairs:
    """Tests for discover_distortion_pairs."""

    def test_discovers_matched_and_unmatched(
        self, synthetic_dataset: tuple[Path, Path]
    ) -> None:
        ref_dir, dist_dir = synthetic_dataset
        pairs, unmatched = discover_distortion_pairs(ref_dir, dist_dir, recursive=True)

        assert len(pairs) == 4
        assert len(unmatched) == 2  # unparseable.png and unknown_ref.BLUR.1.png

        # Verify pair details
        levels = {p.level for p in pairs}
        assert levels == {1, 2}
        distortions = {p.distortion for p in pairs}
        assert distortions == {"AWGN", "JPEG"}

    def test_non_recursive_discovery(
        self, synthetic_dataset: tuple[Path, Path]
    ) -> None:
        ref_dir, dist_dir = synthetic_dataset
        pairs, _unmatched = discover_distortion_pairs(
            ref_dir, dist_dir, recursive=False
        )
        # jpeg subfolder won't be scanned
        assert len(pairs) == 2
        assert all(p.distortion == "AWGN" for p in pairs)

    def test_missing_ref_dir_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError, match="Reference directory not found"):
            discover_distortion_pairs(tmp_path / "nonexistent", tmp_path)

    def test_missing_dist_dir_raises(self, tmp_path: Path) -> None:
        ref_dir = tmp_path / "ref"
        ref_dir.mkdir()
        with pytest.raises(FileNotFoundError, match="Distorted directory not found"):
            discover_distortion_pairs(ref_dir, tmp_path / "nonexistent")

    def test_empty_ref_dir_raises(self, tmp_path: Path) -> None:
        ref_dir = tmp_path / "empty_ref"
        ref_dir.mkdir()
        dist_dir = tmp_path / "dist"
        dist_dir.mkdir()
        with pytest.raises(ValueError, match="No valid reference images found"):
            discover_distortion_pairs(ref_dir, dist_dir)


class TestSinglePairEvaluation:
    """Tests for _evaluate_single_pair worker."""

    def test_successful_evaluation(self, synthetic_dataset: tuple[Path, Path]) -> None:
        ref_dir, dist_dir = synthetic_dataset
        pair = DistortionPair(
            ref_path=ref_dir / "img1.png",
            dist_path=dist_dir / "img1.AWGN.1.png",
            ref_stem="img1",
            distortion="AWGN",
            level=1,
        )
        res = _evaluate_single_pair(pair)
        assert res.ref_stem == "img1"
        assert res.distortion == "AWGN"
        assert res.level == 1
        assert 0.8 < res.ssim <= 1.0
        assert pytest.approx(res.difference + res.ssim, 1e-4) == 1.0

    def test_corrupted_image_handling(self, tmp_path: Path) -> None:
        bad_ref = tmp_path / "bad_ref.png"
        bad_dist = tmp_path / "bad_dist.png"
        bad_ref.write_text("corrupted content")
        bad_dist.write_text("corrupted content")

        pair = DistortionPair(
            ref_path=bad_ref,
            dist_path=bad_dist,
            ref_stem="bad",
            distortion="ERR",
            level=1,
        )
        res = _evaluate_single_pair(pair)
        assert math.isnan(res.ssim)
        assert math.isnan(res.difference)


class TestStatisticsComputation:
    """Tests for compute_level_statistics and compute_distortion_level_statistics."""

    def test_compute_level_statistics(self, tmp_path: Path) -> None:
        dummy_path = tmp_path / "dummy.png"
        results = [
            PairSimilarityResult("img1", dummy_path, dummy_path, "D1", 1, 0.95, 0.05),
            PairSimilarityResult("img2", dummy_path, dummy_path, "D2", 1, 0.85, 0.15),
            PairSimilarityResult("img1", dummy_path, dummy_path, "D1", 2, 0.60, 0.40),
            PairSimilarityResult("img2", dummy_path, dummy_path, "D2", 2, 0.50, 0.50),
        ]

        stats = compute_level_statistics(results)
        assert 1 in stats and 2 in stats

        stat1 = stats[1]
        assert stat1.count == 2
        assert stat1.mean_ssim == pytest.approx(0.90)
        assert stat1.median_ssim == pytest.approx(0.90)
        assert stat1.mean_difference == pytest.approx(0.10)
        assert stat1.median_difference == pytest.approx(0.10)
        assert stat1.min_ssim == pytest.approx(0.85)
        assert stat1.max_ssim == pytest.approx(0.95)

    def test_empty_and_nan_statistics(self, tmp_path: Path) -> None:
        dummy_path = tmp_path / "dummy.png"
        results = [
            PairSimilarityResult(
                "img1", dummy_path, dummy_path, "D1", 1, float("nan"), float("nan")
            ),
        ]
        stats = compute_level_statistics(results)
        stat1 = stats[1]
        assert stat1.count == 1
        assert math.isnan(stat1.mean_ssim)
        assert math.isnan(stat1.median_ssim)

    def test_distortion_level_statistics(self, tmp_path: Path) -> None:
        dummy_path = tmp_path / "dummy.png"
        results = [
            PairSimilarityResult("img1", dummy_path, dummy_path, "AWGN", 1, 0.90, 0.10),
            PairSimilarityResult("img2", dummy_path, dummy_path, "JPEG", 1, 0.95, 0.05),
        ]
        dist_stats = compute_distortion_level_statistics(results)
        assert ("AWGN", 1) in dist_stats
        assert ("JPEG", 1) in dist_stats
        assert dist_stats[("AWGN", 1)].mean_ssim == pytest.approx(0.90)
        assert dist_stats[("JPEG", 1)].mean_ssim == pytest.approx(0.95)

    def test_distortion_level_statistics_nan(self, tmp_path: Path) -> None:
        dummy_path = tmp_path / "dummy.png"
        results = [
            PairSimilarityResult(
                "img1", dummy_path, dummy_path, "AWGN", 1, float("nan"), float("nan")
            ),
        ]
        dist_stats = compute_distortion_level_statistics(results)
        assert math.isnan(dist_stats[("AWGN", 1)].mean_ssim)


class TestDatasetSimilarityReport:
    """Tests for DatasetSimilarityReport methods."""

    def test_format_summary_and_save_csv(self, tmp_path: Path) -> None:
        p1 = tmp_path / "p1.png"
        p2 = tmp_path / "p2.png"
        results = [
            PairSimilarityResult("img1", p1, p2, "AWGN", 1, 0.95, 0.05),
            PairSimilarityResult("img1", p1, p2, "AWGN", 2, 0.80, 0.20),
        ]
        level_stats = {
            1: LevelStatistics(1, 1, 0.95, 0.95, 0.0, 0.95, 0.95, 0.05, 0.05),
            2: LevelStatistics(2, 1, 0.80, 0.80, 0.0, 0.80, 0.80, 0.20, 0.20),
        }
        dist_stats = {
            ("AWGN", 1): LevelStatistics(1, 1, 0.95, 0.95, 0.0, 0.95, 0.95, 0.05, 0.05),
            ("AWGN", 2): LevelStatistics(2, 1, 0.80, 0.80, 0.0, 0.80, 0.80, 0.20, 0.20),
        }

        report = DatasetSimilarityReport(
            results=results,
            level_stats=level_stats,
            distortion_level_stats=dist_stats,
            unmatched_files=[tmp_path / "orphaned.png"],
        )

        summary_text = report.format_summary()
        assert "RESUMO DE SIMILARIDADE POR NÍVEL DE DISTORÇÃO" in summary_text
        assert "Nível 1" in summary_text
        assert "Nível 2" in summary_text
        assert "DETALHAMENTO POR DISTORÇÃO E NÍVEL" in summary_text
        assert "1 arquivos distorcidos não puderam ser pareados" in summary_text

        # Test save_csv
        csv_file = tmp_path / "report.csv"
        report.save_csv(csv_file)
        assert csv_file.is_file()

        with csv_file.open("r", encoding="utf-8") as f:
            reader = list(csv.DictReader(f))
            assert len(reader) == 2
            assert reader[0]["reference_stem"] == "img1"
            assert reader[0]["level"] == "1"
            assert reader[0]["ssim"] == "0.950000"


class TestEvaluateDatasetSimilarity:
    """Integration tests for evaluate_dataset_similarity."""

    def test_evaluate_serial(self, synthetic_dataset: tuple[Path, Path]) -> None:
        ref_dir, dist_dir = synthetic_dataset
        report = evaluate_dataset_similarity(
            ref_dir,
            dist_dir,
            max_workers=1,
            show_progress=False,
        )
        assert len(report.results) == 4
        assert len(report.level_stats) == 2
        assert report.level_stats[1].mean_ssim > report.level_stats[2].mean_ssim

    def test_evaluate_parallel(self, synthetic_dataset: tuple[Path, Path]) -> None:
        ref_dir, dist_dir = synthetic_dataset
        report = evaluate_dataset_similarity(
            ref_dir,
            dist_dir,
            max_workers=2,
            show_progress=True,
        )
        assert len(report.results) == 4
        assert len(report.level_stats) == 2

    def test_evaluate_empty_pairs(self, tmp_path: Path) -> None:
        ref_dir = tmp_path / "ref"
        dist_dir = tmp_path / "dist"
        ref_dir.mkdir()
        dist_dir.mkdir()

        # Valid image in ref
        img = np.zeros((16, 16, 3), dtype=np.uint8)
        cv2.imwrite(str(ref_dir / "ref.png"), img)

        report = evaluate_dataset_similarity(ref_dir, dist_dir, show_progress=False)
        assert len(report.results) == 0
        assert len(report.level_stats) == 0


class TestMain2Script:
    """Tests for main2.py CLI script."""

    def test_main2_execution(
        self,
        synthetic_dataset: tuple[Path, Path],
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        import runpy
        import sys

        ref_dir, dist_dir = synthetic_dataset
        csv_out = tmp_path / "output.csv"

        monkeypatch.setattr(
            sys,
            "argv",
            [
                "main2.py",
                "--ref-dir",
                str(ref_dir),
                "--dist-dir",
                str(dist_dir),
                "--csv",
                str(csv_out),
                "--workers",
                "1",
            ],
        )

        with pytest.raises(SystemExit) as exc_info:
            runpy.run_path("main2.py", run_name="__main__")

        assert exc_info.value.code == 0
        assert csv_out.is_file()
