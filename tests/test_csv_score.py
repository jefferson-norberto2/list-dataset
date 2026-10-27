"""Unit tests for external CSV score loading and filtering stage."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

from dataset_curator.metrics.base import QualityFilterStage
from dataset_curator.metrics.csv_score import (
    CSVScoreFilterStage,
    evaluate_csv_score,
    load_csv_scores,
)


class TestLoadCSVScores:
    """Tests for loading and parsing image quality scores from CSV files."""

    def test_load_csv_scores_success_with_various_image_columns(
        self, tmp_path: Path
    ) -> None:
        """Loads scores correctly when image column is inferred from standard candidates."""
        csv_file = tmp_path / "scores.csv"
        csv_file.write_text(
            "filename,quality_score\nimg1.png,0.85\nsubdir/img2.jpg,0.65\nimg3,0.92\n",
            encoding="utf-8",
        )

        scores = load_csv_scores(csv_file, score_column="quality_score")

        # Verify img1.png is accessible via filename, stem, etc.
        assert scores["img1.png"] == pytest.approx(0.85)
        assert scores["img1"] == pytest.approx(0.85)

        # Verify subdir/img2.jpg is accessible via full relative path, filename, and stem
        assert scores["subdir/img2.jpg"] == pytest.approx(0.65)
        assert scores["img2.jpg"] == pytest.approx(0.65)
        assert scores["img2"] == pytest.approx(0.65)

        assert scores["img3"] == pytest.approx(0.92)

    def test_load_csv_scores_explicit_image_column(self, tmp_path: Path) -> None:
        """Loads scores when image_column is explicitly specified."""
        csv_file = tmp_path / "custom_scores.csv"
        csv_file.write_text(
            "custom_id,score_val,extra_col\n"
            "photo_01.png,0.78,foo\n"
            "photo_02.png,0.45,bar\n",
            encoding="utf-8",
        )

        scores = load_csv_scores(
            csv_file, score_column="score_val", image_column="custom_id"
        )
        assert scores["photo_01.png"] == pytest.approx(0.78)
        assert scores["photo_02"] == pytest.approx(0.45)

    def test_load_csv_scores_fallback_to_first_non_score_column(
        self, tmp_path: Path
    ) -> None:
        """Falls back to the first non-score column when no standard image column matches."""
        csv_file = tmp_path / "fallback.csv"
        csv_file.write_text(
            "identifier,rating\nsample_a.jpg,0.91\n",
            encoding="utf-8",
        )

        scores = load_csv_scores(csv_file, score_column="rating")
        assert scores["sample_a.jpg"] == pytest.approx(0.91)
        assert scores["sample_a"] == pytest.approx(0.91)

    def test_load_csv_scores_first_column_fallback_when_only_one_col(
        self, tmp_path: Path
    ) -> None:
        """Falls back gracefully if the only column is score_column."""
        csv_file = tmp_path / "single_col.csv"
        csv_file.write_text("rating\n0.91\n", encoding="utf-8")
        scores = load_csv_scores(csv_file, score_column="rating")
        # rating is used as img_col fallback and parsed
        assert len(scores) > 0

    def test_load_csv_scores_case_insensitive_header_matching(
        self, tmp_path: Path
    ) -> None:
        """Matches column names case-insensitively."""
        csv_file = tmp_path / "case_test.csv"
        csv_file.write_text(
            "IMAGE_NAME,QUALITY_SCORE\ncase_img.png,0.82\n",
            encoding="utf-8",
        )

        scores = load_csv_scores(
            csv_file, score_column="quality_score", image_column="image_name"
        )
        assert scores["case_img.png"] == pytest.approx(0.82)

    def test_load_csv_scores_auto_detect_uppercase_candidate(
        self, tmp_path: Path
    ) -> None:
        """Auto-detects uppercase candidate column without explicit image_column."""
        csv_file = tmp_path / "upper_case.csv"
        csv_file.write_text("FILENAME,SCORE\nup_img.png,0.77\n", encoding="utf-8")
        scores = load_csv_scores(csv_file, score_column="SCORE")
        assert scores["up_img.png"] == pytest.approx(0.77)

    def test_load_csv_scores_invalid_or_nan_scores(self, tmp_path: Path) -> None:
        """Stores float('nan') when score is empty, non-numeric, or invalid."""
        csv_file = tmp_path / "nans.csv"
        csv_file.write_text(
            "filename,score\n"
            "img_valid.png,0.75\n"
            "img_empty.png,\n"
            "img_text.png,invalid_num\n"
            " ,0.99\n",  # empty filename should be skipped
            encoding="utf-8",
        )

        scores = load_csv_scores(csv_file, score_column="score")
        assert scores["img_valid.png"] == pytest.approx(0.75)
        assert math.isnan(scores["img_empty.png"])
        assert math.isnan(scores["img_text.png"])

    def test_load_csv_scores_file_not_found(self, tmp_path: Path) -> None:
        """Raises FileNotFoundError if file does not exist."""
        with pytest.raises(FileNotFoundError, match="Score CSV file not found"):
            load_csv_scores(tmp_path / "non_existent.csv", score_column="score")

    def test_load_csv_scores_spaces_in_headers(self, tmp_path: Path) -> None:
        """Properly handles spaces in CSV headers and around delimiters."""
        csv_file = tmp_path / "spaced_headers.csv"
        csv_file.write_text(
            " image_name , quality_score \nimg_spaced.png, 0.88\n",
            encoding="utf-8",
        )
        scores = load_csv_scores(csv_file, score_column="quality_score")
        assert scores["img_spaced.png"] == pytest.approx(0.88)
        assert scores["img_spaced"] == pytest.approx(0.88)

    def test_load_csv_scores_windows_backslash_paths(self, tmp_path: Path) -> None:
        """Normalizes Windows backslash paths for cross-platform lookup."""
        csv_file = tmp_path / "win_paths.csv"
        csv_file.write_text(
            "image,score\nsubdir\\win_img.png,0.76\n",
            encoding="utf-8",
        )
        scores = load_csv_scores(csv_file, score_column="score")
        assert scores["subdir\\win_img.png"] == pytest.approx(0.76)
        assert scores["subdir/win_img.png"] == pytest.approx(0.76)
        assert scores["win_img.png"] == pytest.approx(0.76)
        assert scores["win_img"] == pytest.approx(0.76)

    def test_load_csv_scores_scientific_notation(self, tmp_path: Path) -> None:
        """Parses floating point scores written in scientific notation."""
        csv_file = tmp_path / "scientific.csv"
        csv_file.write_text(
            "image_name,quality_mos\nsci_img.png,8.5e-1\n",
            encoding="utf-8",
        )
        scores = load_csv_scores(csv_file, score_column="quality_mos")
        assert scores["sci_img.png"] == pytest.approx(0.85)

    def test_load_csv_scores_row_with_missing_columns(self, tmp_path: Path) -> None:
        """Gracefully handles rows with missing columns where img_col is None."""
        csv_file = tmp_path / "missing_cols.csv"
        # Row with only one value when 2 columns expected
        csv_file.write_text(
            "image_name,score\nvalid.png,0.9\n\n",
            encoding="utf-8",
        )
        scores = load_csv_scores(csv_file, score_column="score")
        assert scores["valid.png"] == pytest.approx(0.9)

    def test_load_csv_scores_row_fewer_columns(self, tmp_path: Path) -> None:
        """Handles rows with fewer columns than header gracefully."""
        csv_file = tmp_path / "ragged.csv"
        # Second row has only one value, so DictReader assigns None to score column or image column
        csv_file.write_text(
            "name,score,extra\nvalid.png,0.85,comment\nragged_row\n",
            encoding="utf-8",
        )
        scores = load_csv_scores(csv_file, score_column="score")
        assert scores["valid.png"] == pytest.approx(0.85)
        # ragged_row was parsed as name with None score (NaN)
        assert math.isnan(scores["ragged_row"])

    def test_load_csv_scores_missing_score_column(self, tmp_path: Path) -> None:
        """Raises KeyError if score column is absent from CSV."""
        csv_file = tmp_path / "bad_cols.csv"
        csv_file.write_text("filename,metric_a\nimg.png,1.0\n", encoding="utf-8")

        with pytest.raises(KeyError, match="Column 'non_existent' not found"):
            load_csv_scores(csv_file, score_column="non_existent")

    def test_load_csv_scores_missing_image_column(self, tmp_path: Path) -> None:
        """Raises KeyError if specified image column is absent from CSV."""
        csv_file = tmp_path / "bad_cols2.csv"
        csv_file.write_text("filename,score\nimg.png,1.0\n", encoding="utf-8")

        with pytest.raises(KeyError, match="Image column 'target_id' not found"):
            load_csv_scores(csv_file, score_column="score", image_column="target_id")

    def test_load_csv_scores_empty_file(self, tmp_path: Path) -> None:
        """Raises KeyError when the CSV file is completely empty."""
        csv_file = tmp_path / "empty.csv"
        csv_file.write_text("", encoding="utf-8")

        with pytest.raises(KeyError, match="empty or missing headers"):
            load_csv_scores(csv_file, score_column="score")


class TestEvaluateCSVScore:
    """Tests for threshold evaluation of scalar CSV scores."""

    def test_evaluate_default_threshold_is_0_7(self) -> None:
        """Default threshold is 0.7 when min_score is omitted."""
        passed, reason = evaluate_csv_score(0.7000)
        assert passed is True
        assert reason is None

        passed, reason = evaluate_csv_score(0.6999)
        assert passed is False
        assert reason == "low_csv_score"

    def test_evaluate_boundary_exact_0_7(self) -> None:
        """Exact boundary verification at 0.7000 threshold."""
        # Exactly 0.7 passes
        passed, reason = evaluate_csv_score(0.7000, min_score=0.7)
        assert passed is True
        assert reason is None

        # Just above 0.7 passes
        passed, reason = evaluate_csv_score(0.700000001, min_score=0.7)
        assert passed is True
        assert reason is None

        # Just below 0.7 fails
        passed, reason = evaluate_csv_score(0.699999999, min_score=0.7)
        assert passed is False
        assert reason == "low_csv_score"

        # 0.6999 fails
        passed, reason = evaluate_csv_score(0.6999, min_score=0.7)
        assert passed is False
        assert reason == "low_csv_score"

    def test_evaluate_custom_thresholds(self) -> None:
        """Custom min_score thresholds evaluate accurately at boundaries."""
        # Custom 0.85
        assert evaluate_csv_score(0.8500, min_score=0.85) == (True, None)
        assert evaluate_csv_score(0.8499, min_score=0.85) == (False, "low_csv_score")

        # Custom 0.0
        assert evaluate_csv_score(0.0000, min_score=0.0) == (True, None)
        assert evaluate_csv_score(-0.0001, min_score=0.0) == (False, "low_csv_score")

    def test_evaluate_passing_score(self) -> None:
        """Score meeting or exceeding threshold passes with no rejection reason."""
        passed, reason = evaluate_csv_score(0.7, min_score=0.7)
        assert passed is True
        assert reason is None

        passed, reason = evaluate_csv_score(0.95, min_score=0.7)
        assert passed is True
        assert reason is None

    def test_evaluate_low_score(self) -> None:
        """Score below threshold is rejected with 'low_csv_score'."""
        passed, reason = evaluate_csv_score(0.699, min_score=0.7)
        assert passed is False
        assert reason == "low_csv_score"

    def test_evaluate_nan_score(self) -> None:
        """NaN score is rejected with 'csv_score_nan'."""
        passed, reason = evaluate_csv_score(float("nan"), min_score=0.7)
        assert passed is False
        assert reason == "csv_score_nan"


class TestCSVScoreFilterStage:
    """Tests verifying CSVScoreFilterStage behavior in quality gate."""

    def test_stage_default_threshold_is_0_7(self) -> None:
        """CSVScoreFilterStage defaults to min_score=0.7."""
        stage = CSVScoreFilterStage()
        assert stage.min_score == 0.7
        assert stage.name == "csv_score"

    def test_stage_satisfies_protocol(self) -> None:
        """Stage must satisfy QualityFilterStage protocol."""
        stage = CSVScoreFilterStage(min_score=0.7)
        assert isinstance(stage, QualityFilterStage)
        assert stage.name == "csv_score"

    def test_stage_with_pre_fetched_score(self) -> None:
        """Explicitly set score via set_current_image takes precedence."""
        stage = CSVScoreFilterStage(min_score=0.7)
        dummy_rgb = np.zeros((10, 10, 3), dtype=np.uint8)
        dummy_gray = np.zeros((10, 10), dtype=np.uint8)

        # Pass case
        stage.set_current_image(filename="photo.jpg", score=0.85)
        res = stage.process(dummy_rgb, dummy_gray)
        assert res.passed is True
        assert res.metric_value == pytest.approx(0.85)
        assert res.rejection_reason is None

        # Exact boundary 0.7000 passes
        stage.set_current_image(filename="photo.jpg", score=0.7000)
        res_boundary = stage.process(dummy_rgb, dummy_gray)
        assert res_boundary.passed is True
        assert res_boundary.rejection_reason is None

        # Just below boundary fails
        stage.set_current_image(filename="photo.jpg", score=0.6999)
        res_below = stage.process(dummy_rgb, dummy_gray)
        assert res_below.passed is False
        assert res_below.rejection_reason == "low_csv_score"

        # Fail case
        stage.set_current_image(filename="photo.jpg", score=0.45)
        res = stage.process(dummy_rgb, dummy_gray)
        assert res.passed is False
        assert res.metric_value == pytest.approx(0.45)
        assert res.rejection_reason == "low_csv_score"

    def test_stage_lookup_by_filename_and_stem(self) -> None:
        """Stage resolves score from internal dictionary by filename and stem."""
        scores = {
            "image_01.png": 0.88,
            "image_02": 0.50,
        }
        stage = CSVScoreFilterStage(min_score=0.7, scores=scores)
        dummy_rgb = np.zeros((10, 10, 3), dtype=np.uint8)
        dummy_gray = np.zeros((10, 10), dtype=np.uint8)

        # Lookup by filename
        stage.set_current_image(filename="image_01.png")
        res1 = stage.process(dummy_rgb, dummy_gray)
        assert res1.passed is True
        assert res1.metric_value == pytest.approx(0.88)

        # Lookup by p.name when relative path provided
        stage.set_current_image(filename="some_folder/image_01.png")
        res_sub = stage.process(dummy_rgb, dummy_gray)
        assert res_sub.passed is True
        assert res_sub.metric_value == pytest.approx(0.88)

        # Lookup by stem
        stage.set_current_image(filename="subdir/image_02.jpg")
        res2 = stage.process(dummy_rgb, dummy_gray)
        assert res2.passed is False
        assert res2.metric_value == pytest.approx(0.50)
        assert res2.rejection_reason == "low_csv_score"

    def test_stage_windows_path_lookup(self) -> None:
        """Stage resolves image when queried with Windows backslashes."""
        scores = {
            "sub/image_win.png": 0.92,
            "image_win.png": 0.92,
        }
        stage = CSVScoreFilterStage(min_score=0.7, scores=scores)
        dummy_rgb = np.zeros((10, 10, 3), dtype=np.uint8)
        dummy_gray = np.zeros((10, 10), dtype=np.uint8)

        stage.set_current_image(filename="sub\\image_win.png")
        res = stage.process(dummy_rgb, dummy_gray)
        assert res.passed is True
        assert res.metric_value == pytest.approx(0.92)

    def test_stage_missing_image_returns_nan(self) -> None:
        """When image is not present in dictionary, returns NaN and fails."""
        stage = CSVScoreFilterStage(min_score=0.7, scores={"known.png": 0.9})
        dummy_rgb = np.zeros((10, 10, 3), dtype=np.uint8)
        dummy_gray = np.zeros((10, 10), dtype=np.uint8)

        stage.set_current_image(filename="unknown.png")
        res = stage.process(dummy_rgb, dummy_gray)
        assert res.passed is False
        assert math.isnan(res.metric_value)
        assert res.rejection_reason == "csv_score_nan"

    def test_stage_without_context_returns_nan(self) -> None:
        """When neither score nor filename is set, returns NaN."""
        stage = CSVScoreFilterStage(min_score=0.7)
        dummy_rgb = np.zeros((10, 10, 3), dtype=np.uint8)
        dummy_gray = np.zeros((10, 10), dtype=np.uint8)

        res = stage.process(dummy_rgb, dummy_gray)
        assert res.passed is False
        assert math.isnan(res.metric_value)
        assert res.rejection_reason == "csv_score_nan"


class TestUHDIQADatasetFormat:
    """Tests evaluating external CSV scores using UHD-IQA metadata format."""

    def test_uhd_iqa_metadata_parsing_and_evaluation(self, tmp_path: Path) -> None:
        """Tests parsing real-world styled uhd-iqa-metadata.csv with image_name and quality_mos."""
        uhd_csv = tmp_path / "uhd-iqa-metadata.csv"
        uhd_csv.write_text(
            "image_name,quality_mos,mos_std,width,height,distortion_type\n"
            "0001_pristine_4k.png,0.9124,0.042,3840,2160,none\n"
            "0002_compressed_4k.png,0.5231,0.088,3840,2160,jpeg\n"
            "0003_boundary_pass.png,0.7000,0.050,3840,2160,noise\n"
            "0004_boundary_fail.png,0.6999,0.061,3840,2160,blur\n"
            "0005_corrupt_score.png,nan,0.000,3840,2160,unknown\n",
            encoding="utf-8",
        )

        scores = load_csv_scores(
            uhd_csv, score_column="quality_mos", image_column="image_name"
        )
        assert len(scores) > 0
        assert scores["0001_pristine_4k.png"] == pytest.approx(0.9124)
        assert scores["0002_compressed_4k.png"] == pytest.approx(0.5231)
        assert scores["0003_boundary_pass.png"] == pytest.approx(0.7000)
        assert scores["0004_boundary_fail.png"] == pytest.approx(0.6999)
        assert math.isnan(scores["0005_corrupt_score.png"])

        stage = CSVScoreFilterStage(min_score=0.7, scores=scores)
        dummy_rgb = np.zeros((10, 10, 3), dtype=np.uint8)
        dummy_gray = np.zeros((10, 10), dtype=np.uint8)

        # 0001 passes
        stage.set_current_image(filename="0001_pristine_4k.png")
        assert stage.process(dummy_rgb, dummy_gray).passed is True

        # 0002 fails (low score)
        stage.set_current_image(filename="0002_compressed_4k.png")
        res2 = stage.process(dummy_rgb, dummy_gray)
        assert res2.passed is False
        assert res2.rejection_reason == "low_csv_score"

        # 0003 boundary passes
        stage.set_current_image(filename="0003_boundary_pass.png")
        assert stage.process(dummy_rgb, dummy_gray).passed is True

        # 0004 boundary fails
        stage.set_current_image(filename="0004_boundary_fail.png")
        res4 = stage.process(dummy_rgb, dummy_gray)
        assert res4.passed is False
        assert res4.rejection_reason == "low_csv_score"

        # 0005 NaN fails
        stage.set_current_image(filename="0005_corrupt_score.png")
        res5 = stage.process(dummy_rgb, dummy_gray)
        assert res5.passed is False
        assert res5.rejection_reason == "csv_score_nan"
