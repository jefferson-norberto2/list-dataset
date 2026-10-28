"""Integration tests for the DatasetCurator service layer."""

from __future__ import annotations

import csv
from pathlib import Path

import cv2
import numpy as np
import pytest

from dataset.curator import CurateConfig, DatasetCurator, CascadedQualityGate


class AlwaysPassGate(CascadedQualityGate):
    """Gate that accepts everything for integration testing."""

    def __init__(self) -> None:
        pass

    def evaluate(self, image_rgb: np.ndarray):
        from dataset.curator import CurationAssessment

        return CurationAssessment(
            is_pristine=True,
            rejection_reason="accepted",
            clipping_ratio=0.001,
            luminance_std=52.0,
            entropy=7.4,
            laplacian_var=160.0,
            niqe_score=2.8,
        )


class AlwaysRejectGate(CascadedQualityGate):
    """Gate that rejects everything for integration testing."""

    def __init__(self) -> None:
        pass

    def evaluate(self, image_rgb: np.ndarray):
        from dataset.curator import CurationAssessment

        return CurationAssessment(
            is_pristine=False,
            rejection_reason="blurry",
            clipping_ratio=0.001,
            luminance_std=52.0,
            entropy=7.4,
            laplacian_var=80.0,
            niqe_score=None,
        )


class TestDatasetCuratorIntegration:
    """End-to-end integration tests for DatasetCurator."""

    def test_curation_workflow_accepted_and_audit(self, tmp_path: Path) -> None:
        """Verifies accepted images are saved as lossless PNGs and logged to CSV."""
        input_dir = tmp_path / "raw"
        output_dir = tmp_path / "pristine"
        audit_csv = output_dir / "audit.csv"
        input_dir.mkdir()

        # Create two valid test images (600x800)
        img1 = np.full((600, 800, 3), 100, dtype=np.uint8)
        img2 = np.full((600, 800, 3), 150, dtype=np.uint8)
        cv2.imwrite(str(input_dir / "scene_01.png"), img1)
        cv2.imwrite(str(input_dir / "scene_02.jpg"), img2)

        config = CurateConfig(
            input_dir=input_dir,
            output_dir=output_dir,
            audit_csv_path=audit_csv,
            target_dim=512,
            crop_size=(512, 512),
        )

        curator = DatasetCurator(config=config, gate=AlwaysPassGate())
        summary = curator.run_curation(show_progress=False)

        assert summary["total_images"] == 2
        assert summary["accepted_pristine"] == 2
        assert summary["rejected"] == 0

        # Verify saved pristine files
        saved_file1 = output_dir / "scene_01_pristine_512.png"
        saved_file2 = output_dir / "scene_02_pristine_512.png"
        assert saved_file1.is_file()
        assert saved_file2.is_file()

        # Verify saved crop dimensions
        loaded = cv2.imread(str(saved_file1))
        assert loaded.shape == (512, 512, 3)

        # Verify CSV log content
        assert audit_csv.is_file()
        with open(audit_csv, mode="r", encoding="utf-8") as f:
            reader = list(csv.DictReader(f))
            assert len(reader) == 2
            assert reader[0]["pristine"] == "True"
            assert reader[0]["rejection_reason"] == "accepted"
            assert reader[0]["output_path"] != ""

    def test_curation_workflow_rejection(self, tmp_path: Path) -> None:
        """Verifies rejected images are logged with rejection reason and not saved."""
        input_dir = tmp_path / "raw"
        output_dir = tmp_path / "pristine"
        audit_csv = output_dir / "audit.csv"
        input_dir.mkdir()

        img = np.full((600, 800, 3), 100, dtype=np.uint8)
        cv2.imwrite(str(input_dir / "blurry_pic.png"), img)

        config = CurateConfig(
            input_dir=input_dir,
            output_dir=output_dir,
            audit_csv_path=audit_csv,
            target_dim=512,
            crop_size=(512, 512),
        )

        curator = DatasetCurator(config=config, gate=AlwaysRejectGate())
        summary = curator.run_curation(show_progress=False)

        assert summary["total_images"] == 1
        assert summary["accepted_pristine"] == 0
        assert summary["rejected"] == 1
        assert summary["rejections_by_reason"] == {"blurry": 1}

        # Verify file was NOT saved to output directory
        assert not (output_dir / "blurry_pic_pristine_512.png").exists()

        # Verify CSV log records rejection
        with open(audit_csv, mode="r", encoding="utf-8") as f:
            reader = list(csv.DictReader(f))
            assert len(reader) == 1
            assert reader[0]["pristine"] == "False"
            assert reader[0]["rejection_reason"] == "blurry"
            assert reader[0]["output_path"] == ""

    def test_curation_resilience_to_corrupt_files(self, tmp_path: Path) -> None:
        """Pipeline must gracefully log corrupt files without aborting batch."""
        input_dir = tmp_path / "raw"
        output_dir = tmp_path / "pristine"
        input_dir.mkdir()

        # Create 1 corrupt file and 1 valid file
        (input_dir / "bad_image.png").write_bytes(b"invalid data")
        valid_img = np.full((600, 800, 3), 120, dtype=np.uint8)
        cv2.imwrite(str(input_dir / "good_image.png"), valid_img)

        config = CurateConfig(
            input_dir=input_dir,
            output_dir=output_dir,
            audit_csv_path=output_dir / "audit.csv",
        )

        curator = DatasetCurator(config=config, gate=AlwaysPassGate())
        summary = curator.run_curation(show_progress=False)

        assert summary["total_images"] == 2
        assert summary["accepted_pristine"] == 1
        assert summary["rejected"] == 1
        assert "corrupted_or_unreadable" in summary["rejections_by_reason"]

    def test_curation_with_external_csv_scores(self, tmp_path: Path) -> None:
        """Verifies end-to-end filtering using external CSV quality scores."""
        input_dir = tmp_path / "raw"
        output_dir = tmp_path / "pristine"
        audit_csv = output_dir / "audit.csv"
        scores_csv = tmp_path / "scores.csv"
        input_dir.mkdir()

        # Create test images
        img = np.full((600, 600, 3), 120, dtype=np.uint8)
        cv2.imwrite(str(input_dir / "pass.png"), img)
        cv2.imwrite(str(input_dir / "low_score.png"), img)
        cv2.imwrite(str(input_dir / "nan_score.png"), img)
        cv2.imwrite(str(input_dir / "missing_from_csv.png"), img)

        scores_csv.write_text(
            "filename,quality_score\n"
            "pass.png,0.88\n"
            "low_score.png,0.45\n"
            "nan_score.png,invalid_val\n",
            encoding="utf-8",
        )

        config = CurateConfig(
            input_dir=input_dir,
            output_dir=output_dir,
            audit_csv_path=audit_csv,
            target_dim=512,
            crop_size=(512, 512),
            csv_score_path=scores_csv,
            csv_score_column="quality_score",
            min_csv_score=0.7,
        )

        curator = DatasetCurator(config=config, gate=AlwaysPassGate())
        summary = curator.run_curation(show_progress=False)

        assert summary["total_images"] == 4
        assert summary["accepted_pristine"] == 1
        assert summary["rejected"] == 3

        with open(audit_csv, mode="r", encoding="utf-8") as f:
            rows = {r["filename"]: r for r in csv.DictReader(f)}

        # pass.png
        assert rows["pass.png"]["pristine"] == "True"
        assert rows["pass.png"]["rejection_reason"] == "accepted"
        assert rows["pass.png"]["csv_score"] == "0.8800"

        # low_score.png
        assert rows["low_score.png"]["pristine"] == "False"
        assert rows["low_score.png"]["rejection_reason"] == "low_csv_score"
        assert rows["low_score.png"]["csv_score"] == "0.4500"

        # nan_score.png
        assert rows["nan_score.png"]["pristine"] == "False"
        assert rows["nan_score.png"]["rejection_reason"] == "csv_score_nan"
        assert rows["nan_score.png"]["csv_score"] == ""

        # missing_from_csv.png
        assert rows["missing_from_csv.png"]["pristine"] == "False"
        assert rows["missing_from_csv.png"]["rejection_reason"] == "csv_score_not_found"
        assert rows["missing_from_csv.png"]["csv_score"] == ""

    def test_curation_with_csv_score_name_and_stem_match(self, tmp_path: Path) -> None:
        """Verifies CSV scores match by filename or stem when nested in subdirectories."""
        input_dir = tmp_path / "raw"
        output_dir = tmp_path / "pristine"
        audit_csv = output_dir / "audit.csv"
        scores_csv = tmp_path / "scores.csv"

        sub = input_dir / "subdir"
        sub.mkdir(parents=True)

        img = np.full((600, 600, 3), 120, dtype=np.uint8)
        cv2.imwrite(str(sub / "nested_by_name.png"), img)
        cv2.imwrite(str(sub / "nested_by_stem.jpg"), img)

        # CSV entries without the "subdir/" prefix
        scores_csv.write_text(
            "name,score\nnested_by_name.png,0.95\nnested_by_stem,0.90\n",
            encoding="utf-8",
        )

        config = CurateConfig(
            input_dir=input_dir,
            output_dir=output_dir,
            audit_csv_path=audit_csv,
            csv_score_path=scores_csv,
            csv_score_column="score",
            min_csv_score=0.7,
        )

        curator = DatasetCurator(config=config, gate=AlwaysPassGate())
        summary = curator.run_curation(show_progress=False)

        assert summary["total_images"] == 2
        assert summary["accepted_pristine"] == 2

        with open(audit_csv, mode="r", encoding="utf-8") as f:
            rows = {r["filename"]: r for r in csv.DictReader(f)}

        assert rows["subdir/nested_by_name.png"]["csv_score"] == "0.9500"
        assert rows["subdir/nested_by_stem.jpg"]["csv_score"] == "0.9000"

    def test_curation_rel_norm_lookup(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Verifies lookup when rel_name contains backslashes and matches rel_norm in csv_scores."""
        input_dir = tmp_path / "raw"
        output_dir = tmp_path / "pristine"
        scores_csv = tmp_path / "scores.csv"
        input_dir.mkdir()

        scores_csv.write_text(
            "image_name,score\nfolder/win.png,0.85\n", encoding="utf-8"
        )
        config = CurateConfig(
            input_dir=input_dir,
            output_dir=output_dir,
            csv_score_path=scores_csv,
            csv_score_column="score",
        )
        curator = DatasetCurator(config=config, gate=AlwaysPassGate())

        # Mock relative_to to return a path with backslash
        class MockPath:
            name = "other.png"
            stem = "other"

            def __str__(self):
                return "folder\\win.png"

            def is_relative_to(self, _):
                return True

            def relative_to(self, _):
                return "folder\\win.png"

        monkeypatch.setattr(
            "dataset.curator.load_image",
            lambda _: np.zeros((600, 600, 3), dtype=np.uint8),
        )
        entry = curator.process_file(MockPath())  # type: ignore[arg-type]
        assert entry.csv_score == pytest.approx(0.85)

    def test_curation_empty_dataset(self, tmp_path: Path) -> None:
        """Running curation on empty folder returns clean summary."""
        input_dir = tmp_path / "empty_dir"
        input_dir.mkdir()
        config = CurateConfig(
            input_dir=input_dir,
            output_dir=tmp_path / "out",
        )
        curator = DatasetCurator(config=config)
        summary = curator.run_curation(show_progress=False)
        assert summary["total_images"] == 0

    def test_curation_inactive_when_only_one_parameter_provided(
        self, tmp_path: Path
    ) -> None:
        """When only csv_score_path or only csv_score_column is passed, the measure is ignored."""
        input_dir = tmp_path / "raw"
        output_dir = tmp_path / "pristine"
        audit_csv = output_dir / "audit.csv"
        scores_csv = tmp_path / "scores.csv"
        input_dir.mkdir()

        img = np.full((600, 600, 3), 120, dtype=np.uint8)
        cv2.imwrite(str(input_dir / "test.png"), img)
        scores_csv.write_text(
            "image_name,quality_mos\ntest.png,0.20\n", encoding="utf-8"
        )

        # 1. Only csv_score_path passed -> inactive, image passes with AlwaysPassGate
        cfg_path_only = CurateConfig(
            input_dir=input_dir,
            output_dir=output_dir,
            audit_csv_path=audit_csv,
            csv_score_path=scores_csv,
            csv_score_column=None,
        )
        curator_path_only = DatasetCurator(config=cfg_path_only, gate=AlwaysPassGate())
        assert curator_path_only.csv_scores is None
        summary1 = curator_path_only.run_curation(show_progress=False)
        assert summary1["accepted_pristine"] == 1

        # 2. Only csv_score_column passed -> inactive, image passes with AlwaysPassGate
        cfg_col_only = CurateConfig(
            input_dir=input_dir,
            output_dir=output_dir,
            audit_csv_path=audit_csv,
            csv_score_path=None,
            csv_score_column="quality_mos",
        )
        curator_col_only = DatasetCurator(config=cfg_col_only, gate=AlwaysPassGate())
        assert curator_col_only.csv_scores is None
        summary2 = curator_col_only.run_curation(show_progress=False)
        assert summary2["accepted_pristine"] == 1

    def test_curation_uhd_iqa_real_world_format(self, tmp_path: Path) -> None:
        """Verifies curation against actual UHD-IQA metadata CSV format with default 0.7 threshold."""
        input_dir = tmp_path / "raw_uhd"
        output_dir = tmp_path / "pristine_uhd"
        audit_csv = output_dir / "audit.csv"
        metadata_csv = tmp_path / "uhd-iqa-metadata.csv"
        input_dir.mkdir()

        # Mimic UHD-IQA CSV format: "image_name","quality_mos","set",...
        metadata_csv.write_text(
            '"image_name","quality_mos","set","orig_width","orig_height"\n'
            '"1.jpg",0.7600,"training",5515,3375\n'
            '"2.jpg",0.7000,"training",4850,3145\n'
            '"3.jpg",0.6999,"training",4000,3000\n'
            '"4.jpg",0.5745,"test",5456,3632\n',
            encoding="utf-8",
        )

        img = np.full((600, 600, 3), 120, dtype=np.uint8)
        for name in ["1.jpg", "2.jpg", "3.jpg", "4.jpg"]:
            cv2.imwrite(str(input_dir / name), img)

        config = CurateConfig(
            input_dir=input_dir,
            output_dir=output_dir,
            audit_csv_path=audit_csv,
            csv_score_path=metadata_csv,
            csv_score_column="quality_mos",
            # min_csv_score defaults to 0.7
        )
        assert config.min_csv_score == 0.7

        curator = DatasetCurator(config=config, gate=AlwaysPassGate())
        summary = curator.run_curation(show_progress=False)

        assert summary["total_images"] == 4
        # 1.jpg (0.76) and 2.jpg (0.70) pass; 3.jpg (0.6999) and 4.jpg (0.5745) fail
        assert summary["accepted_pristine"] == 2
        assert summary["rejected"] == 2

        with open(audit_csv, mode="r", encoding="utf-8") as f:
            rows = {r["filename"]: r for r in csv.DictReader(f)}

        assert rows["1.jpg"]["pristine"] == "True"
        assert rows["1.jpg"]["csv_score"] == "0.7600"
        assert rows["2.jpg"]["pristine"] == "True"
        assert rows["2.jpg"]["csv_score"] == "0.7000"

        assert rows["3.jpg"]["pristine"] == "False"
        assert rows["3.jpg"]["rejection_reason"] == "low_csv_score"
        assert rows["3.jpg"]["csv_score"] == "0.6999"

        assert rows["4.jpg"]["pristine"] == "False"
        assert rows["4.jpg"]["rejection_reason"] == "low_csv_score"
        assert rows["4.jpg"]["csv_score"] == "0.5745"

    def test_curation_early_exit_skips_image_loading(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """When CSV score check fails, load_image must never be called."""
        input_dir = tmp_path / "raw"
        output_dir = tmp_path / "pristine"
        scores_csv = tmp_path / "scores.csv"
        input_dir.mkdir()

        # Create dummy file
        (input_dir / "rejected.png").write_bytes(b"dummy")
        scores_csv.write_text("image_name,score\nrejected.png,0.30\n", encoding="utf-8")

        config = CurateConfig(
            input_dir=input_dir,
            output_dir=output_dir,
            csv_score_path=scores_csv,
            csv_score_column="score",
        )

        load_calls = []

        def mock_load(path: Path):
            load_calls.append(path)
            return np.zeros((100, 100, 3), dtype=np.uint8)

        monkeypatch.setattr("dataset.curator.load_image", mock_load)

        curator = DatasetCurator(config=config, gate=AlwaysPassGate())
        entry = curator.process_file(input_dir / "rejected.png")

        assert entry.pristine is False
        assert entry.rejection_reason == "low_csv_score"
        # load_image was never called due to early exit!
        assert len(load_calls) == 0

    def test_curation_exception_handling(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Verifies error handling for load, preprocess, save errors, and explicit image lists."""
        input_dir = tmp_path / "raw"
        output_dir = tmp_path / "pristine"
        input_dir.mkdir()
        test_img = input_dir / "test.png"
        cv2.imwrite(str(test_img), np.full((600, 600, 3), 120, dtype=np.uint8))

        config = CurateConfig(input_dir=input_dir, output_dir=output_dir)

        # 1. FileNotFoundError during load_image
        def mock_fnf(_):
            raise FileNotFoundError("missing")

        monkeypatch.setattr("dataset.curator.load_image", mock_fnf)
        curator = DatasetCurator(config=config)
        e1 = curator.process_file(test_img)
        assert e1.rejection_reason == "file_not_found"

        # 2. Unexpected RuntimeError during load_image
        def mock_load_err(_):
            raise RuntimeError("load crash")

        monkeypatch.setattr("dataset.curator.load_image", mock_load_err)
        e2 = curator.process_file(test_img)
        assert e2.rejection_reason == "load_error: RuntimeError"

        # 3. Unexpected RuntimeError during preprocess_image
        monkeypatch.undo()

        def mock_prep_err(*args, **kwargs):
            raise RuntimeError("prep crash")

        monkeypatch.setattr("dataset.curator.preprocess_image", mock_prep_err)
        e3 = curator.process_file(test_img)
        assert e3.rejection_reason == "preprocessing_error: RuntimeError"

        # 4. Save error during save_image_lossless_png
        monkeypatch.undo()

        def mock_save_err(*args, **kwargs):
            raise OSError("disk full")

        monkeypatch.setattr(
            "dataset.curator.save_image_lossless_png", mock_save_err
        )
        curator_pass = DatasetCurator(config=config, gate=AlwaysPassGate())
        e4 = curator_pass.process_file(test_img)
        assert e4.rejection_reason == "save_error: OSError"

        # 5. Passing explicit list of images to run_curation
        monkeypatch.undo()
        curator_explicit = DatasetCurator(config=config, gate=AlwaysPassGate())
        summary = curator_explicit.run_curation(images=[test_img], show_progress=False)
        assert summary["total_images"] == 1
        assert summary["accepted_pristine"] == 1
