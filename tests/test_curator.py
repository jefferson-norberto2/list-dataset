"""Integration tests for the DatasetCurator service layer."""

from __future__ import annotations

import csv
from pathlib import Path

import cv2
import numpy as np

from dataset_curator.config import CurateConfig
from dataset_curator.curator import DatasetCurator
from dataset_curator.pipeline import CascadedQualityGate


class AlwaysPassGate(CascadedQualityGate):
    """Gate that accepts everything for integration testing."""

    def __init__(self) -> None:
        pass

    def evaluate(self, image_rgb: np.ndarray):
        from dataset_curator.pipeline import CurationAssessment

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
        from dataset_curator.pipeline import CurationAssessment

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
