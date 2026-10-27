"""End-to-end integration tests for the dataset curation CLI."""

from __future__ import annotations

import csv
from pathlib import Path

import cv2
import numpy as np

from dataset_curator.cli import main


class TestCLIIntegration:
    """Tests executing the curation pipeline through the CLI interface."""

    def test_cli_end_to_end_cascade(self, tmp_path: Path) -> None:
        """Executes the CLI against synthetic test cases exercising every failure mode and acceptance."""
        input_dir = tmp_path / "raw_dataset"
        output_dir = tmp_path / "pristine_dataset"
        audit_csv = tmp_path / "audit_report.csv"
        input_dir.mkdir()

        # 1. Clipped Image (fails clipping stage)
        clipped_img = np.full((600, 600, 3), 128, dtype=np.uint8)
        clipped_img[:50, :, :] = 255  # ~8% clipped > 1.0%
        cv2.imwrite(str(input_dir / "case_clipped.png"), clipped_img)

        # 2. Low Contrast Image (fails contrast stage: std < 40.0)
        low_contrast = np.full((600, 600, 3), 128, dtype=np.uint8)
        noise = (
            np.random.RandomState(42)
            .randint(-3, 4, size=(600, 600, 3))
            .astype(np.int16)
        )
        low_contrast = np.clip(low_contrast.astype(np.int16) + noise, 0, 255).astype(
            np.uint8
        )
        cv2.imwrite(str(input_dir / "case_low_contrast.png"), low_contrast)

        # 3. Low Entropy Image (passes clipping and contrast, fails entropy stage: entropy < 7.0)
        palette = np.array(
            [[50, 50, 50], [90, 90, 90], [160, 160, 160], [205, 205, 205]],
            dtype=np.uint8,
        )
        indices = np.random.RandomState(42).randint(0, 4, size=(6, 6))
        indices = np.repeat(np.repeat(indices, 100, axis=0), 100, axis=1)
        low_entropy = palette[indices]
        cv2.imwrite(str(input_dir / "case_low_entropy.png"), low_entropy)

        # 4. Blurry Image (passes clipping, contrast, and entropy; fails sharpness stage: Laplacian var < 120.0)
        _y, x = np.mgrid[0:600, 0:600]
        grad = ((x / 600.0) * 180 + 35).astype(np.uint8)
        blurry = cv2.GaussianBlur(grad, (25, 25), 0)
        blurry_rgb = np.stack([blurry, blurry, blurry], axis=-1)
        cv2.imwrite(str(input_dir / "case_blurry.png"), blurry_rgb)

        # 5. Corrupted File (header unreadable)
        (input_dir / "case_corrupt.png").write_bytes(b"bad header data")

        # 6. Undersized Image (< 512)
        small_img = np.full((256, 256, 3), 128, dtype=np.uint8)
        cv2.imwrite(str(input_dir / "case_small.png"), small_img)

        # Run CLI with generous NIQE threshold so synthetic sharp texture can pass NIQE
        exit_code = main(
            [
                "--input-dir",
                str(input_dir),
                "--output-dir",
                str(output_dir),
                "--audit-csv",
                str(audit_csv),
                "--max-niqe",
                "25.0",
                "--device",
                "cpu",
                "--quiet",
            ]
        )

        assert exit_code == 0
        assert audit_csv.is_file()

        # Parse CSV results
        with open(audit_csv, mode="r", encoding="utf-8") as f:
            rows = {r["filename"]: r for r in csv.DictReader(f)}

        assert len(rows) == 6
        assert rows["case_clipped.png"]["pristine"] == "False"
        assert rows["case_clipped.png"]["rejection_reason"] == "clipping_exceeded"
        # Contrast was skipped, so luminance_std must be empty
        assert rows["case_clipped.png"]["luminance_std"] == ""

        assert rows["case_low_contrast.png"]["pristine"] == "False"
        assert rows["case_low_contrast.png"]["rejection_reason"] == "low_contrast"
        # Clipping was computed, but entropy was skipped
        assert rows["case_low_contrast.png"]["clipping_ratio"] != ""
        assert rows["case_low_contrast.png"]["entropy"] == ""

        assert rows["case_low_entropy.png"]["pristine"] == "False"
        assert rows["case_low_entropy.png"]["rejection_reason"] == "low_entropy"
        assert rows["case_low_entropy.png"]["laplacian_variance"] == ""

        assert rows["case_blurry.png"]["pristine"] == "False"
        assert rows["case_blurry.png"]["rejection_reason"] == "blurry"
        assert rows["case_blurry.png"]["niqe_score"] == ""

        assert rows["case_corrupt.png"]["pristine"] == "False"
        assert rows["case_corrupt.png"]["rejection_reason"] == "corrupted_or_unreadable"

        assert rows["case_small.png"]["pristine"] == "False"
        assert rows["case_small.png"]["rejection_reason"] == "resolution_too_small"
