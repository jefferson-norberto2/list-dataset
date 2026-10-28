"""Service layer orchestrating the image dataset curation workflow."""

from __future__ import annotations

import logging
import math
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from tqdm import tqdm

from dataset.curator.audit import AuditEntry, AuditLogger
from dataset.curator.config import CurateConfig
from dataset.curator.metrics.csv_score import load_csv_scores
from dataset.curator.pipeline import CascadedQualityGate
from dataset.curator.preprocessing import (
    ImageLoadingError,
    ImageResolutionError,
    find_supported_images,
    load_image,
    preprocess_image,
    save_image_lossless_png,
)

logger = logging.getLogger(__name__)


class DatasetCurator:
    """Core service layer for curating pristine reference image datasets.

    Coordinates image discovery, decoding, geometric normalization, cascaded quality
    filtering, artifact saving, and CSV audit logging.
    """

    def __init__(
        self,
        config: CurateConfig,
        gate: CascadedQualityGate | None = None,
        audit_logger: AuditLogger | None = None,
    ) -> None:
        """Initializes the DatasetCurator service.

        Args:
            config: Operational configuration settings and quality thresholds.
            gate: Custom quality gate instance. Defaults to configured CascadedQualityGate.
            audit_logger: Custom audit logger instance. Defaults to AuditLogger at config.audit_csv_path.
        """
        self.config = config
        self.csv_scores: dict[str, float] | None = None
        if config.is_csv_score_enabled:
            assert config.csv_score_path is not None
            assert config.csv_score_column is not None
            self.csv_scores = load_csv_scores(
                csv_path=config.csv_score_path,
                score_column=config.csv_score_column,
            )

        self.gate = gate or CascadedQualityGate(config=config)
        self.audit_logger = audit_logger or AuditLogger(csv_path=config.audit_csv_path)

    def process_file(self, image_path: Path) -> AuditEntry:
        """Processes a single image file through loading, preprocessing, and quality cascade.

        Args:
            image_path: Path to the image file to curate.

        Returns:
            AuditEntry: Audit log entry containing the decision and metrics.
        """
        rel_name = (
            str(image_path.relative_to(self.config.input_dir))
            if self.config.input_dir
            and image_path.is_relative_to(self.config.input_dir)
            else image_path.name
        )

        # 0. External CSV Quality Score Check (if configured)
        score: float | None = None
        if self.csv_scores is not None:
            rel_norm = rel_name.replace("\\", "/")
            if rel_name in self.csv_scores:
                score = self.csv_scores[rel_name]
            elif rel_norm in self.csv_scores:
                score = self.csv_scores[rel_norm]
            elif image_path.name in self.csv_scores:
                score = self.csv_scores[image_path.name]
            elif image_path.stem in self.csv_scores:
                score = self.csv_scores[image_path.stem]
            else:
                logger.info("Image rejected: score not found in CSV (%s)", rel_name)
                return self.audit_logger.record_entry(
                    filename=rel_name,
                    pristine=False,
                    rejection_reason="csv_score_not_found",
                    csv_score=None,
                )

            if math.isnan(score):
                logger.info("Image rejected: score is NaN in CSV (%s)", rel_name)
                return self.audit_logger.record_entry(
                    filename=rel_name,
                    pristine=False,
                    rejection_reason="csv_score_nan",
                    csv_score=None,
                )

            if score < self.config.min_csv_score:
                logger.info(
                    "Image rejected: low CSV quality score %.4f < %.4f (%s)",
                    score,
                    self.config.min_csv_score,
                    rel_name,
                )
                return self.audit_logger.record_entry(
                    filename=rel_name,
                    pristine=False,
                    rejection_reason="low_csv_score",
                    csv_score=score,
                )

        # 1. Image Loading
        try:
            image_rgb = load_image(image_path)
        except FileNotFoundError:
            logger.warning("Image file not found: %s", image_path)
            return self.audit_logger.record_error(
                rel_name, "file_not_found", csv_score=score
            )
        except ImageLoadingError as exc:
            logger.warning("Unreadable or corrupted image (%s): %s", image_path, exc)
            return self.audit_logger.record_error(
                rel_name, "corrupted_or_unreadable", csv_score=score
            )
        except Exception as exc:  # noqa: BLE001 - safeguard batch processing from unhandled image errors
            logger.error("Unexpected error loading image %s: %s", image_path, exc)
            return self.audit_logger.record_error(
                rel_name, f"load_error: {type(exc).__name__}", csv_score=score
            )

        # 2. Geometric Preprocessing (Lanczos4 downsampling + center crop)
        try:
            crop_rgb = preprocess_image(
                image_rgb,
                target_dim=self.config.target_dim,
                crop_size=self.config.crop_size,
            )
        except ImageResolutionError as exc:
            logger.info(
                "Image rejected due to low resolution (%s): %s", image_path, exc
            )
            return self.audit_logger.record_error(
                rel_name, "resolution_too_small", csv_score=score
            )
        except Exception as exc:  # noqa: BLE001 - safeguard batch processing from corrupted tensors
            logger.error("Unexpected error preprocessing image %s: %s", image_path, exc)
            return self.audit_logger.record_error(
                rel_name, f"preprocessing_error: {type(exc).__name__}", csv_score=score
            )

        # 3. Cascaded Quality Gate
        try:
            assessment = self.gate.evaluate(
                crop_rgb, filename=rel_name, csv_score=score
            )
        except TypeError:
            assessment = self.gate.evaluate(crop_rgb)

        if assessment.csv_score is None and score is not None:
            assessment.csv_score = score

        # 4. Artifact Handling
        saved_path_str = ""
        if assessment.is_pristine:
            filename = self.config.filename_pattern.format(
                stem=image_path.stem,
                target_dim=self.config.target_dim,
            )
            destination = self.config.output_dir / filename
            try:
                save_image_lossless_png(
                    image_rgb=crop_rgb,
                    destination=destination,
                    compression_level=self.config.png_compression,
                )
                saved_path_str = str(destination)
                logger.info("Accepted pristine image: %s -> %s", rel_name, destination)
            except Exception as exc:  # noqa: BLE001 - handle filesystem or permission errors gracefully
                logger.error("Failed to save accepted image %s: %s", destination, exc)
                return self.audit_logger.record_error(
                    rel_name, f"save_error: {type(exc).__name__}", csv_score=score
                )
        else:
            logger.info(
                "Rejected non-pristine image: %s (reason: %s)",
                rel_name,
                assessment.rejection_reason,
            )

        return self.audit_logger.record_assessment(
            filename=rel_name,
            assessment=assessment,
            output_path=saved_path_str,
        )

    def run_curation(
        self,
        images: Sequence[Path] | None = None,
        show_progress: bool = True,
    ) -> dict[str, Any]:
        """Executes batch curation over input images.

        Args:
            images: Explicit list of image paths. If None, scans config.input_dir.
            show_progress: Whether to display a tqdm progress bar.

        Returns:
            dict[str, Any]: Aggregate statistics summary.
        """
        if images is None:
            logger.info(
                "Scanning %s for supported image formats...", self.config.input_dir
            )
            image_list = find_supported_images(
                self.config.input_dir,
                supported_extensions=self.config.supported_formats,
            )
        else:
            image_list = list(images)

        total_files = len(image_list)
        logger.info("Discovered %d candidate images to process.", total_files)

        if total_files == 0:
            logger.warning(
                "No candidate images found matching criteria in %s",
                self.config.input_dir,
            )
            return self.audit_logger.summary()

        # Ensure output directory exists
        self.config.output_dir.mkdir(parents=True, exist_ok=True)

        accepted_count = 0
        pbar = tqdm(
            image_list, desc="Curating Pristine Dataset", disable=not show_progress
        )
        for img_path in pbar:
            entry = self.process_file(img_path)
            if entry.pristine:
                accepted_count += 1
            pbar.set_postfix(
                accepted=accepted_count,
                rejected=len(self.audit_logger.entries) - accepted_count,
            )

        summary = self.audit_logger.summary()
        logger.info(
            "Curation finished: %d total, %d pristine accepted (%.1f%%), %d rejected.",
            summary["total_images"],
            summary["accepted_pristine"],
            summary["acceptance_rate_pct"],
            summary["rejected"],
        )
        return summary
