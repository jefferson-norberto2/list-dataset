"""Structured CSV audit logging and reporting for curated datasets."""

from __future__ import annotations

import csv
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar

from dataset_curator.pipeline import CurationAssessment

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class AuditEntry:
    """Audit log entry for an individual processed image file."""

    filename: str
    pristine: bool
    rejection_reason: str
    clipping_ratio: float | None = None
    luminance_std: float | None = None
    entropy: float | None = None
    laplacian_variance: float | None = None
    niqe_score: float | None = None
    output_path: str = ""

    def to_row(self) -> dict[str, Any]:
        """Converts entry to a dictionary formatted for CSV writing."""
        return {
            "filename": self.filename,
            "pristine": str(self.pristine),
            "rejection_reason": self.rejection_reason,
            "clipping_ratio": f"{self.clipping_ratio:.6f}"
            if self.clipping_ratio is not None
            else "",
            "luminance_std": f"{self.luminance_std:.4f}"
            if self.luminance_std is not None
            else "",
            "entropy": f"{self.entropy:.4f}" if self.entropy is not None else "",
            "laplacian_variance": f"{self.laplacian_variance:.4f}"
            if self.laplacian_variance is not None
            else "",
            "niqe_score": f"{self.niqe_score:.4f}"
            if self.niqe_score is not None
            else "",
            "output_path": self.output_path,
        }


class AuditLogger:
    """Manages writing structured CSV audit logs and aggregating processing statistics."""

    FIELDNAMES: ClassVar[list[str]] = [
        "filename",
        "pristine",
        "rejection_reason",
        "clipping_ratio",
        "luminance_std",
        "entropy",
        "laplacian_variance",
        "niqe_score",
        "output_path",
    ]

    def __init__(self, csv_path: Path) -> None:
        """Initializes the audit logger.

        Args:
            csv_path: Destination path for the CSV audit log file.
        """
        self.csv_path = csv_path
        self.entries: list[AuditEntry] = []
        self._initialized = False

    def _ensure_file_initialized(self) -> None:
        """Creates the parent directory and writes the CSV header row if not present."""
        if not self._initialized:
            self.csv_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.csv_path, mode="w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=self.FIELDNAMES)
                writer.writeheader()
            self._initialized = True

    def record_assessment(
        self,
        filename: str,
        assessment: CurationAssessment,
        output_path: str = "",
    ) -> AuditEntry:
        """Appends an assessment outcome to the audit log.

        Args:
            filename: Name or relative path of the processed source file.
            assessment: Outcome returned by the cascaded quality gate.
            output_path: Saved crop file path if accepted, else empty string.

        Returns:
            AuditEntry: The logged audit record.
        """
        self._ensure_file_initialized()

        entry = AuditEntry(
            filename=filename,
            pristine=assessment.is_pristine,
            rejection_reason=assessment.rejection_reason,
            clipping_ratio=assessment.clipping_ratio,
            luminance_std=assessment.luminance_std,
            entropy=assessment.entropy,
            laplacian_variance=assessment.laplacian_var,
            niqe_score=assessment.niqe_score,
            output_path=output_path,
        )

        self.entries.append(entry)

        with open(self.csv_path, mode="a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=self.FIELDNAMES)
            writer.writerow(entry.to_row())

        return entry

    def record_error(
        self,
        filename: str,
        error_reason: str,
    ) -> AuditEntry:
        """Records an unprocessable or corrupted file in the audit log.

        Args:
            filename: Source file name or path.
            error_reason: Reason why the file could not be processed.

        Returns:
            AuditEntry: The recorded audit record.
        """
        self._ensure_file_initialized()

        entry = AuditEntry(
            filename=filename,
            pristine=False,
            rejection_reason=error_reason,
            output_path="",
        )

        self.entries.append(entry)

        with open(self.csv_path, mode="a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=self.FIELDNAMES)
            writer.writerow(entry.to_row())

        return entry

    def summary(self) -> dict[str, Any]:
        """Calculates aggregate curation statistics.

        Returns:
            dict[str, Any]: Summary dictionary with counts and percentages.
        """
        total = len(self.entries)
        accepted = sum(1 for e in self.entries if e.pristine)
        rejected = total - accepted

        rejection_counts: dict[str, int] = {}
        for e in self.entries:
            if not e.pristine:
                rejection_counts[e.rejection_reason] = (
                    rejection_counts.get(e.rejection_reason, 0) + 1
                )

        acceptance_rate = (accepted / total * 100.0) if total > 0 else 0.0

        return {
            "total_images": total,
            "accepted_pristine": accepted,
            "rejected": rejected,
            "acceptance_rate_pct": acceptance_rate,
            "rejections_by_reason": rejection_counts,
            "audit_file": str(self.csv_path),
        }
