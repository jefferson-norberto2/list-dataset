"""External CSV quality score evaluation and filter stage."""

from __future__ import annotations

import csv
import math
from pathlib import Path

import numpy as np

from dataset.curator.metrics.base import FilterResult, QualityFilterStage

IMAGE_COLUMN_CANDIDATES: tuple[str, ...] = (
    "image_name",
    "filename",
    "image",
    "file",
    "name",
    "path",
)


def load_csv_scores(
    csv_path: Path,
    score_column: str,
    image_column: str | None = None,
) -> dict[str, float]:
    """Loads image quality scores from an external CSV file.

    Reads the specified CSV file and constructs a lookup dictionary mapping
    image identifiers (original value, POSIX path, filename with extension,
    and stem) to their parsed floating-point quality scores.

    Args:
        csv_path: Path to the external CSV file.
        score_column: Name of the column containing quality scores.
        image_column: Optional name of the column identifying image filenames.
            If None, attempts auto-detection using candidates ('image_name',
            'filename', 'image', 'file', 'name', 'path'), or falls back to
            the first available column.

    Returns:
        dict[str, float]: Mapping of image filename variations to quality score.

    Raises:
        FileNotFoundError: If the CSV file does not exist.
        KeyError: If score_column or specified image_column is not found in the CSV header.
    """
    path = Path(csv_path)
    if not path.is_file():
        raise FileNotFoundError(f"Score CSV file not found: {path}")

    with open(path, mode="r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise KeyError(f"Score CSV file '{path}' is empty or missing headers.")

        fieldnames = [field.strip() for field in reader.fieldnames if field is not None]
        reader.fieldnames = fieldnames

        # Validate score column
        score_col_clean = score_column.strip()
        if score_col_clean not in fieldnames:
            lower_map = {field.lower(): field for field in fieldnames}
            if score_col_clean.lower() in lower_map:
                score_col_clean = lower_map[score_col_clean.lower()]
            else:
                raise KeyError(
                    f"Column '{score_column}' not found in CSV '{path}'. "
                    f"Available columns: {fieldnames}"
                )

        # Detect image column
        if image_column is not None:
            img_col_clean = image_column.strip()
            if img_col_clean not in fieldnames:
                lower_map = {field.lower(): field for field in fieldnames}
                if img_col_clean.lower() in lower_map:
                    img_col = lower_map[img_col_clean.lower()]
                else:
                    raise KeyError(
                        f"Image column '{image_column}' not found in CSV '{path}'. "
                        f"Available columns: {fieldnames}"
                    )
            else:
                img_col = img_col_clean
        else:
            lower_map = {field.lower(): field for field in fieldnames}
            detected: str | None = None
            for cand in IMAGE_COLUMN_CANDIDATES:
                if cand in fieldnames and cand != score_col_clean:
                    detected = cand
                    break
                if (
                    cand.lower() in lower_map
                    and lower_map[cand.lower()] != score_col_clean
                ):
                    detected = lower_map[cand.lower()]
                    break

            if detected is not None:
                img_col = detected
            else:
                other_cols = [c for c in fieldnames if c != score_col_clean]
                img_col = other_cols[0] if other_cols else fieldnames[0]

        scores: dict[str, float] = {}
        for row in reader:
            raw_img = row.get(img_col)
            if raw_img is None:
                continue
            img_str = raw_img.strip()
            if not img_str:
                continue

            raw_score = row.get(score_col_clean)
            if raw_score is None or raw_score.strip() == "":
                score_val = float("nan")
            else:
                try:
                    score_val = float(raw_score.strip())
                except ValueError:
                    score_val = float("nan")

            img_norm = img_str.replace("\\", "/")
            p = Path(img_norm)
            scores[img_str] = score_val
            scores[img_norm] = score_val
            scores[p.as_posix()] = score_val
            scores[p.name] = score_val
            scores[p.stem] = score_val

    return scores


def evaluate_csv_score(score: float, min_score: float = 0.7) -> tuple[bool, str | None]:
    """Evaluates whether an image quality score meets the minimum threshold.

    Args:
        score: Computed or parsed scalar score from the external CSV.
        min_score: Minimum allowable quality score (default: 0.7).

    Returns:
        tuple[bool, str | None]: (True, None) if score >= min_score,
            (False, 'csv_score_nan') if score is NaN,
            (False, 'low_csv_score') if score < min_score.
    """
    if math.isnan(score):
        return False, "csv_score_nan"
    if score < min_score:
        return False, "low_csv_score"
    return True, None


class CSVScoreFilterStage(QualityFilterStage):
    """Quality gate stage evaluating external image quality scores loaded from CSV."""

    name: str = "csv_score"

    def __init__(
        self,
        min_score: float = 0.7,
        scores: dict[str, float] | None = None,
    ) -> None:
        """Initializes the CSV score filter stage.

        Args:
            min_score: Minimum allowable score threshold (default: 0.7).
            scores: Pre-loaded dictionary mapping filenames or stems to scores.
        """
        self.min_score = min_score
        self.scores: dict[str, float] = scores if scores is not None else {}
        self.current_filename: str | None = None
        self.current_score: float | None = None

    def set_current_image(
        self,
        filename: str | None = None,
        score: float | None = None,
    ) -> None:
        """Sets the current image filename and/or pre-fetched score for evaluation.

        Args:
            filename: Relative path or filename of the image being evaluated.
            score: Optional explicit pre-fetched quality score.
        """
        self.current_filename = filename
        self.current_score = score

    def compute(self, image_rgb: np.ndarray, gray: np.ndarray) -> float:
        """Resolves the quality score for the current candidate image.

        Args:
            image_rgb: 3-channel RGB image (unused).
            gray: 1-channel Grayscale image (unused).

        Returns:
            float: Quality score, or float('nan') if not found or invalid.
        """
        if self.current_score is not None:
            return float(self.current_score)

        if self.current_filename is not None and self.scores:
            if self.current_filename in self.scores:
                return self.scores[self.current_filename]
            fn_norm = self.current_filename.replace("\\", "/")
            if fn_norm in self.scores:
                return self.scores[fn_norm]
            p = Path(fn_norm)
            if p.name in self.scores:
                return self.scores[p.name]
            if p.stem in self.scores:
                return self.scores[p.stem]

        return float("nan")

    def evaluate(self, metric_value: float) -> tuple[bool, str | None]:
        """Evaluates whether the metric value passes the quality threshold.

        Args:
            metric_value: The scalar quality score.

        Returns:
            tuple[bool, str | None]: (True, None) if accepted, or rejection reason.
        """
        return evaluate_csv_score(metric_value, min_score=self.min_score)

    def process(self, image_rgb: np.ndarray, gray: np.ndarray) -> FilterResult:
        """Executes score computation and threshold evaluation.

        Args:
            image_rgb: RGB image.
            gray: Grayscale image.

        Returns:
            FilterResult: Result with measured score and pass/fail decision.
        """
        val = self.compute(image_rgb, gray)
        passed, reason = self.evaluate(val)
        return FilterResult(
            stage_name=self.name,
            metric_value=val,
            passed=passed,
            rejection_reason=reason,
        )
