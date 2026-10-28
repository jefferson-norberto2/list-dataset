#!/usr/bin/env python3
"""Utility script to scaffold boilerplate for a new quality metric and its unit tests."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


METRIC_TEMPLATE = '''"""{metric_title} evaluation filter stage."""

from __future__ import annotations

import numpy as np

from dataset.curator.metrics import FilterResult, QualityFilterStage


def compute_{metric_name}(image_rgb: np.ndarray, gray: np.ndarray) -> float:
    """Calculates the {metric_name} score.

    Args:
        image_rgb: 3-channel RGB image uint8 of shape (H, W, 3).
        gray: 1-channel Grayscale luminance uint8 of shape (H, W).

    Returns:
        float: Computed {metric_name} score.
    """
    if gray.size == 0:
        return 0.0

    # TODO: Implement metric computation
    score = float(np.mean(gray))
    return score


class {class_name}FilterStage(QualityFilterStage):
    """Quality gate stage evaluating {metric_title}."""

    name: str = "{metric_name}"

    def __init__(self, threshold: float = {default_threshold}) -> None:
        """Initializes the {metric_title} stage.

        Args:
            threshold: Quality threshold.
        """
        self.threshold = threshold

    def compute(self, image_rgb: np.ndarray, gray: np.ndarray) -> float:
        """Computes {metric_title}."""
        return compute_{metric_name}(image_rgb, gray)

    def evaluate(self, metric_value: float) -> tuple[bool, str | None]:
        """Evaluates whether metric value passes threshold."""
        # TODO: Adjust condition (< or >) depending on whether higher or lower is better
        if metric_value < self.threshold:
            return False, "{rejection_reason}"
        return True, None

    def process(self, image_rgb: np.ndarray, gray: np.ndarray) -> FilterResult:
        """Executes computation and threshold check."""
        val = self.compute(image_rgb, gray)
        passed, reason = self.evaluate(val)
        return FilterResult(
            stage_name=self.name,
            metric_value=val,
            passed=passed,
            rejection_reason=reason,
        )
'''

TEST_TEMPLATE = '''"""Unit tests for {metric_title} metric and filter stage."""

from __future__ import annotations

import numpy as np
import pytest

from dataset_curator.metrics.{metric_name} import (
    {class_name}FilterStage,
    compute_{metric_name},
)


class Test{class_name}Metric:
    """Tests for compute_{metric_name}."""

    def test_computation_baseline(self) -> None:
        """Verify metric calculation on baseline synthetic image."""
        gray = np.full((100, 100), 128, dtype=np.uint8)
        rgb = np.stack([gray] * 3, axis=-1)

        val = compute_{metric_name}(rgb, gray)
        assert isinstance(val, float)


class Test{class_name}FilterStage:
    """Tests for {class_name}FilterStage evaluation."""

    def test_pass_condition(self) -> None:
        """Stage passes when metric satisfies threshold."""
        stage = {class_name}FilterStage(threshold=100.0)
        gray = np.full((100, 100), 150, dtype=np.uint8)
        rgb = np.stack([gray] * 3, axis=-1)

        result = stage.process(rgb, gray)
        assert result.passed is True
        assert result.rejection_reason is None

    def test_fail_condition(self) -> None:
        """Stage rejects when metric violates threshold."""
        stage = {class_name}FilterStage(threshold=100.0)
        gray = np.full((100, 100), 50, dtype=np.uint8)
        rgb = np.stack([gray] * 3, axis=-1)

        result = stage.process(rgb, gray)
        assert result.passed is False
        assert result.rejection_reason == "{rejection_reason}"
'''


def main() -> int:
    parser = argparse.ArgumentParser(description="Scaffold a new quality filter stage and test file.")
    parser.add_argument("--name", required=True, help="Metric snake_case name (e.g., brisque, color_cast).")
    parser.add_argument("--threshold", type=float, default=50.0, help="Default quality threshold.")
    parser.add_argument("--rejection-reason", default=None, help="Rejection string if failed.")

    args = parser.parse_args()
    metric_name = args.name.lower().strip()
    class_name = "".join(part.capitalize() for part in metric_name.split("_"))
    metric_title = " ".join(part.capitalize() for part in metric_name.split("_"))
    rejection_reason = args.rejection_reason or f"low_{metric_name}"

    repo_root = Path(__file__).resolve().parents[4]
    metric_file = repo_root / "src" / "dataset_curator" / "metrics" / f"{metric_name}.py"
    test_file = repo_root / "tests" / f"test_{metric_name}.py"

    if metric_file.exists():
        print(f"Error: {metric_file} already exists!", file=sys.stderr)
        return 1
    if test_file.exists():
        print(f"Error: {test_file} already exists!", file=sys.stderr)
        return 1

    metric_code = METRIC_TEMPLATE.format(
        metric_name=metric_name,
        class_name=class_name,
        metric_title=metric_title,
        default_threshold=args.threshold,
        rejection_reason=rejection_reason,
    )
    test_code = TEST_TEMPLATE.format(
        metric_name=metric_name,
        class_name=class_name,
        metric_title=metric_title,
        rejection_reason=rejection_reason,
    )

    metric_file.write_text(metric_code, encoding="utf-8")
    test_file.write_text(test_code, encoding="utf-8")

    print(f"Created metric stage: {metric_file.relative_to(repo_root)}")
    print(f"Created unit test  : {test_file.relative_to(repo_root)}")
    print("\nNext steps:")
    print(f"1. Implement computation in {metric_file.name}")
    print("2. Export stage in src/dataset_curator/metrics/__init__.py")
    print("3. Wire into CascadedQualityGate in src/dataset_curator/pipeline.py")
    print("4. Add configuration parameters in src/dataset_curator/config.py and cli.py")
    print("5. Run .agents/skills/add-curator-feature/scripts/verify_feature.sh")
    return 0


if __name__ == "__main__":
    sys.exit(main())
