"""Unit tests verifying the cascaded quality gate and early-exit strategy."""

from __future__ import annotations

import numpy as np
import pytest

from dataset_curator.metrics.base import FilterResult, QualityFilterStage
from dataset_curator.pipeline import CascadedQualityGate


class MockStage(QualityFilterStage):
    """Mock filter stage recording execution calls."""

    def __init__(
        self, name: str, passed: bool, value: float, reason: str | None = None
    ) -> None:
        self._name = name
        self._passed = passed
        self._value = value
        self._reason = reason
        self.call_count = 0

    @property
    def name(self) -> str:
        return self._name

    def compute(self, image_rgb: np.ndarray, gray: np.ndarray) -> float:
        return self._value

    def evaluate(self, metric_value: float) -> tuple[bool, str | None]:
        return self._passed, self._reason

    def process(self, image_rgb: np.ndarray, gray: np.ndarray) -> FilterResult:
        self.call_count += 1
        return FilterResult(
            stage_name=self._name,
            metric_value=self._value,
            passed=self._passed,
            rejection_reason=self._reason,
        )


class TestCascadedEarlyExit:
    """Tests verifying the cascaded short-circuit execution order."""

    def test_first_stage_failure_skips_all_subsequent(self) -> None:
        """When stage 1 fails, stages 2 through 5 must never be executed."""
        s1 = MockStage("clipping", passed=False, value=0.05, reason="clipping_exceeded")
        s2 = MockStage("contrast", passed=True, value=50.0)
        s3 = MockStage("entropy", passed=True, value=7.5)
        s4 = MockStage("sharpness", passed=True, value=200.0)
        s5 = MockStage("niqe", passed=True, value=2.8)

        gate = CascadedQualityGate(stages=[s1, s2, s3, s4, s5])
        dummy_rgb = np.full((100, 100, 3), 128, dtype=np.uint8)

        assessment = gate.evaluate(dummy_rgb)

        assert assessment.is_pristine is False
        assert assessment.rejection_reason == "clipping_exceeded"
        assert assessment.clipping_ratio == pytest.approx(0.05)
        assert assessment.luminance_std is None
        assert assessment.entropy is None
        assert assessment.laplacian_var is None
        assert assessment.niqe_score is None

        # Verify execution counts
        assert s1.call_count == 1
        assert s2.call_count == 0
        assert s3.call_count == 0
        assert s4.call_count == 0
        assert s5.call_count == 0

    def test_intermediate_stage_failure_halts_cascade(self) -> None:
        """When stage 3 (entropy) fails, stages 1 & 2 run, but 4 & 5 do not."""
        s1 = MockStage("clipping", passed=True, value=0.002)
        s2 = MockStage("contrast", passed=True, value=45.0)
        s3 = MockStage("entropy", passed=False, value=5.1, reason="low_entropy")
        s4 = MockStage("sharpness", passed=True, value=150.0)
        s5 = MockStage("niqe", passed=True, value=3.0)

        gate = CascadedQualityGate(stages=[s1, s2, s3, s4, s5])
        dummy_rgb = np.full((100, 100, 3), 128, dtype=np.uint8)

        assessment = gate.evaluate(dummy_rgb)

        assert assessment.is_pristine is False
        assert assessment.rejection_reason == "low_entropy"
        assert assessment.clipping_ratio == pytest.approx(0.002)
        assert assessment.luminance_std == pytest.approx(45.0)
        assert assessment.entropy == pytest.approx(5.1)
        assert assessment.laplacian_var is None
        assert assessment.niqe_score is None

        assert s1.call_count == 1
        assert s2.call_count == 1
        assert s3.call_count == 1
        assert s4.call_count == 0
        assert s5.call_count == 0

    def test_all_stages_pass(self) -> None:
        """When all stages pass, image is marked pristine and all metrics are populated."""
        s1 = MockStage("clipping", passed=True, value=0.001)
        s2 = MockStage("contrast", passed=True, value=55.0)
        s3 = MockStage("entropy", passed=True, value=7.3)
        s4 = MockStage("sharpness", passed=True, value=180.0)
        s5 = MockStage("niqe", passed=True, value=2.9)

        gate = CascadedQualityGate(stages=[s1, s2, s3, s4, s5])
        dummy_rgb = np.full((100, 100, 3), 128, dtype=np.uint8)

        assessment = gate.evaluate(dummy_rgb)

        assert assessment.is_pristine is True
        assert assessment.rejection_reason == "accepted"
        assert assessment.clipping_ratio == pytest.approx(0.001)
        assert assessment.luminance_std == pytest.approx(55.0)
        assert assessment.entropy == pytest.approx(7.3)
        assert assessment.laplacian_var == pytest.approx(180.0)
        assert assessment.niqe_score == pytest.approx(2.9)
