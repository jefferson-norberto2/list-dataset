"""Unit tests verifying the cascaded quality gate and early-exit strategy."""

from __future__ import annotations

import numpy as np
import pytest

from dataset_curator.config import CurateConfig
from dataset_curator.metrics.base import FilterResult, QualityFilterStage
from dataset_curator.metrics.csv_score import CSVScoreFilterStage
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

    def test_csv_score_gate_cascade(self, tmp_path) -> None:
        """When CSV score is enabled, it acts as the first stage and halts cascade on failure."""
        scores_csv = tmp_path / "scores.csv"
        scores_csv.write_text(
            "image,score\npass.png,0.85\nfail.png,0.40\nexact_pass.png,0.7000\nexact_fail.png,0.6999\n",
            encoding="utf-8",
        )

        cfg = CurateConfig(
            input_dir=tmp_path,
            output_dir=tmp_path,
            csv_score_path=scores_csv,
            csv_score_column="score",
            min_csv_score=0.7,
        )

        gate = CascadedQualityGate(config=cfg)
        assert len(gate.stages) == 6
        assert isinstance(gate.stages[0], CSVScoreFilterStage)

        dummy_rgb = np.full((100, 100, 3), 128, dtype=np.uint8)

        # Evaluate failing image
        assessment_fail = gate.evaluate(dummy_rgb, filename="fail.png")
        assert assessment_fail.is_pristine is False
        assert assessment_fail.rejection_reason == "low_csv_score"
        assert assessment_fail.csv_score == pytest.approx(0.40)
        # Downstream metrics must be skipped
        assert assessment_fail.clipping_ratio is None
        assert assessment_fail.luminance_std is None
        assert assessment_fail.entropy is None
        assert assessment_fail.laplacian_var is None
        assert assessment_fail.niqe_score is None

        # Boundary pass: 0.7000
        assessment_boundary_pass = gate.evaluate(dummy_rgb, filename="exact_pass.png")
        assert assessment_boundary_pass.csv_score == pytest.approx(0.7000)
        # CSV stage passed, reached clipping
        assert assessment_boundary_pass.clipping_ratio is not None

        # Boundary fail: 0.6999
        assessment_boundary_fail = gate.evaluate(dummy_rgb, filename="exact_fail.png")
        assert assessment_boundary_fail.is_pristine is False
        assert assessment_boundary_fail.rejection_reason == "low_csv_score"
        assert assessment_boundary_fail.csv_score == pytest.approx(0.6999)
        assert assessment_boundary_fail.clipping_ratio is None

        # Evaluate passing image (flat synthetic image will fail later stage like contrast/entropy)
        assessment_pass = gate.evaluate(dummy_rgb, filename="pass.png")
        assert assessment_pass.csv_score == pytest.approx(0.85)
        # Reached clipping (or later), clipping was evaluated
        assert assessment_pass.clipping_ratio is not None

    def test_pipeline_inactive_when_only_one_parameter_provided(self, tmp_path) -> None:
        """Pipeline initializes without CSVScoreFilterStage when only one parameter is set."""
        scores_csv = tmp_path / "scores.csv"
        scores_csv.write_text("image,score\npass.png,0.85\n", encoding="utf-8")

        # Only path provided
        cfg_path_only = CurateConfig(
            input_dir=tmp_path,
            output_dir=tmp_path,
            csv_score_path=scores_csv,
        )
        gate_path_only = CascadedQualityGate(config=cfg_path_only)
        assert len(gate_path_only.stages) == 5
        assert all(
            not isinstance(s, CSVScoreFilterStage) for s in gate_path_only.stages
        )

        # Only column provided
        cfg_col_only = CurateConfig(
            input_dir=tmp_path,
            output_dir=tmp_path,
            csv_score_column="score",
        )
        gate_col_only = CascadedQualityGate(config=cfg_col_only)
        assert len(gate_col_only.stages) == 5
        assert all(not isinstance(s, CSVScoreFilterStage) for s in gate_col_only.stages)

    def test_early_exit_guarantees_downstream_stages_never_called(self) -> None:
        """When CSV score check fails, downstream stages are guaranteed never to be executed."""
        call_log: list[str] = []

        class DummyCSVStage(QualityFilterStage):
            name = "csv_score"

            def compute(self, image_rgb: np.ndarray, gray: np.ndarray) -> float:
                call_log.append("csv_score.compute")
                return 0.45

            def evaluate(self, metric_value: float) -> tuple[bool, str | None]:
                call_log.append("csv_score.evaluate")
                return False, "low_csv_score"

            def process(self, image_rgb: np.ndarray, gray: np.ndarray) -> FilterResult:
                val = self.compute(image_rgb, gray)
                passed, reason = self.evaluate(val)
                return FilterResult(self.name, val, passed, reason)

        class ExplodingStage(QualityFilterStage):
            def __init__(self, stage_name: str) -> None:
                self._name = stage_name

            @property
            def name(self) -> str:
                return self._name

            def compute(self, image_rgb: np.ndarray, gray: np.ndarray) -> float:
                raise AssertionError(
                    f"Stage '{self._name}' compute must not be called!"
                )

            def evaluate(self, metric_value: float) -> tuple[bool, str | None]:
                raise AssertionError(
                    f"Stage '{self._name}' evaluate must not be called!"
                )

            def process(self, image_rgb: np.ndarray, gray: np.ndarray) -> FilterResult:
                raise AssertionError(
                    f"Stage '{self._name}' process must not be called!"
                )

        stages = [
            DummyCSVStage(),
            ExplodingStage("clipping"),
            ExplodingStage("contrast"),
            ExplodingStage("entropy"),
            ExplodingStage("sharpness"),
            ExplodingStage("niqe"),
        ]

        gate = CascadedQualityGate(stages=stages)
        dummy_rgb = np.zeros((32, 32, 3), dtype=np.uint8)

        # evaluate should halt after DummyCSVStage without invoking any ExplodingStage
        assessment = gate.evaluate(dummy_rgb)

        assert assessment.is_pristine is False
        assert assessment.rejection_reason == "low_csv_score"
        assert assessment.csv_score == pytest.approx(0.45)
        assert call_log == ["csv_score.compute", "csv_score.evaluate"]
        assert assessment.clipping_ratio is None
        assert assessment.luminance_std is None
        assert assessment.entropy is None
        assert assessment.laplacian_var is None
        assert assessment.niqe_score is None
