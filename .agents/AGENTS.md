# Dataset Curator Agent Instructions & Rules

You are an expert Computer Vision and Python Software Engineer specializing in academic data pipelines, image quality assessment (IQA), and synthetic dataset generation.
When modifying or extending this codebase, adhere strictly to the guidelines, architectural patterns, and quality gates below.

---

## 1. Environment & Tooling Guidelines

- **Python Version:** Python 3.13+ is required.
- **Virtual Environment:** Always use the local project virtual environment:
  ```bash
  source /home/jmn/Devel/Python/list-dataset/.venv/bin/activate
  ```
  Binary location: `/home/jmn/Devel/Python/list-dataset/.venv/bin/`
- **Linting & Code Formatting:**
  - Linter: `ruff check src tests`
  - Formatter: `ruff format src tests`
  - Zero warnings or formatting discrepancies permitted.
- **Testing & Coverage:**
  - Test runner: `pytest tests/ --cov=src/dataset --cov-report=term-missing`
  - All tests must pass, with minimum **90% code coverage** maintained across all modules.

---

## 2. Project Architecture & Package Structure

The codebase is organized under the root package `dataset` with three core subsystems:

```
src/dataset/
├── curator/       # Reference dataset curation pipeline, quality gates, and audit logging
│   ├── metrics/   # Modular quality filter stages (clipping, contrast, entropy, sharpness, niqe, csv_score)
│   ├── audit.py   # Structured CSV audit logging (AuditEntry, AuditLogger)
│   ├── cli.py     # CLI entry point (curate-dataset)
│   ├── config.py  # Frozen configuration dataclass (CurateConfig)
│   ├── curator.py # DatasetCurator service layer
│   ├── pipeline.py# CascadedQualityGate and CurationAssessment
│   └── preprocessing.py # Image decoding, Lanczos4 downsampling, center crop, lossless PNG export
├── analyzer/      # Visual similarity analysis and standalone metrics
│   ├── __init__.py
│   └── similarity.py # SSIM evaluation (compute_ssim, check_image_similarity) and CLI (check-similarity)
└── distortion/    # Synthetic image degradation and distortion generation
    ├── __init__.py
    └── generator.py # DistortionGenerator (Gaussian blur, noise, JPEG, JPEG2000, contrast, pink noise)
```

---

## 3. Core Architectural Principles

1. **Protocol-Driven Modularity:**
   - Filter stages must satisfy the `QualityFilterStage` protocol defined in [`src/dataset/curator/metrics/base.py`](file:///home/jmn/Devel/Python/list-dataset/src/dataset/curator/metrics/base.py).
   - Separate metric calculation (`compute(image_rgb, gray) -> float`) from decision thresholding (`evaluate(metric_val) -> tuple[bool, str | None]`).

2. **Cascaded Early-Exit Strategy:**
   - Filters in [`CascadedQualityGate`](file:///home/jmn/Devel/Python/list-dataset/src/dataset/curator/pipeline.py) and external score checks in [`DatasetCurator`](file:///home/jmn/Devel/Python/list-dataset/src/dataset/curator/curator.py) must be ordered strictly from **lowest to highest computational cost**:
     0. External CSV Score Check ($O(1)$ hash map lookup, before decoding image)
     1. Clipping / Exposure ($O(N)$ comparisons)
     2. Contrast / Luminance Std ($O(N)$ accumulator)
     3. Shannon Entropy ($O(N)$ histogram binning)
     4. Focus Sharpness ($O(N \cdot K^2)$ Laplacian convolution)
     5. Deep / Tensor IQA ($O(\text{Model})$ PyTorch / PyIQA)
   - When a stage fails, execution must halt immediately for that candidate image. Downstream stages must NOT execute, leaving their values as `None`.

3. **Lossless Artifact Preservation:**
   - Preserved pristine images must be stored using lossless PNG encoding (`cv2.IMWRITE_PNG_COMPRESSION: 0`).
   - Output naming must follow reproducible convention: `<stem>_pristine_<target_dim>.png`.

4. **Structured Audit Logging:**
   - Every candidate image (accepted or rejected, plus corrupted files) must be recorded in the CSV audit log via [`AuditLogger`](file:///home/jmn/Devel/Python/list-dataset/src/dataset/curator/audit.py).

5. **Device Resilience & Robust Fallback:**
   - Tensor operations must automatically detect CUDA availability and fall back gracefully to CPU if GPU is unavailable or if memory limits are reached.
   - Corrupted image headers or unreadable files must be caught and logged without aborting batch processing.

---

## 4. Workflow for Adding New Features

When asked to implement new features:
1. Consult the skill [`add-curator-feature`](file:///home/jmn/Devel/Python/list-dataset/.agents/skills/add-curator-feature/SKILL.md).
2. Use the scaffolding script [`scripts/scaffold_metric.py`](file:///home/jmn/Devel/Python/list-dataset/.agents/skills/add-curator-feature/scripts/scaffold_metric.py) if introducing new quality metrics.
3. Update [`CurateConfig`](file:///home/jmn/Devel/Python/list-dataset/src/dataset/curator/config.py) and CLI arguments in [`src/dataset/curator/cli.py`](file:///home/jmn/Devel/Python/list-dataset/src/dataset/curator/cli.py).
4. Implement dedicated unit tests in `tests/test_<feature>.py`.
5. Run the verification script:
   ```bash
   bash .agents/skills/add-curator-feature/scripts/verify_feature.sh
   ```
