---
name: add-curator-feature
description: >-
  Use this skill when adding new features, quality filter stages, metrics, preprocessing
  operations, or CLI capabilities to the dataset-curator codebase, along with corresponding
  unit and integration tests.
---

# Adding Features to Dataset Curator

This skill provides the standard step-by-step procedure to introduce new features and quality metrics into `dataset-curator` while preserving the cascaded early-exit architecture, strict Python 3.13+ typing, and test coverage standards ($\ge 90\%$).

---

## Prerequisites & Environment

Always use the project's local virtual environment:
```bash
source /home/jmn/Devel/Python/list-dataset/.venv/bin/activate
```

---

## Procedure

### Step 1: Analyze Architecture & Computational Tier

1. Review [Architecture Reference](./references/architecture.md) and [Metric Conventions](./references/metric-conventions.md).
2. If adding a new quality filter stage:
   - Identify the computational cost tier (pixel comparison, accumulator, histogram, spatial convolution, or deep model).
   - Determine whether the metric requires RGB (`image_rgb`) or Grayscale (`gray`).
   - Determine threshold direction (higher is better vs. lower is better).

---

### Step 2: Scaffold or Implement the Feature

#### Option A: Adding a Quality Metric Stage
You can use the automated scaffolding script:
```bash
python .agents/skills/add-curator-feature/scripts/scaffold_metric.py \
  --name <metric_name> \
  --threshold <default_val> \
  --rejection-reason <reason_slug>
```
This generates:
- `src/dataset_curator/metrics/<metric_name>.py`
- `tests/test_<metric_name>.py`

Or reference the complete implementation pattern in [Color Cast Filter Example](./examples/new_metric_example.py).

#### Option B: Adding Preprocessing or I/O Features
- Implement functions in [`src/dataset_curator/preprocessing.py`](file:///home/jmn/Devel/Python/list-dataset/src/dataset_curator/preprocessing.py).
- Ensure image data remains 8-bit RGB `uint8` with canonical dimensions.

---

### Step 3: Wire the Feature into the Pipeline

1. **Update Configuration ([`config.py`](file:///home/jmn/Devel/Python/list-dataset/src/dataset_curator/config.py)):**
   - Add new attributes to `CurateConfig` with appropriate defaults.
   - Add validation inside `CurateConfig.__post_init__()`.

2. **Register in Pipeline ([`pipeline.py`](file:///home/jmn/Devel/Python/list-dataset/src/dataset_curator/pipeline.py)):**
   - Insert the new stage into `CascadedQualityGate.stages` at the correct computational cost position.
   - Add the metric field to `CurationAssessment`.

3. **Update Audit Logging ([`audit.py`](file:///home/jmn/Devel/Python/list-dataset/src/dataset_curator/audit.py)):**
   - Add column to `AuditLogger.FIELDNAMES`.
   - Add field to `AuditEntry` and map it in `AuditEntry.to_row()`.

4. **Expose in CLI ([`cli.py`](file:///home/jmn/Devel/Python/list-dataset/src/dataset_curator/cli.py)):**
   - Add corresponding command-line flags in `build_parser()`.
   - Pass arguments to `CurateConfig` in `main()`.

5. **Export in Package `__init__.py`:**
   - Update [`src/dataset_curator/metrics/__init__.py`](file:///home/jmn/Devel/Python/list-dataset/src/dataset_curator/metrics/__init__.py) and [`src/dataset_curator/__init__.py`](file:///home/jmn/Devel/Python/list-dataset/src/dataset_curator/__init__.py).

---

### Step 4: Write Unit & Integration Tests

1. Create or complete `tests/test_<feature_name>.py`. See [Example Test Reference](./examples/new_metric_test_example.py).
2. Test both the core mathematical function and the filter stage class.
3. Test edge cases:
   - Zero values, maximum values, flat images.
   - Acceptance boundary and rejection boundary.
   - `NaN` handling and recovery.
4. Update integration tests in [`tests/test_cli.py`](file:///home/jmn/Devel/Python/list-dataset/tests/test_cli.py) to exercise the new CLI flags and early-exit behavior.

---

### Step 5: Verification & Validation

Run the automated verification script:
```bash
bash .agents/skills/add-curator-feature/scripts/verify_feature.sh
```

Ensure:
- `ruff check src tests` passes with 0 warnings.
- `ruff format --check src tests` reports clean formatting.
- `pytest` passes 100% of tests with $\ge 90\%$ code coverage.
