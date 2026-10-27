# Architecture Reference Manual

This document details the internal design, lifecycle, and extensibility points of `dataset-curator`.

---

## 1. System Components Overview

```
                      +-------------------+
                      |   CLI (cli.py)    |
                      +---------+---------+
                                |
                                v
                   +------------------------+
                   |  CurateConfig (config) |
                   +------------+-----------+
                                |
                                v
               +--------------------------------+
               |    DatasetCurator (curator)    |
               +---+------------------------+---+
                   |                        |
                   v                        v
        +---------------------+   +---------------------+
        | Preprocessing Layer |   | CascadedQualityGate |
        | (preprocessing.py)  |   |   (pipeline.py)     |
        +---------------------+   +----------+----------+
                                             |
                                 +-----------+-----------+
                                 | QualityFilterStages   |
                                 | - clipping            |
                                 | - contrast            |
                                 | - entropy             |
                                 | - sharpness           |
                                 | - niqe                |
                                 | - [new_metrics...]    |
                                 +-----------+-----------+
                                             |
                                             v
                                  +---------------------+
                                  | AuditLogger (audit) |
                                  +---------------------+
```

---

## 2. Component Roles

1. **`CurateConfig` (`config.py`):**
   - Immutable frozen dataclass storing all operational thresholds, dimensions, file patterns, and device preferences.
   - Responsible for upfront parameter validation.

2. **`DatasetCurator` (`curator.py`):**
   - High-level orchestrator service. Discovers input files, triggers decoding and geometric normalization, executes quality evaluation, saves accepted artifacts, and records every file in the audit log.
   - Encapsulates batch-level fault tolerance: any unreadable or corrupted file is logged as an audit rejection rather than crashing the batch.

3. **`CascadedQualityGate` (`pipeline.py`):**
   - Holds an ordered sequence of stages satisfying `QualityFilterStage`.
   - Precomputes grayscale luminance once per crop to avoid redundant conversions.
   - Implements the short-circuiting cascade: upon the first filter failing, terminates immediately and returns a `CurationAssessment`.

4. **`AuditLogger` (`audit.py`):**
   - Thread-safe, stream-friendly CSV log writer.
   - Appends audit entries row-by-row so results are preserved even if a job is interrupted.
