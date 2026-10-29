# Architecture Reference Manual

This document details the internal design, lifecycle, and extensibility points of the `dataset` package.

---

## 1. System Components Overview

The package `dataset` comprises three specialized subsystems:

```
                                  +---------------------+
                                  |     CLI Commands    |
                                  | - curate-dataset    |
                                  | - check-similarity  |
                                  +----------+----------+
                                             |
             +-------------------------------+-------------------------------+
             |                               |                               |
             v                               v                               v
+------------------------+      +------------------------+      +------------------------+
|    dataset.curator     |      |    dataset.analyzer    |      |   dataset.distortion   |
| (Reference Curation)   |      |  (Similarity / SSIM)   |      |  (Synthetic Generator)|
+-----------+------------+      +------------------------+      +------------------------+
            |
            +--> CurateConfig (config.py)
            |
            +--> DatasetCurator (curator.py)
            |       |
            |       +--> Preprocessing Layer (preprocessing.py)
            |       |       - Image loading & validation
            |       |       - Lanczos4 downsampling
            |       |       - Center cropping
            |       |       - Lossless PNG export (compression: 0)
            |       |
            |       +--> CascadedQualityGate (pipeline.py)
            |       |       - Stage 0: External CSV score ($O(1)$)
            |       |       - Stage 1: Clipping / Exposure ($O(N)$)
            |       |       - Stage 2: Contrast / Luminance Std ($O(N)$)
            |       |       - Stage 3: Shannon Entropy ($O(N)$)
            |       |       - Stage 4: Sharpness / Laplacian ($O(N \cdot K^2)$)
            |       |       - Stage 5: NIQE / PyIQA ($O(\text{Model})$)
            |       |
            |       +--> AuditLogger (audit.py)
            |               - Append-only CSV logging
```

---

## 2. Component Roles

### Subsystem: `dataset.curator`
1. **`CurateConfig` (`config.py`):**
   - Immutable frozen dataclass storing all operational thresholds, dimensions, file patterns, device preferences, and CSV metadata settings.
   - Responsible for upfront parameter validation.

2. **`DatasetCurator` (`curator.py`):**
   - High-level orchestrator service. Discovers input files, coordinates early CSV score checks, geometric normalization, quality evaluation, saves accepted artifacts, and records every file in the audit log.
   - Encapsulates batch-level fault tolerance: any unreadable or corrupted file is logged as an audit rejection rather than crashing the batch.

3. **`CascadedQualityGate` (`pipeline.py`):**
   - Holds an ordered sequence of stages satisfying `QualityFilterStage`.
   - Precomputes grayscale luminance once per crop to avoid redundant conversions.
   - Implements the short-circuiting cascade: upon the first filter failing, terminates immediately and returns a `CurationAssessment`.

4. **`AuditLogger` (`audit.py`):**
   - Thread-safe, stream-friendly CSV log writer.
   - Appends audit entries row-by-row so results are preserved even if a job is interrupted.

### Subsystem: `dataset.analyzer`
- **`similarity.py` (`compute_ssim`, `check_image_similarity`, `compare_images_ssim`):**
  - Standalone structural similarity assessment between image pairs.
  - Automatically handles differing resolutions via optional Lanczos4 resizing.
  - Provides CLI entry point `check-similarity`.

### Subsystem: `dataset.distortion`
- **`generator.py` (`DistortionGenerator`):**
  - Synthetically applies controlled degradations across 5 severity levels:
    - Gaussian blur
    - Poisson-Gaussian noise (AWGN)
    - JPEG compression
    - JPEG 2000 compression
    - Contrast decrements
    - Pink noise (1/f spatial noise)
  - Exports artifacts using lossless PNG encoding.
