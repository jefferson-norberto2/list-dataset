# Dataset Curator & Vision Processing Toolkit

A modular, production-ready Python tool and library designed to filter and curate pristine reference image datasets from high-resolution sources (such as UHD-IQA), verify image similarity via SSIM, and generate realistic synthetic distortions across multiple severity levels.

---

## Repository Structure

The project is structured under the `dataset` namespace package with three core subsystems:

```
src/dataset/
├── curator/                  # Reference dataset curation pipeline & quality gate
│   ├── metrics/              # Modular filter stages (clipping, contrast, entropy, sharpness, niqe, csv_score)
│   ├── audit.py              # Structured CSV audit logging (AuditEntry, AuditLogger)
│   ├── cli.py                # Console script CLI: curate-dataset
│   ├── config.py             # Immutable configuration settings (CurateConfig)
│   ├── curator.py            # DatasetCurator batch processing engine
│   ├── pipeline.py           # CascadedQualityGate and CurationAssessment
│   └── preprocessing.py      # Lanczos4 downsampling, center crop, lossless PNG I/O
├── analyzer/                 # Visual similarity analysis and comparison
│   ├── __init__.py           # Exports: check_image_similarity, compare_images_ssim, compute_ssim
│   └── similarity.py         # SSIM implementation and console script CLI: check-similarity
└── distortion/               # Synthetic image degradation and distortion generation
    ├── __init__.py           # Exports: DistortionGenerator
    └── generator.py          # 6 degradation types (blur, noise, jpeg, jp2k, contrast, pink noise)
```

---

## 1. Dataset Curation Pipeline (`dataset.curator`)

### Architecture & Cascaded Early-Exit Strategy

The dataset curation pipeline follows a **Cascaded Quality Gate (Early-Exit Strategy)**. Filters are executed in order of strictly increasing computational cost. When an image fails any filter stage, the pipeline terminates immediately for that candidate, recording the failure reason in the audit log and avoiding downstream compute costs.

```
                    +---------------------------+
                    |  Raw High-Res Input Image |
                    +-------------+-------------+
                                  |
            [Stage 0: O(1)]       v       FAIL
      +---------------------------+ ---------------> [Rejection: low_csv_score / not_found]
      | External CSV Quality Score| (Checked before decoding image)
      +-------------+-------------+
                    | PASS
                    v
      +---------------------------+
      |  Lanczos4 Downsampling    | (Shorter edge rescaled to target_dim)
      +-------------+-------------+
                    |
                    v
      +---------------------------+
      |        Center Crop        | (e.g. 512x512 or 1800x1800)
      +-------------+-------------+
                    |
      ================ Cascaded Quality Gate Cascade ================
                    |
            [Stage 1: O(N)]       v       FAIL
      +---------------------------+ ---------------> [Rejection: clipping_exceeded]
      |  Exposure / Saturated Pxls|
      +-------------+-------------+
                    | PASS
            [Stage 2: O(N)]       v       FAIL
      +---------------------------+ ---------------> [Rejection: low_contrast]
      | Dynamic Range / Contrast  |
      +-------------+-------------+
                    | PASS
            [Stage 3: O(N)]       v       FAIL
      +---------------------------+ ---------------> [Rejection: low_entropy]
      |  Shannon Entropy Density  |
      +-------------+-------------+
                    | PASS
          [Stage 4: O(N*K^2)]     v       FAIL
      +---------------------------+ ---------------> [Rejection: blurry]
      | Focus Sharpness (Laplace) |
      +-------------+-------------+
                    | PASS
          [Stage 5: Deep/PyIQA]   v       FAIL
      +---------------------------+ ---------------> [Rejection: high_niqe / niqe_nan]
      | No-Reference IQA (NIQE)   |
      +-------------+-------------+
                    | PASS
                    v
      +-------------------------------+
      |  Save Lossless PNG (Comp: 0)  |  --> <original_stem>_pristine_<target_dim>.png
      +-------------------------------+
                    |
                    v
      +-------------------------------+
      | Structured CSV Audit Logging  |  --> Records all evaluated stages & metrics
      +-------------------------------+
```

### Cascaded Quality Metrics & Thresholds

| Order | Metric Stage | Computational Cost | Formulation / Implementation | Default Threshold | Failure Reason |
|:---:|:---|:---:|:---|:---:|:---|
| **0** | **External CSV Score** | $O(1)$ lookup | MOS or subjective quality score from external metadata CSV | $\ge 0.70$ (configurable) | `low_csv_score` / `csv_score_not_found` |
| **1** | **Clipping / Exposure** | Lowest ($O(N)$ comparisons) | Proportion of pixels where luminance $\le 1$ or $\ge 254$ | $\le 1.0\%$ (`0.01`) | `clipping_exceeded` |
| **2** | **Dynamic Range / Contrast** | Low ($O(N)$ pass) | Standard deviation of grayscale luminance $\sigma = \text{std}(Y)$ | $\ge 40.0$ | `low_contrast` |
| **3** | **Information Density** | Low-Medium (Histogram $O(N)$) | Shannon entropy $H = -\sum p_i \log_2(p_i)$ bits/pixel | $\ge 7.0$ bits/pixel | `low_entropy` |
| **4** | **Focus Sharpness** | Medium ($3 \times 3$ convolution) | Variance of the Laplacian operator $\text{Var}(\nabla^2 Y)$ (Pech-Pacheco et al., 2000) | $\ge 120.0$ | `blurry` |
| **5** | **No-Reference IQA (NIQE)** | High (PyTorch / PyIQA) | Naturalness Image Quality Evaluator (Mittal et al., 2013) on CUDA/CPU | $\le 3.5$ and not NaN | `high_niqe` / `niqe_nan` |

---

## 2. Image Similarity Verification (`dataset.analyzer`)

Provides standalone pairwise similarity assessment between images using the **Structural Similarity Index (SSIM)** (Wang et al., 2004), completely decoupled from the curation pipeline.

---

## 3. Synthetic Distortion Generator (`dataset.distortion`)

Applies 6 controlled optical and digital degradations across 5 standardized severity levels:
1. **Gaussian Blur:** Spatial low-pass filtering across varying $\sigma$ kernels.
2. **Poisson-Gaussian Noise (AWGN):** Combined shot noise (signal-dependent Poisson) and read noise (Gaussian).
3. **JPEG Compression:** DCT-based lossy quantization.
4. **JPEG 2000 Compression:** Discrete Wavelet Transform (DWT) compression with variable bit-rate ratios.
5. **Contrast Decrements:** Global contrast reduction centered around neutral gray ($0.5$).
6. **Pink Noise ($1/f$):** 2D spatial frequency pink noise synthesized via 2D FFT.

---

## Installation

### Prerequisites
- Python 3.13+
- CUDA-compatible GPU (optional; automatically falls back to CPU)

### Setup
```bash
# Clone the repository
git clone <repository_url>
cd list-dataset

# Install in editable mode
pip install -e .

# Or install with dev/test dependencies
pip install -e ".[dev]"
```

---

## Command Line Interface (CLI)

### 1. `curate-dataset`

```bash
curate-dataset \
  --input-dir /path/to/raw_uhd_images \
  --output-dir /path/to/pristine_reference_dataset \
  --audit-csv /path/to/audit_log.csv \
  --target-dim 512 \
  --crop-size 512 512 \
  --max-clipping-ratio 0.01 \
  --min-luminance-std 40.0 \
  --min-entropy 7.0 \
  --min-laplacian-var 120.0 \
  --max-niqe 3.5 \
  --device auto
```

#### CLI Options:
```
options:
  -h, --help            Show this help message and exit
  --input-dir, -i       Input directory containing candidate images (required)
  --output-dir, -o      Output directory to store pristine crops (required)
  --audit-csv, -a       Custom path for the CSV audit log (default: <output-dir>/curation_audit.csv)
  --target-dim          Target resolution for shorter dimension downsampling (default: 512)
  --crop-size           Crop dimensions [HEIGHT, WIDTH] (default: 512 512)
  --max-clipping-ratio  Max proportion of saturated pixels (default: 0.01 = 1.0%)
  --min-luminance-std   Min standard deviation of grayscale luminance (default: 40.0)
  --min-entropy         Min Shannon entropy in bits/pixel (default: 7.0)
  --min-laplacian-var   Min variance of Laplacian for focus sharpness (default: 120.0)
  --max-niqe            Max allowable NIQE score (default: 3.5)
  --csv-score-path      Optional path to external CSV containing image quality scores
  --csv-score-column    Column name in external CSV containing the quality scores
  --min-csv-score       Minimum allowable quality score from external CSV (default: 0.7)
  --device              Device for tensor calculations: 'auto', 'cuda', 'cpu' (default: auto)
  --png-compression     Lossless PNG compression level 0-9 (default: 0, uncompressed lossless)
  --quiet               Disable the interactive progress bar
  --verbose, -v         Enable detailed debug logs
```

### 2. `check-similarity`

```bash
# Basic similarity check (default threshold: 0.8)
check-similarity reference.png candidate.png

# Custom threshold with automatic resizing if dimensions differ
check-similarity reference.png candidate.png --threshold 0.85 --auto-resize

# JSON formatted output
check-similarity reference.png candidate.png --json
```

---

## Python API Usage

### Curation Pipeline

```python
from pathlib import Path
from dataset.curator import CurateConfig, DatasetCurator, CascadedQualityGate, load_image, preprocess_image

# 1. Custom Configuration
config = CurateConfig(
    input_dir=Path("./data/raw_images"),
    output_dir=Path("./data/pristine_crops"),
    target_dim=512,
    crop_size=(512, 512),
    max_clipping_ratio=0.01,
    min_luminance_std=40.0,
    min_entropy=7.0,
    min_laplacian_var=120.0,
    max_niqe=3.5,
    device="auto",
)

# 2. Run Batch Curation Service
curator = DatasetCurator(config=config)
summary = curator.run_curation()
print(f"Accepted {summary['accepted_pristine']} out of {summary['total_images']} images.")

# 3. Or use individual components directly
image_rgb = load_image(Path("sample.jpg"))
crop_rgb = preprocess_image(image_rgb, target_dim=512, crop_size=(512, 512))

gate = CascadedQualityGate(config=config)
assessment = gate.evaluate(crop_rgb)

if assessment.is_pristine:
    print(f"Pristine! NIQE score: {assessment.niqe_score:.2f}")
else:
    print(f"Rejected: {assessment.rejection_reason}")
```

### Image Similarity Assessment

```python
from dataset.analyzer import check_image_similarity, compare_images_ssim, compute_ssim

# 1. High-level evaluation with threshold decision
result = check_image_similarity("image1.png", "image2.png", threshold=0.85, auto_resize=True)
if result.is_similar:
    print(f"Images are similar! SSIM: {result.score:.4f}")
else:
    print(f"Images are dissimilar. SSIM: {result.score:.4f} < {result.threshold}")

# 2. Direct SSIM scalar calculation from file paths
score = compare_images_ssim("image1.png", "image2.png", auto_resize=True)
print(f"SSIM Score: {score:.4f}")
```

### Synthetic Distortion Generation

```python
from dataset.distortion import DistortionGenerator

generator = DistortionGenerator()
generator.change_image("path/to/pristine_image.png")

# 1. Apply Gaussian Blur
blurred = generator.add_gaussian_blur(sigma=2.0)
generator.save_output(blurred, "blur", level=3)

# 2. Apply Poisson-Gaussian Noise
noisy = generator.add_poisson_gaussian_noise(shot_noise=0.03, read_noise=0.002)
generator.save_output(noisy, "awgn", level=3)

# 3. Apply JPEG Compression
jpeg_dist = generator.add_jpeg_compression(quality=10)
generator.save_output(jpeg_dist, "jpeg", level=3)

# 4. Apply JPEG 2000 Compression
jp2k_dist = generator.add_jpeg2000_compression(compression_ratio=7)
generator.save_output(jp2k_dist, "jpeg2000", level=3)

# 5. Apply Contrast Reduction
low_contrast = generator.change_contrast(alpha=0.4)
generator.save_output(low_contrast, "contrast", level=3)

# 6. Apply 1/f Pink Noise
pink = generator.add_pink_noise(intensity=0.12, spatial_scale=1.0)
generator.save_output(pink, "fnoise", level=3)
```

---

## Structured CSV Audit Log Format

Every candidate image evaluated by `dataset.curator` is recorded in a CSV audit file. Columns:

1. `filename`: Relative path or filename of source image.
2. `pristine`: `True` or `False`.
3. `rejection_reason`: `accepted`, `low_csv_score`, `csv_score_not_found`, `csv_score_nan`, `clipping_exceeded`, `low_contrast`, `low_entropy`, `blurry`, `high_niqe`, `resolution_too_small`, `corrupted_or_unreadable`, or `file_not_found`.
4. `clipping_ratio`: Proportion of pixels saturated $\le 1$ or $\ge 254$ (empty if skipped by early exit).
5. `luminance_std`: Grayscale standard deviation (empty if skipped by early exit).
6. `entropy`: Shannon entropy in bits/pixel (empty if skipped by early exit).
7. `laplacian_variance`: Focus sharpness variance (empty if skipped by early exit).
8. `niqe_score`: PyIQA NIQE score (empty if skipped by early exit).
9. `csv_score`: External CSV MOS / quality score (empty if feature disabled).
10. `output_path`: Absolute path of saved pristine PNG crop (empty if rejected).

---

## Running Unit Tests & Quality Verification

```bash
# Run pytest with code coverage
pytest tests/ --cov=src/dataset --cov-report=term-missing

# Run ruff code quality checks
ruff check src tests

# Run ruff formatting check
ruff format --check src tests

# Or run the all-in-one verification script
bash .agents/skills/add-curator-feature/scripts/verify_feature.sh
```
