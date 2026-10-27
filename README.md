# Dataset Curator

A modular, production-ready Python tool and library designed to filter and curate pristine reference image datasets from high-resolution sources (such as UHD-IQA) based on objective quality metrics from literature.

---

## Architecture & Design Principles

The dataset curation pipeline follows a **Cascaded Quality Gate (Early-Exit Strategy)**. Filters are executed in order of strictly increasing computational cost. When an image crop fails any filter, the pipeline terminates immediately for that crop, recording the failure reason in the audit log and avoiding downstream compute costs.

```
                    +---------------------------+
                    |  Raw High-Res Input Image |
                    +-------------+-------------+
                                  |
                                  v
                    +---------------------------+
                    |  Lanczos4 Downsampling    | (Shorter edge rescaled to 512px)
                    +-------------+-------------+
                                  |
                                  v
                    +---------------------------+
                    |     512x512 Center Crop   |
                    +-------------+-------------+
                                  |
      ================ Cascaded Quality Gate Cascade ================
                                  |
            [Stage 1: O(N)]       v
      +---------------------------+       FAIL
      |  Exposure / Clipping      | ---------------> [Rejection: clipping_exceeded]
      +-------------+-------------+
                    | PASS
            [Stage 2: O(N)]       v
      +---------------------------+       FAIL
      | Dynamic Range / Contrast  | ---------------> [Rejection: low_contrast]
      +-------------+-------------+
                    | PASS
            [Stage 3: O(N)]       v
      +---------------------------+       FAIL
      |  Shannon Entropy Density  | ---------------> [Rejection: low_entropy]
      +-------------+-------------+
                    | PASS
          [Stage 4: O(N*K^2)]     v
      +---------------------------+       FAIL
      | Focus Sharpness (Laplace) | ---------------> [Rejection: blurry]
      +-------------+-------------+
                    | PASS
          [Stage 5: Deep/PyIQA]   v
      +---------------------------+       FAIL
      | No-Reference IQA (NIQE)   | ---------------> [Rejection: high_niqe / nan]
      +-------------+-------------+
                    | PASS
                    v
    +-------------------------------+
    |  Save Lossless PNG (Comp: 0)  |  --> <original_stem>_pristine_512.png
    +-------------------------------+
                    |
                    v
    +-------------------------------+
    | Structured CSV Audit Logging  |  --> Records all stages & metrics
    +-------------------------------+
```

---

## Cascaded Quality Metrics & Thresholds

| Order | Metric Stage | Computational Cost | Formulation / Implementation | Default Threshold | Failure Reason |
|:---:|:---|:---:|:---|:---:|:---|
| **1** | **Clipping / Exposure** | Lowest ($O(N)$ comparisons) | Proportion of pixels where luminance $\le 1$ or $\ge 254$ | $\le 1.0\%$ (`0.01`) | `clipping_exceeded` |
| **2** | **Dynamic Range / Contrast** | Low ($O(N)$ pass) | Standard deviation of grayscale luminance $\sigma = \text{std}(Y)$ | $\ge 40.0$ | `low_contrast` |
| **3** | **Information Density** | Low-Medium (Histogram $O(N)$) | Shannon entropy $H = -\sum p_i \log_2(p_i)$ bits/pixel | $\ge 7.0$ bits/pixel | `low_entropy` |
| **4** | **Focus Sharpness** | Medium ($3 \times 3$ convolution) | Variance of the Laplacian operator $\text{Var}(\nabla^2 Y)$ (Pech-Pacheco et al., 2000) | $\ge 120.0$ | `blurry` |
| **5** | **No-Reference IQA (NIQE)** | High (PyTorch / PyIQA) | Naturalness Image Quality Evaluator (Mittal et al., 2013) on CUDA/CPU | $\le 3.5$ and not NaN | `high_niqe` / `niqe_nan` |

---

## Installation

### Prerequisites
- Python 3.13+
- CUDA-compatible GPU (optional, automatically falls back to CPU)

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

The CLI tool `curate-dataset` is installed automatically as an executable console script.

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

### CLI Arguments & Options

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
  --device              Device for tensor calculations: 'auto', 'cuda', 'cpu' (default: auto)
  --png-compression     Lossless PNG compression level 0-9 (default: 0, uncompressed lossless)
  --quiet               Disable the interactive progress bar
  --verbose, -v         Enable detailed debug logs
```

---

## Standalone Image Similarity Verification (SSIM)

The library also includes a standalone feature to assess visual similarity between two images using the **Structural Similarity Index (SSIM)** (Wang et al., 2004). This utility operates independently from the dataset curation pipeline.

### Command Line: `check-similarity`

The CLI tool `check-similarity` is installed automatically as an executable console script:

```bash
# Basic similarity check (default threshold: 0.8)
check-similarity reference.png candidate.png

# Custom threshold and automatic resizing if image dimensions differ
check-similarity reference.png candidate.png --threshold 0.85 --auto-resize

# Output results in JSON format
check-similarity reference.png candidate.png --json
```

**CLI Options:**
```
arguments:
  image1                Path to reference or first image.
  image2                Path to candidate or second image.

options:
  --threshold THRESHOLD SSIM similarity threshold between 0.0 and 1.0 (default: 0.8).
  --auto-resize         Automatically resize the second image to match the first image's dimensions.
  --win-size WIN_SIZE   Odd window size for SSIM filter (<= min(H, W)).
  --json                Output results in JSON format.
```

The command returns exit code `0` if images meet or exceed the similarity threshold, or `1` if dissimilar or upon error.

### Python API

```python
from dataset_curator import check_image_similarity, compare_images_ssim, compute_ssim

# 1. High-level evaluation with threshold decision
result = check_image_similarity("image1.png", "image2.png", threshold=0.85, auto_resize=True)
if result.is_similar:
    print(f"Images are similar! SSIM: {result.score:.4f}")
else:
    print(f"Images are dissimilar. SSIM: {result.score:.4f} < {result.threshold}")

# 2. Direct SSIM scalar calculation from file paths or NumPy arrays
score = compare_images_ssim("image1.png", "image2.png", auto_resize=True)
print(f"SSIM Score: {score:.4f}")

# 3. Direct array computation on NumPy images (RGB or Grayscale)
import cv2
img1 = cv2.imread("image1.png")
img2 = cv2.imread("image2.png")
score = compute_ssim(img1, img2, auto_resize=True)
```

---

## Python API Usage

The tool is completely modular and can be integrated directly into any Python data pipeline or PyTorch DataLoader:

```python
from pathlib import Path
from dataset_curator.config import CurateConfig
from dataset_curator.curator import DatasetCurator
from dataset_curator.pipeline import CascadedQualityGate
from dataset_curator.preprocessing import preprocess_image, load_image

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

---

## Structured CSV Audit Log Format

Every processed candidate file is recorded in a structured CSV audit file. Columns:

1. `filename`: Relative path or filename of source image.
2. `pristine`: `True` or `False`.
3. `rejection_reason`: `accepted`, `clipping_exceeded`, `low_contrast`, `low_entropy`, `blurry`, `high_niqe`, `resolution_too_small`, or `corrupted_or_unreadable`.
4. `clipping_ratio`: Proportion of pixels saturated $\le 1$ or $\ge 254$ (or empty if skipped).
5. `luminance_std`: Grayscale standard deviation (or empty if skipped).
6. `entropy`: Shannon entropy in bits/pixel (or empty if skipped).
7. `laplacian_variance`: Focus sharpness variance (or empty if skipped).
8. `niqe_score`: PyIQA NIQE score (or empty if skipped).
9. `output_path`: Absolute path of saved pristine PNG crop (or empty if rejected).

---

## Running Unit Tests & Verification

The test suite covers unit tests for every individual metric, geometric preprocessing, early-exit short-circuiting, configuration validation, and end-to-end integration:

```bash
# Run pytest with code coverage
pytest tests/ --cov=dataset_curator --cov-report=term-missing

# Run ruff code quality checks
ruff check src tests
```
