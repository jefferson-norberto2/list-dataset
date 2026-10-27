# Quality Metric Conventions & Guidelines

When implementing quality metrics for image dataset curation, follow these conventions:

---

## 1. Input Signatures

The standard method signature is:
```python
def compute(self, image_rgb: np.ndarray, gray: np.ndarray) -> float: ...
```
- `image_rgb`: Standard 3-channel RGB `uint8` array of shape `(H, W, 3)`.
- `gray`: 1-channel Grayscale luminance `uint8` array of shape `(H, W)` precalculated via `cv2.COLOR_RGB2GRAY`.
- If your metric only needs luminance (e.g. Laplacian variance, entropy), use `gray` directly.
- If your metric requires color channels (e.g. chromatic aberration, color cast, RGB clipping), use `image_rgb`.

---

## 2. Scalar Return & Float Conversion

- Metrics must return a native Python `float`.
- If using PyTorch or NumPy scalars (e.g. `torch.Tensor` or `np.float64`), explicitly cast using `float(...)` or `.item()`.

---

## 3. Directionality & Threshold Naming

Always document whether higher or lower values correspond to higher quality:

| Metric Type | Higher Value Means | Threshold Operator | Example |
|:---|:---|:---|:---|
| **Sharpness / Contrast** | Better Quality | Discard if `val < threshold` | Laplacian var $\ge 120.0$ |
| **Entropy / Information** | Higher Information | Discard if `val < threshold` | Entropy $\ge 7.0$ bits |
| **Noise / Distortion** | Worse Quality | Discard if `val > threshold` | NIQE $\le 3.5$ |
| **Artifacts / Clipping** | Worse Quality | Discard if `val > threshold` | Clipping $\le 0.01$ |

---

## 4. Robustness & Fallback Standards

- If computation fails or produces invalid values, return `float("nan")`.
- In `evaluate()`, check `math.isnan(metric_value)` first and return `False, "<metric_name>_nan"`.
- If utilizing CUDA, automatically verify `torch.cuda.is_available()`. Wrap CUDA operations in try-except blocks to catch out-of-memory or CUDA driver exceptions, falling back to CPU.
