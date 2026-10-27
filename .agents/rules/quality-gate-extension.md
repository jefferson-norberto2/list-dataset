# Cascaded Quality Gate Extension Rules

When modifying existing stages or introducing new metrics to the cascaded quality pipeline, you MUST adhere to the following contracts:

---

## 1. Stage Implementation Contract

Every filter stage must implement the `QualityFilterStage` protocol from `dataset_curator.metrics.base`:

```python
class QualityFilterStage(Protocol):
    @property
    def name(self) -> str: ...
    def compute(self, image_rgb: np.ndarray, gray: np.ndarray) -> float: ...
    def evaluate(self, metric_value: float) -> tuple[bool, str | None]: ...
    def process(self, image_rgb: np.ndarray, gray: np.ndarray) -> FilterResult: ...
```

- `name`: Lowercase string identifier (e.g., `"brisque"`, `"chromatic_aberration"`).
- `compute`: Accepts both RGB `uint8` and Grayscale luminance `uint8`. Avoid recomputing grayscale inside the function if Grayscale is already supplied.
- `evaluate`: Returns `(True, None)` if accepted; returns `(False, "<reason_string>")` if rejected.
- `process`: Helper that executes `compute` followed by `evaluate` and returns a `FilterResult`.

---

## 2. Computational Cost Ranking

Stages MUST be inserted into `CascadedQualityGate.stages` strictly according to computational complexity:

| Tier | Typical Operation | Examples |
|:---|:---|:---|
| **Tier 1: Pixel Comparisons** | $O(N)$ elementwise masks | Clipping, Dead pixel counts |
| **Tier 2: Single-pass Statistics** | $O(N)$ accumulator loops | Luminance std, Mean brightness |
| **Tier 3: Histogram Binning** | $O(N)$ 256-bin counts | Shannon entropy, Color histograms |
| **Tier 4: Spatial Convolutions** | $O(N \cdot K^2)$ small 2D filters | Laplacian variance, Sobel gradients |
| **Tier 5: Deep / Iterative Models** | Neural networks, steerable filters | NIQE, BRISQUE, PIQE, CLIP-IQA |

Never place a higher-tier metric before a lower-tier metric.

---

## 3. Early-Exit Guarantee

The pipeline runner must stop immediately upon the first stage returning `passed=False`:
- Do NOT compute subsequent metrics.
- Uncomputed metrics must be reported as `None` in `CurationAssessment` and as empty strings (`""`) in CSV audit logs.

---

## 4. Audit Log Field Synchronization

When adding a new metric:
1. Add the metric property to `CurationAssessment` in `pipeline.py`.
2. Add the column name to `AuditLogger.FIELDNAMES` in `audit.py`.
3. Add the field to `AuditEntry` and its `to_row()` mapping.
4. Verify that existing audit logs parse backwards-compatibly.
