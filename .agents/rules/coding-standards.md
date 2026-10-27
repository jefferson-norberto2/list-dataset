# Coding Standards & Quality Guidelines

Apply these standards to all Python code within `dataset-curator`.

---

## 1. Typing & Signatures

- Use Python 3.13+ standard annotations:
  - Prefer `from __future__ import annotations`.
  - Use builtin generics: `list[str]`, `dict[str, Any]`, `tuple[int, int]`, `set[Path]`.
  - Import collections protocols from `collections.abc`: `Sequence`, `Iterable`, `Callable`, `Mapping`.
  - Use `pathlib.Path` for all filesystem paths. Do not use raw strings for path manipulation.
  - Return explicit types (`None`, `bool`, `float`, etc.) for every function and method.

---

## 2. Data Modeling & Immutability

- Use `@dataclass(frozen=True, slots=True)` for configurations and value objects that should not mutate.
- Use `@dataclass(slots=True)` for mutable containers like stateful logs or accumulator records.
- Implement post-init validation in `@dataclass` classes to reject invalid numbers, negative thresholds, and missing arguments immediately.

---

## 3. Documentation

- Follow Google-style docstrings for every module, class, method, and function.
- Include:
  - A concise 1-sentence summary.
  - Detailed explanation if necessary.
  - `Args:` block with descriptions.
  - `Returns:` block with types and physical meaning.
  - `Raises:` block detailing custom and standard exceptions.

---

## 4. Exception Handling

- Never catch bare `except:`.
- At library boundaries (e.g. `DatasetCurator.process_file`), catch specific expected errors (`ImageLoadingError`, `ImageResolutionError`, `FileNotFoundError`) first. If catching general `Exception` to prevent batch aborts, explicitly document with `# noqa: BLE001` and log the exception.
- Avoid passing exception objects redundantly to `logger.exception(...)` (as it already appends traceback).

---

## 5. Numerical & Tensor Safety

- Guard against division by zero in metrics:
  - If array size is zero: return safe baseline (e.g., `0.0`).
  - If probability distribution sum is zero: handle before logarithm.
- Guard against `NaN` values:
  - In tensor or IQA evaluations, check `math.isnan(val)` or `np.isnan(val)`.
  - Rejection reason for `NaN` should be explicit (e.g., `"niqe_nan"` or `"<metric>_nan"`).
