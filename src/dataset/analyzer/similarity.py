"""Standalone image similarity assessment using Structural Similarity Index (SSIM)."""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from skimage.metrics import structural_similarity

from dataset.curator.preprocessing import load_image


@dataclass
class SimilarityResult:
    """Result of an SSIM similarity comparison between two images.

    Attributes:
        score: Computed SSIM score in [-1.0, 1.0], or NaN.
        is_similar: True if score is non-NaN and >= threshold, False otherwise.
        threshold: Minimum SSIM score threshold used for comparison.
        img1: Path or description of first image.
        img2: Path or description of second image.
    """

    score: float
    is_similar: bool
    threshold: float
    img1: str = ""
    img2: str = ""


def compute_ssim(
    image1: np.ndarray,
    image2: np.ndarray,
    *,
    multichannel: bool = True,
    win_size: int | None = None,
    auto_resize: bool = False,
) -> float:
    """Calculates Structural Similarity Index (SSIM) between two numpy arrays.

    Supports RGB (3-channel) and Grayscale (2-channel) images.

    Args:
        image1: Primary reference/candidate image array (H, W, 3) or (H, W).
        image2: Secondary image array to compare against image1.
        multichannel: Whether to treat 3D array as multichannel color image.
        win_size: Optional odd window size for SSIM filter (<= min(H, W)).
        auto_resize: If True, automatically resizes image2 to match image1 dimensions.

    Returns:
        float: Computed SSIM score in [-1.0, 1.0], or NaN if computation fails.

    Raises:
        ValueError: If image shapes differ and auto_resize is False.
    """
    if not isinstance(image1, np.ndarray) or not isinstance(image2, np.ndarray):
        return float("nan")

    if image1.ndim < 2 or image2.ndim < 2:
        return float("nan")

    # Shape validation and optional auto-resizing
    if image1.shape != image2.shape:
        if not auto_resize:
            raise ValueError(
                f"Image shapes do not match: {image1.shape} vs {image2.shape}. "
                "Set auto_resize=True to automatically resize image2."
            )
        # Resize image2 spatial dimensions to match image1
        h1, w1 = image1.shape[:2]
        image2 = cv2.resize(image2, (w1, h1))
        if image1.ndim == 3 and image2.ndim == 2:
            image2 = cv2.cvtColor(image2, cv2.COLOR_GRAY2RGB)
        elif image1.ndim == 2 and image2.ndim == 3:
            image2 = cv2.cvtColor(image2, cv2.COLOR_RGB2GRAY)
        if image1.shape != image2.shape:
            raise ValueError(
                f"Image shapes do not match after resize: {image1.shape} vs {image2.shape}"
            )

    h, w = image1.shape[:2]
    # Guard against images smaller than 3x3
    if h < 3 or w < 3:
        return float("nan")

    min_dim = min(h, w)
    if win_size is None:
        actual_win_size = min(7, min_dim)
        if actual_win_size % 2 == 0:
            actual_win_size -= 1
    else:
        actual_win_size = min(win_size, min_dim)
        if actual_win_size % 2 == 0:
            actual_win_size -= 1

    if actual_win_size < 3:
        return float("nan")

    channel_axis: int | None = None
    if image1.ndim == 3 and multichannel:
        channel_axis = -1

    if np.issubdtype(image1.dtype, np.floating) or np.issubdtype(
        image2.dtype, np.floating
    ):
        if np.isnan(image1).any() or np.isnan(image2).any():
            return float("nan")
        try:
            max_val = max(float(np.nanmax(image1)), float(np.nanmax(image2)))
            min_val = min(float(np.nanmin(image1)), float(np.nanmin(image2)))
            data_range = max(max_val - min_val, 1.0)
        except Exception:  # noqa: BLE001 - handle floating point edge cases
            return float("nan")
    else:
        data_range = 255.0

    try:
        score = float(
            structural_similarity(
                image1,
                image2,
                win_size=actual_win_size,
                channel_axis=channel_axis,
                data_range=data_range,
            )
        )
        if math.isnan(score):
            return float("nan")
        return max(-1.0, min(1.0, score))
    except Exception:  # noqa: BLE001 - return NaN on any computation error
        return float("nan")


def compare_images_ssim(
    img1: str | Path | np.ndarray,
    img2: str | Path | np.ndarray,
    *,
    auto_resize: bool = False,
    win_size: int | None = None,
) -> float:
    """Calculates SSIM score between two images supplied as paths or numpy arrays.

    Args:
        img1: Path to first image (str or Path) or numpy ndarray.
        img2: Path to second image (str or Path) or numpy ndarray.
        auto_resize: Whether to automatically resize img2 to match img1.
        win_size: Optional odd window size for SSIM filter.

    Returns:
        float: SSIM score in [-1.0, 1.0] or NaN if comparison fails.

    Raises:
        TypeError: If img1 or img2 is not str, Path, or np.ndarray.
        FileNotFoundError: If an image path does not exist.
        ImageLoadingError: If an image cannot be read or decoded.
    """
    if not isinstance(img1, (str, Path, np.ndarray)):
        raise TypeError(
            f"Unsupported type for img1: {type(img1)}. Expected str, Path, or np.ndarray."
        )
    if not isinstance(img2, (str, Path, np.ndarray)):
        raise TypeError(
            f"Unsupported type for img2: {type(img2)}. Expected str, Path, or np.ndarray."
        )

    im1 = load_image(Path(img1)) if isinstance(img1, (str, Path)) else img1
    im2 = load_image(Path(img2)) if isinstance(img2, (str, Path)) else img2

    return compute_ssim(im1, im2, auto_resize=auto_resize, win_size=win_size)


def check_image_similarity(
    img1: str | Path | np.ndarray,
    img2: str | Path | np.ndarray,
    *,
    threshold: float = 0.8,
    auto_resize: bool = False,
    win_size: int | None = None,
) -> SimilarityResult:
    """High-level function to check if two images are similar according to an SSIM threshold.

    Args:
        img1: Path or numpy array of first image.
        img2: Path or numpy array of second image.
        threshold: SSIM similarity threshold (default: 0.8).
        auto_resize: Whether to resize img2 to match img1 dimensions.
        win_size: Optional odd window size for SSIM filter.

    Returns:
        SimilarityResult: Data object containing score, is_similar flag, threshold, and image names.
    """
    score = compare_images_ssim(img1, img2, auto_resize=auto_resize, win_size=win_size)
    is_similar = not math.isnan(score) and (score >= threshold)

    name1 = str(img1) if isinstance(img1, (str, Path)) else f"<ndarray {img1.shape}>"
    name2 = str(img2) if isinstance(img2, (str, Path)) else f"<ndarray {img2.shape}>"

    return SimilarityResult(
        score=score,
        is_similar=is_similar,
        threshold=threshold,
        img1=name1,
        img2=name2,
    )


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point to check similarity between two images.

    Args:
        argv: Optional sequence of CLI arguments (defaults to sys.argv[1:]).

    Returns:
        int: 0 if images are similar (score >= threshold), 1 if dissimilar or on error.
    """
    parser = argparse.ArgumentParser(
        prog="check-similarity",
        description="Verify visual similarity between two images using SSIM.",
    )
    parser.add_argument("image1", type=str, help="Path to reference or first image.")
    parser.add_argument("image2", type=str, help="Path to candidate or second image.")
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.8,
        help="SSIM similarity threshold (default: 0.8).",
    )
    parser.add_argument(
        "--auto-resize",
        action="store_true",
        help="Automatically resize second image to match first image dimensions.",
    )
    parser.add_argument(
        "--win-size",
        type=int,
        default=None,
        help="Odd window size for SSIM filter (<= min(H, W)).",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output results in JSON format.",
    )

    args = parser.parse_args(argv)

    try:
        result = check_image_similarity(
            args.image1,
            args.image2,
            threshold=args.threshold,
            auto_resize=args.auto_resize,
            win_size=args.win_size,
        )
    except Exception as exc:  # noqa: BLE001 - CLI top-level error handler
        if args.json:
            print(
                json.dumps(
                    {
                        "error": str(exc),
                        "image1": args.image1,
                        "image2": args.image2,
                        "is_similar": False,
                    },
                    indent=2,
                )
            )
        else:
            print(f"Error: {exc}", file=sys.stderr)
        return 1

    if args.json:
        print(
            json.dumps(
                {
                    "image1": result.img1,
                    "image2": result.img2,
                    "score": None if math.isnan(result.score) else float(result.score),
                    "threshold": result.threshold,
                    "is_similar": result.is_similar,
                },
                indent=2,
            )
        )
    else:
        if math.isnan(result.score):
            print(
                f"DISSIMILAR: SSIM could not be computed (NaN). Threshold: {result.threshold:.4f}"
            )
        elif result.is_similar:
            print(
                f"SIMILAR: SSIM score {result.score:.4f} >= threshold {result.threshold:.4f}"
            )
        else:
            print(
                f"DISSIMILAR: SSIM score {result.score:.4f} < threshold {result.threshold:.4f}"
            )

    return 0 if result.is_similar else 1


if __name__ == "__main__":
    sys.exit(main())
