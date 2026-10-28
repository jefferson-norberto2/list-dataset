"""Image input/output and geometric preprocessing module."""

from __future__ import annotations

import logging
from collections.abc import Sequence
from pathlib import Path

import cv2
import numpy as np

logger = logging.getLogger(__name__)


class ImageLoadingError(Exception):
    """Raised when an image file cannot be read or is corrupted."""


class ImageResolutionError(Exception):
    """Raised when an image is smaller than the target crop resolution."""


def load_image(path: Path) -> np.ndarray:
    """Safely loads an image from disk and converts it to canonical 8-bit RGB format.

    Supports paths with unicode characters and checks for file existence and header integrity.

    Args:
        path: Path to the image file.

    Returns:
        np.ndarray: Loaded image in RGB format with shape (H, W, 3) and dtype uint8.

    Raises:
        FileNotFoundError: If the file does not exist.
        ImageLoadingError: If the file cannot be decoded or is empty.
    """
    if not path.is_file():
        raise FileNotFoundError(f"Image file does not exist: {path}")

    # Read binary bytes first to support arbitrary file paths and verify read access
    try:
        raw_bytes = np.fromfile(str(path), dtype=np.uint8)
        if raw_bytes.size == 0:
            raise ImageLoadingError(f"Image file is empty (0 bytes): {path}")
        image_bgr = cv2.imdecode(raw_bytes, cv2.IMREAD_UNCHANGED)
    except Exception as exc:
        if isinstance(exc, ImageLoadingError):
            raise
        raise ImageLoadingError(
            f"Failed to read image bytes from {path}: {exc}"
        ) from exc

    if image_bgr is None:
        raise ImageLoadingError(
            f"Corrupted or unsupported image file could not be decoded: {path}"
        )

    # Normalize to 3-channel RGB uint8
    if image_bgr.ndim == 2:
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_GRAY2RGB)
    elif image_bgr.ndim == 3:
        channels = image_bgr.shape[2]
        if channels == 3:
            image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        elif channels == 4:
            image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGRA2RGB)
        elif channels == 1:
            image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_GRAY2RGB)
        else:
            raise ImageLoadingError(
                f"Unsupported number of image channels ({channels}) in file: {path}"
            )
    else:
        raise ImageLoadingError(
            f"Unsupported image array dimensions ({image_bgr.ndim}) in file: {path}"
        )

    # Ensure 8-bit depth
    if image_rgb.dtype != np.uint8:
        if image_rgb.dtype == np.uint16:
            image_rgb = (image_rgb / 256.0).clip(0, 255).astype(np.uint8)
        elif np.issubdtype(image_rgb.dtype, np.floating):
            image_rgb = (image_rgb * 255.0).clip(0, 255).astype(np.uint8)
        else:
            image_rgb = image_rgb.clip(0, 255).astype(np.uint8)

    return image_rgb


def downsample_shorter_edge(image: np.ndarray, target_dim: int = 512) -> np.ndarray:
    """Downsamples an image proportionally using Lanczos4 interpolation.

    The shorter dimension (height or width) is resized to match target_dim,
    preserving the aspect ratio of the image.

    Args:
        image: RGB input image of shape (H, W, 3).
        target_dim: Target resolution for the shorter dimension (default: 512).

    Returns:
        np.ndarray: Rescaled image with min(H', W') == target_dim.

    Raises:
        ValueError: If target_dim is non-positive or image has invalid shape.
        ImageResolutionError: If original image dimensions are smaller than target_dim.
    """
    if target_dim <= 0:
        raise ValueError(f"target_dim must be positive, got {target_dim}")
    if image.ndim not in (2, 3):
        raise ValueError(f"Expected 2D or 3D image array, got ndim={image.ndim}")

    h, w = image.shape[:2]
    if h < target_dim or w < target_dim:
        raise ImageResolutionError(
            f"Image dimensions ({w}x{h}) are smaller than target resolution ({target_dim})"
        )

    if h == target_dim and w == target_dim:
        return image

    scale = target_dim / min(h, w)
    if h <= w:
        new_h = target_dim
        new_w = max(target_dim, round(w * scale))
    else:
        new_w = target_dim
        new_h = max(target_dim, round(h * scale))

    # cv2.resize expects (width, height)
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
    return resized


def center_crop(
    image: np.ndarray, crop_size: tuple[int, int] = (512, 512)
) -> np.ndarray:
    """Extracts a fixed-dimension center crop from the input image.

    Args:
        image: Input image of shape (H, W, C) or (H, W).
        crop_size: Tuple (crop_height, crop_width) specifying the crop dimensions.

    Returns:
        np.ndarray: Cropped image of shape (crop_height, crop_width, C) or (crop_height, crop_width).

    Raises:
        ImageResolutionError: If image dimensions are smaller than requested crop size.
    """
    crop_h, crop_w = crop_size
    h, w = image.shape[:2]

    if h < crop_h or w < crop_w:
        raise ImageResolutionError(
            f"Image dimensions ({w}x{h}) smaller than requested crop size ({crop_w}x{crop_h})"
        )

    start_y = (h - crop_h) // 2
    start_x = (w - crop_w) // 2
    return image[start_y : start_y + crop_h, start_x : start_x + crop_w]


def preprocess_image(
    image: np.ndarray,
    target_dim: int = 512,
    crop_size: tuple[int, int] = (512, 512),
) -> np.ndarray:
    """Performs the full preprocessing pipeline: proportional Lanczos4 downsampling and center crop.

    Args:
        image: Input RGB image.
        target_dim: Target resolution for shorter dimension.
        crop_size: Final crop dimensions (height, width).

    Returns:
        np.ndarray: Cropped and downsampled image.
    """
    downsampled = downsample_shorter_edge(image, target_dim=target_dim)
    cropped = center_crop(downsampled, crop_size=crop_size)
    return cropped


def save_image_lossless_png(
    image_rgb: np.ndarray,
    destination: Path,
    compression_level: int = 0,
) -> Path:
    """Saves an RGB image as a lossless PNG to the destination path.

    Args:
        image_rgb: Input RGB image of shape (H, W, 3) with dtype uint8.
        destination: Destination file path.
        compression_level: PNG compression level (0 = uncompressed lossless, 9 = max compression).

    Returns:
        Path: Destination path of the saved file.

    Raises:
        IOError: If OpenCV fails to write the file.
    """
    destination.parent.mkdir(parents=True, exist_ok=True)
    image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    params = [cv2.IMWRITE_PNG_COMPRESSION, compression_level]
    success = cv2.imwrite(str(destination), image_bgr, params)
    if not success:
        raise OSError(f"Failed to write image to {destination}")
    return destination


def find_supported_images(
    directory: Path,
    supported_extensions: Sequence[str] = (
        ".png",
        ".jpg",
        ".jpeg",
        ".bmp",
        ".tiff",
        ".tif",
    ),
) -> list[Path]:
    """Finds all supported image files in a directory recursively.

    Args:
        directory: Directory to search for images.
        supported_extensions: File extensions to match (case-insensitive).

    Returns:
        list[Path]: Sorted list of matching file paths.
    """
    if not directory.is_dir():
        return []

    exts = {ext.lower() for ext in supported_extensions}
    matched = [
        path
        for path in directory.rglob("*")
        if path.is_file() and path.suffix.lower() in exts
    ]
    return sorted(matched)
