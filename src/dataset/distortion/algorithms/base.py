"""Abstract base class and lifecycle management for distortion algorithms."""

from __future__ import annotations

import abc
from collections.abc import Callable
from pathlib import Path
from typing import Any

import cv2
import numpy as np


class BaseDistortion(abc.ABC):
    """Abstract base class for image distortion algorithms.

    Manages image lifecycle, format normalization, sequential round-robin execution,
    and lossless artifact persistence.

    Subclasses must define `DISTORTION_METHODS` and implement `apply`.
    """

    DISTORTION_METHODS: tuple[str, ...] = ()

    def __init__(self, image: np.ndarray | str | Path | None = None) -> None:
        """Initializes the distortion engine.

        Args:
            image: Optional input image as a NumPy array or file path.
        """
        self.img: np.ndarray | None = None
        self.img_uint8: np.ndarray | None = None
        self.height: int | None = None
        self.width: int | None = None
        self.channels: int | None = None
        self.img_name: str = "image"
        self._sequence_index: int = 0
        self.last_applied_method: str | None = None

        if image is not None:
            if isinstance(image, (str, Path)):
                self.change_image(image)
            elif isinstance(image, np.ndarray):
                self.set_image(image)
            else:
                raise TypeError(f"Unsupported image type: {type(image).__name__}")

    @property
    def current_sequence_step(self) -> int:
        """Returns the index of the next distortion method to be applied."""
        return self._sequence_index

    def reset_sequence(self) -> None:
        """Resets the sequential application counter to the first distortion method."""
        self._sequence_index = 0
        self.last_applied_method = None

    def change_image(self, new_image_path: str | Path) -> None:
        """Loads and normalizes a new image from disk.

        Args:
            new_image_path: Path to the image file to load.

        Raises:
            ValueError: If the file cannot be found or decoded.
        """
        path_str = str(new_image_path)
        img_bgr = cv2.imread(path_str)
        if img_bgr is None:
            raise ValueError(f"Image not found at path: {new_image_path}")

        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        self.set_image(img_rgb)
        self.img_name = Path(new_image_path).stem

    def set_image(self, image: np.ndarray) -> None:
        """Sets the working image from an in-memory NumPy array.

        Converts input to float32 normalized in [0.0, 1.0].

        Args:
            image: Grayscale or RGB image array (uint8 or float).

        Raises:
            ValueError: If image array is empty or has invalid dimensionality.
        """
        if not isinstance(image, np.ndarray) or image.size == 0:
            raise ValueError("Invalid image array provided.")

        if image.ndim not in (2, 3):
            raise ValueError(f"Expected 2D or 3D image array, got ndim={image.ndim}")

        if np.issubdtype(image.dtype, np.integer):
            self.img = image.astype(np.float32) / 255.0
            self.img_uint8 = np.clip(image, 0, 255).astype(np.uint8)
        else:
            self.img = np.clip(image, 0.0, 1.0).astype(np.float32)
            self.img_uint8 = (self.img * 255.0).astype(np.uint8)

        self.height, self.width = self.img.shape[:2]
        self.channels = self.img.shape[2] if self.img.ndim == 3 else 1

    def _check_image_loaded(self) -> None:
        """Verifies that an image is loaded before running operations.

        Raises:
            ValueError: If no image has been loaded.
        """
        if self.img is None:
            raise ValueError(
                "No image loaded. Use change_image() or set_image() to load an image."
            )

    def _select_method(
        self,
        method: str | None = None,
        step: int | None = None,
    ) -> tuple[str, Callable[..., np.ndarray]]:
        """Selects the distortion method callable and manages sequence progression.

        Args:
            method: Explicit method name to run instead of the sequential pointer.
            step: Explicit step index to run instead of the internal counter.

        Returns:
            tuple[str, Callable]: The selected method name and the callable method function.

        Raises:
            ValueError: If specified method is not found or not in DISTORTION_METHODS.
        """
        if method is not None:
            if not hasattr(self, method):
                raise ValueError(
                    f"Unknown distortion method '{method}'. Available methods: {self.DISTORTION_METHODS}"
                )
            selected_method = method
        elif step is not None:
            selected_method = self.DISTORTION_METHODS[
                step % len(self.DISTORTION_METHODS)
            ]
        else:
            selected_method = self.DISTORTION_METHODS[self._sequence_index]
            self._sequence_index = (self._sequence_index + 1) % len(
                self.DISTORTION_METHODS
            )

        self.last_applied_method = selected_method
        method_fn = getattr(self, selected_method)
        return selected_method, method_fn

    @abc.abstractmethod
    def apply(
        self,
        *args: Any,
        image: np.ndarray | None = None,
        method: str | None = None,
        step: int | None = None,
        **kwargs: Any,
    ) -> np.ndarray:
        """Applies a distortion method to the image. Must be implemented by subclasses."""

    def save_output(
        self,
        distorted_img: np.ndarray,
        output_path: str | Path,
    ) -> Path:
        """Saves a distorted image array using lossless PNG compression.

        Args:
            distorted_img: Image array in float32 [0.0, 1.0] or uint8.
            output_path: Path where the PNG artifact will be stored.

        Returns:
            Path: Absolute path to the saved image file.
        """
        dest = Path(output_path)
        dest.parent.mkdir(parents=True, exist_ok=True)

        img_clipped = np.clip(distorted_img, 0.0, 1.0)
        img_out = (img_clipped * 255.0).astype(np.uint8)

        # Convert RGB to BGR for OpenCV
        if img_out.ndim == 3 and img_out.shape[2] == 3:
            img_to_write = cv2.cvtColor(img_out, cv2.COLOR_RGB2BGR)
        else:
            img_to_write = img_out

        cv2.imwrite(str(dest), img_to_write, [cv2.IMWRITE_PNG_COMPRESSION, 0])
        return dest.resolve()
