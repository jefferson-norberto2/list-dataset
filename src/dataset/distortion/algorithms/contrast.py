"""Optical and perceptual contrast adjustment and dynamic range distortion algorithms."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np


class ContrastDistortion:
    """Simulates realistic optical contrast degradation and dynamic range modifications.

    Supports multiple contrast modification models:
    - Mean-adaptive linear contrast scaling (global or independent per channel)
    - Sigmoidal / logistic non-linear contrast mapping
    - Perceptual luminance contrast adjustment in CIELAB color space
    - Dynamic range compression and bound scaling

    Provides a sequential `apply` method that round-robins across all contrast
    algorithms across successive calls, using an adaptive argument resolution
    strategy to map generic parameters to method-specific arguments.
    """

    DISTORTION_METHODS: tuple[str, ...] = (
        "change_contrast_mean_adaptive",
        "change_contrast_sigmoidal",
        "change_contrast_luminance",
        "change_contrast_range_scale",
    )

    def __init__(self, image: np.ndarray | str | Path | None = None) -> None:
        """Initializes the ContrastDistortion engine.

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
        """Returns the index (0 to 3) of the next distortion method to be applied."""
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

    @staticmethod
    def resolve_contrast_arguments(
        method_name: str,
        alpha: float = 0.5,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Resolves and adapts arguments for a target contrast algorithm.

        Maps unified/generic parameters (such as `alpha`, `factor`, or `contrast`) into
        algorithm-specific parameters, while prioritizing explicitly provided keyword arguments.

        Args:
            method_name: Target contrast method name.
            alpha: General contrast scaling factor (default: 0.5; <1.0 decreases, >1.0 increases).
            **kwargs: Explicit keyword arguments provided by caller.

        Returns:
            dict[str, Any]: Resolved keyword arguments tailored for the method.
        """
        resolved: dict[str, Any] = dict(kwargs)
        base_alpha = float(
            resolved.get(
                "alpha", resolved.get("factor", resolved.get("contrast", alpha))
            )
        )

        if method_name == "change_contrast_mean_adaptive":
            if "alpha" not in resolved:
                resolved["alpha"] = base_alpha
            if "per_channel" not in resolved:
                resolved["per_channel"] = False

        elif method_name == "change_contrast_sigmoidal":
            if "slope" not in resolved:
                # Map alpha: alpha=0.5 -> slope=5.0, alpha=1.0 -> slope=10.0
                resolved["slope"] = max(0.5, base_alpha * 10.0)
            if "midpoint" not in resolved:
                resolved["midpoint"] = 0.5

        elif method_name == "change_contrast_luminance":
            if "alpha" not in resolved:
                resolved["alpha"] = base_alpha

        elif method_name == "change_contrast_range_scale":
            if "low_bound" not in resolved or "high_bound" not in resolved:
                span = float(np.clip(base_alpha, 0.05, 1.0))
                margin = (1.0 - span) / 2.0
                if "low_bound" not in resolved:
                    resolved["low_bound"] = margin
                if "high_bound" not in resolved:
                    resolved["high_bound"] = 1.0 - margin

        return resolved

    def apply(
        self,
        alpha: float = 0.5,
        image: np.ndarray | None = None,
        method: str | None = None,
        step: int | None = None,
        **kwargs: Any,
    ) -> np.ndarray:
        """Sequentially applies the next contrast distortion in the sequence.

        Cycles round-robin across all 4 contrast algorithms on successive invocations.
        Uses an adaptive argument resolution strategy to route and scale contrast
        parameters according to the specific requirements of the selected algorithm.

        Args:
            alpha: General contrast parameter (<1.0 decreases, >1.0 increases contrast).
            image: Optional image array to set before applying distortion.
            method: Explicit method name to run instead of the sequential pointer.
            step: Explicit step index (0-3) to run instead of the internal counter.
            **kwargs: Extra parameters passed to the selected contrast algorithm
                     (e.g., slope, midpoint, low_bound, high_bound, per_channel).

        Returns:
            np.ndarray: Contrast-adjusted image array as float32 in [0.0, 1.0].

        Raises:
            ValueError: If no image is loaded or if specified method is invalid.
        """
        if image is not None:
            self.set_image(image)

        self._check_image_loaded()

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

        resolved_kwargs = self.resolve_contrast_arguments(
            selected_method,
            alpha=alpha,
            **kwargs,
        )

        return method_fn(**resolved_kwargs)

    def change_contrast_mean_adaptive(
        self,
        alpha: float = 0.5,
        per_channel: bool = False,
        **kwargs: Any,
    ) -> np.ndarray:
        """Scales contrast linearly relative to image mean luminance.

        Args:
            alpha: Contrast multiplier (<1.0 reduces contrast towards mean, >1.0 expands).
            per_channel: If True, calculates and scales around independent channel means.
            **kwargs: Ignored compatibility arguments.

        Returns:
            np.ndarray: Contrast modified image array as float32 in [0.0, 1.0].
        """
        self._check_image_loaded()
        assert self.img is not None

        if per_channel and self.img.ndim == 3:
            mean_val = np.mean(self.img, axis=(0, 1), keepdims=True)
        else:
            mean_val = float(np.mean(self.img))

        contrast_img = alpha * (self.img - mean_val) + mean_val
        return np.clip(contrast_img, 0.0, 1.0).astype(np.float32)

    def change_contrast_sigmoidal(
        self,
        slope: float = 10.0,
        midpoint: float = 0.5,
        **kwargs: Any,
    ) -> np.ndarray:
        """Applies non-linear logistic sigmoidal contrast adjustment.

        Maps values through an S-curve: steeper slope increases mid-tone contrast,
        flatter slope compresses contrast.

        Args:
            slope: Steepness parameter of the logistic curve.
            midpoint: Center point of the sigmoid transition (default: 0.5).
            **kwargs: Ignored compatibility arguments.

        Returns:
            np.ndarray: Sigmoidal contrast image array in float32 in [0.0, 1.0].
        """
        self._check_image_loaded()
        assert self.img is not None

        s = float(slope)
        m = float(midpoint)

        def sigmoid(x: np.ndarray | float) -> np.ndarray | float:
            z = np.clip(-s * (x - m), -60.0, 60.0)
            return 1.0 / (1.0 + np.exp(z))

        s_img = sigmoid(self.img)
        s_min = float(sigmoid(0.0))
        s_max = float(sigmoid(1.0))

        denom = s_max - s_min
        if denom == 0.0:
            denom = 1.0

        contrast_img = (s_img - s_min) / denom
        return np.clip(contrast_img, 0.0, 1.0).astype(np.float32)

    def change_contrast_luminance(
        self,
        alpha: float = 0.5,
        **kwargs: Any,
    ) -> np.ndarray:
        """Scales perceptual luminance contrast in CIELAB color space while preserving chromaticity.

        Modifies the L* channel around its mean without altering the a* and b* chromatic channels.

        Args:
            alpha: Luminance contrast factor (<1.0 flattens luminance, >1.0 expands).
            **kwargs: Ignored compatibility arguments.

        Returns:
            np.ndarray: Color image with adjusted luminance contrast in [0.0, 1.0].

        Raises:
            ValueError: If input image is not a 3-channel RGB image.
        """
        self._check_image_loaded()
        assert self.img is not None

        if self.img.ndim != 3 or self.img.shape[2] != 3:
            raise ValueError("RGB image required for luminance contrast adjustment.")

        lab = cv2.cvtColor(self.img.astype(np.float32), cv2.COLOR_RGB2LAB)
        l_channel = lab[:, :, 0]

        l_mean = float(np.mean(l_channel))
        lab[:, :, 0] = np.clip(alpha * (l_channel - l_mean) + l_mean, 0.0, 100.0)

        contrast_img = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
        return np.clip(contrast_img, 0.0, 1.0).astype(np.float32)

    def change_contrast_range_scale(
        self,
        low_bound: float = 0.2,
        high_bound: float = 0.8,
        **kwargs: Any,
    ) -> np.ndarray:
        """Compresses or stretches the dynamic range between specified lower and upper bounds.

        Args:
            low_bound: Lower bound of output dynamic range in [0.0, 1.0].
            high_bound: Upper bound of output dynamic range in [0.0, 1.0].
            **kwargs: Ignored compatibility arguments.

        Returns:
            np.ndarray: Dynamic range compressed image in [0.0, 1.0].
        """
        self._check_image_loaded()
        assert self.img is not None

        low = min(float(low_bound), float(high_bound))
        high = max(float(low_bound), float(high_bound))

        contrast_img = self.img * (high - low) + low
        return np.clip(contrast_img, 0.0, 1.0).astype(np.float32)

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
