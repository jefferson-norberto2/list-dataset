"""Optical and perceptual contrast adjustment and dynamic range distortion algorithms."""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np

from dataset.distortion.algorithms.base import BaseDistortion


class ContrastDistortion(BaseDistortion):
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
        selected_method, method_fn = self._select_method(method=method, step=step)

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
