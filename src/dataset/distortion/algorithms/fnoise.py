"""1/f fractal pink noise distortion algorithms."""

from __future__ import annotations

from typing import Any

import numpy as np

from dataset.distortion.algorithms.base import BaseDistortion


class FnoiseDistortion(BaseDistortion):
    """Simulates realistic 1/f fractal (pink) noise and power-law spectral degradations.

    Supports multiple colored/fractal noise models:
    - Correlated pink noise with controllable inter-channel chrominance/luminance correlation
    - Anisotropic oriented pink noise with directional spectral scaling
    - Bandlimited pink noise restricted to specific spatial frequency octaves
    - Multiplicative pink noise modulating signal intensity

    Provides a sequential `apply` method that round-robins across all pink noise
    algorithms across successive calls, using an adaptive argument resolution
    strategy to map generic parameters to method-specific arguments.
    """

    DISTORTION_METHODS: tuple[str, ...] = (
        "add_correlated_pink_noise",
        "add_anisotropic_pink_noise",
        "add_bandlimited_pink_noise",
        "add_multiplicative_pink_noise",
    )

    @staticmethod
    def resolve_fnoise_arguments(
        method_name: str,
        intensity: float = 0.05,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Resolves and adapts arguments for a target pink noise algorithm.

        Maps unified/generic parameters (such as `intensity`, `noise`, or `strength`) into
        algorithm-specific parameters, while prioritizing explicitly provided keyword arguments.

        Args:
            method_name: Target pink noise method name.
            intensity: General noise amplitude scaling factor (default: 0.05).
            **kwargs: Explicit keyword arguments provided by caller.

        Returns:
            dict[str, Any]: Resolved keyword arguments tailored for the method.
        """
        resolved: dict[str, Any] = dict(kwargs)
        base_intensity = float(
            resolved.get(
                "intensity",
                resolved.get("noise", resolved.get("strength", intensity)),
            )
        )
        base_spatial_scale = float(
            resolved.get("spatial_scale", resolved.get("scale", 1.0))
        )

        if method_name == "add_correlated_pink_noise":
            if "intensity" not in resolved:
                resolved["intensity"] = base_intensity
            if "spatial_scale" not in resolved:
                resolved["spatial_scale"] = base_spatial_scale
            if "color_correlation" not in resolved:
                resolved["color_correlation"] = 0.85

        elif method_name == "add_anisotropic_pink_noise":
            if "intensity" not in resolved:
                resolved["intensity"] = base_intensity
            if "scale_u" not in resolved:
                resolved["scale_u"] = base_spatial_scale * 1.5
            if "scale_v" not in resolved:
                resolved["scale_v"] = base_spatial_scale * 0.5
            if "angle_deg" not in resolved:
                resolved["angle_deg"] = resolved.get("angle", 0.0)

        elif method_name == "add_bandlimited_pink_noise":
            if "intensity" not in resolved:
                resolved["intensity"] = base_intensity
            if "spatial_scale" not in resolved:
                resolved["spatial_scale"] = base_spatial_scale
            if "f_min" not in resolved:
                resolved["f_min"] = 0.02
            if "f_max" not in resolved:
                resolved["f_max"] = 0.4

        elif method_name == "add_multiplicative_pink_noise":
            if "intensity" not in resolved:
                resolved["intensity"] = base_intensity * 2.0
            if "spatial_scale" not in resolved:
                resolved["spatial_scale"] = base_spatial_scale

        return resolved

    def apply(
        self,
        intensity: float = 0.05,
        image: np.ndarray | None = None,
        method: str | None = None,
        step: int | None = None,
        **kwargs: Any,
    ) -> np.ndarray:
        """Sequentially applies the next pink noise distortion in the sequence.

        Cycles round-robin across all 4 pink noise algorithms on successive invocations.
        Uses an adaptive argument resolution strategy to route and scale noise parameters
        according to the specific requirements of the selected algorithm.

        Args:
            intensity: General noise amplitude parameter (default: 0.05).
            image: Optional image array to set before applying distortion.
            method: Explicit method name to run instead of the sequential pointer.
            step: Explicit step index (0-3) to run instead of the internal counter.
            **kwargs: Extra parameters passed to the selected algorithm
                     (e.g., spatial_scale, color_correlation, scale_u, scale_v, angle_deg, f_min, f_max).

        Returns:
            np.ndarray: Distorted noisy image array as float32 in [0.0, 1.0].

        Raises:
            ValueError: If no image is loaded or if specified method is invalid.
        """
        if image is not None:
            self.set_image(image)

        self._check_image_loaded()
        selected_method, method_fn = self._select_method(method=method, step=step)

        resolved_kwargs = self.resolve_fnoise_arguments(
            selected_method,
            intensity=intensity,
            **kwargs,
        )

        return method_fn(**resolved_kwargs)

    def add_correlated_pink_noise(
        self,
        intensity: float = 0.05,
        spatial_scale: float = 1.0,
        color_correlation: float = 0.85,
        **kwargs: Any,
    ) -> np.ndarray:
        """Generates 1/f pink noise with inter-channel correlation.

        Luminance noise is shared across channels while chrominance noise is generated
        independently, blended according to `color_correlation`.

        Args:
            intensity: Noise amplitude scaling factor.
            spatial_scale: Exponent beta for power-law 1/f^beta profile.
            color_correlation: Degree of shared noise across color channels in [0.0, 1.0].
            **kwargs: Ignored compatibility arguments.

        Returns:
            np.ndarray: Noisy image array in float32 in [0.0, 1.0].
        """
        self._check_image_loaded()
        assert self.img is not None
        assert self.height is not None and self.width is not None

        def generate_pink_component(
            shape: tuple[int, int], scale_exp: float
        ) -> np.ndarray:
            rows, cols = shape
            u = np.fft.fftfreq(rows)
            v = np.fft.fftfreq(cols)
            u_grid, v_grid = np.meshgrid(u, v, indexing="ij")

            f = np.sqrt(u_grid**2 + v_grid**2)
            f[0, 0] = 1.0

            filter_spec = 1.0 / (f**scale_exp + 1e-5)
            filter_spec[0, 0] = 0.0

            white_spec = np.fft.fft2(np.random.normal(0.0, 1.0, shape))
            pink = np.fft.ifft2(white_spec * filter_spec).real
            std_val = float(np.std(pink))
            if std_val > 0.0:
                return (pink - float(np.mean(pink))) / std_val
            return pink

        h, w = self.height, self.width
        base_noise = generate_pink_component((h, w), float(spatial_scale))
        corr = float(np.clip(color_correlation, 0.0, 1.0))

        if self.img.ndim == 2:
            noise_layer = base_noise
        else:
            noise_layer = np.zeros_like(self.img)
            for ch in range(self.channels or 1):
                indep_noise = generate_pink_component((h, w), float(spatial_scale))
                noise_layer[:, :, ch] = corr * base_noise + (1.0 - corr) * indep_noise

        noisy_img = self.img + (noise_layer * float(intensity))
        return np.clip(noisy_img, 0.0, 1.0).astype(np.float32)

    def add_anisotropic_pink_noise(
        self,
        intensity: float = 0.05,
        scale_u: float = 1.5,
        scale_v: float = 0.5,
        angle_deg: float = 0.0,
        **kwargs: Any,
    ) -> np.ndarray:
        """Applies oriented anisotropic 1/f noise with elliptical frequency spectrum.

        Args:
            intensity: Noise amplitude scaling factor.
            scale_u: Major-axis frequency scale factor.
            scale_v: Minor-axis frequency scale factor.
            angle_deg: Rotation angle in degrees of principal noise orientation.
            **kwargs: Ignored compatibility arguments.

        Returns:
            np.ndarray: Noisy image array in float32 in [0.0, 1.0].
        """
        self._check_image_loaded()
        assert self.img is not None
        assert self.height is not None and self.width is not None

        def directional_pink_2d(shape: tuple[int, int]) -> np.ndarray:
            rows, cols = shape
            u = np.fft.fftfreq(rows)
            v = np.fft.fftfreq(cols)
            u_grid, v_grid = np.meshgrid(u, v, indexing="ij")

            theta = np.deg2rad(angle_deg)
            cos_t, sin_t = np.cos(theta), np.sin(theta)
            u_rot = u_grid * cos_t - v_grid * sin_t
            v_rot = u_grid * sin_t + v_grid * cos_t

            su = max(1e-4, float(scale_u))
            sv = max(1e-4, float(scale_v))
            f_aniso = np.sqrt((u_rot * su) ** 2 + (v_rot * sv) ** 2)
            f_aniso[0, 0] = 1.0

            scale = 1.0 / (f_aniso + 1e-5)
            scale[0, 0] = 0.0

            white_spec = np.fft.fft2(np.random.normal(0.0, 1.0, shape))
            pink = np.fft.ifft2(white_spec * scale).real
            std_val = float(np.std(pink))
            if std_val > 0.0:
                return (pink - float(np.mean(pink))) / std_val
            return pink

        shape = (self.height, self.width)
        if self.img.ndim == 2:
            noise_layer = directional_pink_2d(shape)
        else:
            noise_layer = np.zeros_like(self.img)
            for ch in range(self.channels or 1):
                noise_layer[:, :, ch] = directional_pink_2d(shape)

        noisy_img = self.img + (noise_layer * float(intensity))
        return np.clip(noisy_img, 0.0, 1.0).astype(np.float32)

    def add_bandlimited_pink_noise(
        self,
        intensity: float = 0.05,
        spatial_scale: float = 1.0,
        f_min: float = 0.02,
        f_max: float = 0.4,
        **kwargs: Any,
    ) -> np.ndarray:
        """Applies bandpass-filtered pink noise restricted to specific spatial frequencies.

        Args:
            intensity: Noise amplitude scaling factor.
            spatial_scale: Spectral power-law falloff exponent.
            f_min: Lower cutoff normalized spatial frequency in [0.0, 0.5].
            f_max: Upper cutoff normalized spatial frequency in [0.0, 0.5].
            **kwargs: Ignored compatibility arguments.

        Returns:
            np.ndarray: Bandlimited noisy image array in [0.0, 1.0].
        """
        self._check_image_loaded()
        assert self.img is not None
        assert self.height is not None and self.width is not None

        def bandlimited_pink_2d(shape: tuple[int, int]) -> np.ndarray:
            rows, cols = shape
            u = np.fft.fftfreq(rows)
            v = np.fft.fftfreq(cols)
            u_grid, v_grid = np.meshgrid(u, v, indexing="ij")

            f = np.sqrt(u_grid**2 + v_grid**2)
            safe_f = np.where(f == 0.0, 1.0, f)
            sp_scale = max(0.1, float(spatial_scale))
            scale = 1.0 / (safe_f**sp_scale)

            f_lo = min(float(f_min), float(f_max))
            f_hi = max(float(f_min), float(f_max))
            band_mask = (f >= f_lo) & (f <= f_hi)
            scale = scale * band_mask
            scale[0, 0] = 0.0

            white_spec = np.fft.fft2(np.random.normal(0.0, 1.0, shape))
            pink = np.fft.ifft2(white_spec * scale).real
            std_val = float(np.std(pink))
            if std_val > 0.0:
                return (pink - float(np.mean(pink))) / std_val
            return pink

        shape = (self.height, self.width)
        if self.img.ndim == 2:
            noise_layer = bandlimited_pink_2d(shape)
        else:
            noise_layer = np.zeros_like(self.img)
            for ch in range(self.channels or 1):
                noise_layer[:, :, ch] = bandlimited_pink_2d(shape)

        noisy_img = self.img + (noise_layer * float(intensity))
        return np.clip(noisy_img, 0.0, 1.0).astype(np.float32)

    def add_multiplicative_pink_noise(
        self,
        intensity: float = 0.1,
        spatial_scale: float = 1.0,
        **kwargs: Any,
    ) -> np.ndarray:
        """Applies multiplicative pink noise modulation (speckle-like texture).

        Computes: I' = I * (1 + intensity * noise).

        Args:
            intensity: Multiplicative noise scaling factor.
            spatial_scale: Power-law spectral exponent.
            **kwargs: Ignored compatibility arguments.

        Returns:
            np.ndarray: Modulated noisy image array in [0.0, 1.0].
        """
        self._check_image_loaded()
        assert self.img is not None
        assert self.height is not None and self.width is not None

        rows, cols = self.height, self.width
        u = np.fft.fftfreq(rows)
        v = np.fft.fftfreq(cols)
        u_grid, v_grid = np.meshgrid(u, v, indexing="ij")

        f = np.sqrt(u_grid**2 + v_grid**2)
        f[0, 0] = 1.0
        sp_scale = max(0.1, float(spatial_scale))
        scale = 1.0 / (f**sp_scale + 1e-5)
        scale[0, 0] = 0.0

        if self.img.ndim == 2:
            white_spec = np.fft.fft2(np.random.normal(0.0, 1.0, (rows, cols)))
            pink = np.fft.ifft2(white_spec * scale).real
            std_val = float(np.std(pink))
            noise_layer = (
                (pink - float(np.mean(pink))) / std_val if std_val > 0.0 else pink
            )
        else:
            noise_layer = np.zeros_like(self.img)
            for ch in range(self.channels or 1):
                white_spec = np.fft.fft2(np.random.normal(0.0, 1.0, (rows, cols)))
                pink = np.fft.ifft2(white_spec * scale).real
                std_val = float(np.std(pink))
                noise_layer[:, :, ch] = (
                    (pink - float(np.mean(pink))) / std_val if std_val > 0.0 else pink
                )

        noisy_img = self.img * (1.0 + float(intensity) * noise_layer)
        return np.clip(noisy_img, 0.0, 1.0).astype(np.float32)
