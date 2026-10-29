"""Poisson-Gaussian and physical sensor noise distortion algorithms."""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.ndimage import gaussian_filter

from dataset.distortion.algorithms.base import BaseDistortion


class PoissonDistortion(BaseDistortion):
    """Simulates realistic optical sensor and Poisson-Gaussian image noise.

    Supports multiple noise models including standard heteroscedastic noise,
    radial vignetting amplification, spatially correlated noise, column fixed-pattern
    noise (FPN), channel-dependent sensor noise, and discrete photon Poisson noise.

    Provides a sequential `apply` method that round-robins through all distortion
    algorithms across successive calls.
    """

    DISTORTION_METHODS: tuple[str, ...] = (
        "add_poisson_gaussian_noise",
        "add_radial_vignette_noise",
        "add_spatially_correlated_noise",
        "add_fixed_pattern_noise",
        "add_channel_dependent_noise",
        "add_true_poisson_gaussian_noise",
    )

    def apply(
        self,
        shot_noise: float = 0.01,
        read_noise: float = 0.001,
        image: np.ndarray | None = None,
        method: str | None = None,
        step: int | None = None,
        **kwargs: Any,
    ) -> np.ndarray:
        """Sequentially applies the next Poisson noise distortion in the sequence.

        On the first call, applies `add_poisson_gaussian_noise`. On the second call,
        applies `add_radial_vignette_noise`, and so forth across all 6 algorithms
        in round-robin fashion.

        Args:
            shot_noise: Signal-dependent noise scaling factor.
            read_noise: Signal-independent electronic read noise variance.
            image: Optional image array to set before applying distortion.
            method: Explicit method name to run instead of the sequential pointer.
            step: Explicit step index (0-5) to run instead of the internal counter.
            **kwargs: Extra parameters passed to the selected distortion algorithm.

        Returns:
            np.ndarray: Distorted image array as float32.

        Raises:
            ValueError: If no image is loaded or if specified method is invalid.
        """
        if image is not None:
            self.set_image(image)

        self._check_image_loaded()
        _, method_fn = self._select_method(method=method, step=step)

        return method_fn(shot_noise=shot_noise, read_noise=read_noise, **kwargs)

    def add_poisson_gaussian_noise(
        self,
        shot_noise: float = 0.01,
        read_noise: float = 0.001,
        **kwargs: Any,
    ) -> np.ndarray:
        """Applies standard heteroscedastic Poisson-Gaussian noise.

        Formulated as N(0, (I * shot_noise) + read_noise).

        Args:
            shot_noise: Signal-dependent noise scaling factor.
            read_noise: Additive electronic read noise variance.
            **kwargs: Ignored compatibility arguments.

        Returns:
            np.ndarray: Noisy image array.
        """
        self._check_image_loaded()
        assert self.img is not None
        variance_map = (self.img * shot_noise) + read_noise
        sigma_map = np.sqrt(np.clip(variance_map, 0.0, None))
        noise = np.random.normal(0.0, 1.0, self.img.shape) * sigma_map
        return self.img + noise

    def add_radial_vignette_noise(
        self,
        shot_noise: float = 0.01,
        read_noise: float = 0.001,
        edge_factor: float = 2.5,
        **kwargs: Any,
    ) -> np.ndarray:
        """Applies radial vignette noise with quadratic gain toward the periphery.

        Simulates optical lens falloff and sensor edge amplification.

        Args:
            shot_noise: Signal-dependent noise scaling factor.
            read_noise: Additive electronic read noise variance.
            edge_factor: Quadratic noise amplification factor at the image corners.
            **kwargs: Ignored compatibility arguments.

        Returns:
            np.ndarray: Noisy image array with edge-amplified variance.
        """
        self._check_image_loaded()
        assert self.img is not None
        h, w = self.img.shape[:2]
        y, x = np.ogrid[:h, :w]
        center_y, center_x = h / 2.0, w / 2.0

        max_radius = np.sqrt(center_x**2 + center_y**2)
        if max_radius == 0.0:
            max_radius = 1.0

        radius_map = np.sqrt((x - center_x) ** 2 + (y - center_y) ** 2) / max_radius
        spatial_gain = 1.0 + (edge_factor - 1.0) * (radius_map**2)
        if self.img.ndim == 3:
            spatial_gain = spatial_gain[:, :, np.newaxis]

        variance_map = ((self.img * shot_noise) + read_noise) * spatial_gain
        sigma_map = np.sqrt(np.clip(variance_map, 0.0, None))
        noise = np.random.normal(0.0, 1.0, self.img.shape) * sigma_map
        return self.img + noise

    def add_spatially_correlated_noise(
        self,
        shot_noise: float = 0.01,
        read_noise: float = 0.001,
        correlation_radius: float = 0.8,
        **kwargs: Any,
    ) -> np.ndarray:
        """Applies spatially correlated Poisson noise via Gaussian smoothing.

        Models cross-talk between neighboring sensor pixels.

        Args:
            shot_noise: Signal-dependent noise scaling factor.
            read_noise: Additive electronic read noise variance.
            correlation_radius: Gaussian blur standard deviation for noise correlation.
            **kwargs: Ignored compatibility arguments.

        Returns:
            np.ndarray: Image corrupted with spatially correlated noise.
        """
        self._check_image_loaded()
        assert self.img is not None
        variance_map = (self.img * shot_noise) + read_noise
        sigma_map = np.sqrt(np.clip(variance_map, 0.0, None))

        raw_noise = np.random.normal(0.0, 1.0, self.img.shape)

        if self.img.ndim == 3:
            filtered_noise = np.zeros_like(raw_noise)
            for c in range(self.img.shape[2]):
                filtered_noise[:, :, c] = gaussian_filter(
                    raw_noise[:, :, c], sigma=correlation_radius
                )
        else:
            filtered_noise = gaussian_filter(raw_noise, sigma=correlation_radius)

        std_dev = float(np.std(filtered_noise))
        if std_dev > 0.0:
            filtered_noise = filtered_noise / std_dev

        return self.img + (filtered_noise * sigma_map)

    def add_fixed_pattern_noise(
        self,
        shot_noise: float = 0.01,
        read_noise: float = 0.001,
        column_noise_std: float = 0.004,
        **kwargs: Any,
    ) -> np.ndarray:
        """Applies column Fixed Pattern Noise (FPN) combined with Poisson-Gaussian noise.

        Simulates column readout circuit non-uniformity common in CMOS sensors.

        Args:
            shot_noise: Signal-dependent noise scaling factor.
            read_noise: Additive electronic read noise variance.
            column_noise_std: Standard deviation of vertical column offset variations.
            **kwargs: Ignored compatibility arguments.

        Returns:
            np.ndarray: Noisy image array containing vertical stripe artifacts.
        """
        self._check_image_loaded()
        assert self.img is not None
        w = self.img.shape[1]

        column_offsets = np.random.normal(0.0, column_noise_std, (1, w))
        if self.img.ndim == 3:
            column_offsets = column_offsets[:, :, np.newaxis]

        variance_map = (self.img * shot_noise) + read_noise
        sigma_map = np.sqrt(np.clip(variance_map, 0.0, None))
        random_noise = np.random.normal(0.0, 1.0, self.img.shape) * sigma_map

        return self.img + random_noise + column_offsets

    def add_channel_dependent_noise(
        self,
        shot_noise: float = 0.01,
        read_noise: float = 0.001,
        channel_multipliers: tuple[float, float, float] = (1.6, 1.0, 2.0),
        **kwargs: Any,
    ) -> np.ndarray:
        """Applies channel-dependent noise modeling differing color filter quantum efficiencies.

        Typically blue and red sensor pixels have lower quantum efficiency and higher noise
        than green pixels under low-light conditions.

        Args:
            shot_noise: Base signal-dependent noise factor.
            read_noise: Base electronic read noise variance.
            channel_multipliers: Noise multipliers for (R, G, B) color channels.
            **kwargs: Ignored compatibility arguments.

        Returns:
            np.ndarray: RGB image with channel-specific noise variance.

        Raises:
            ValueError: If image is not a 3-channel RGB image.
        """
        self._check_image_loaded()
        assert self.img is not None
        if self.img.ndim != 3 or self.img.shape[2] != 3:
            raise ValueError("RGB image required for channel-dependent noise.")

        noisy_img = np.zeros_like(self.img)
        for c, mult in enumerate(channel_multipliers):
            channel_shot = shot_noise * mult
            channel_read = read_noise * mult

            variance = (self.img[:, :, c] * channel_shot) + channel_read
            sigma = np.sqrt(np.clip(variance, 0.0, None))
            noise = np.random.normal(0.0, 1.0, self.img[:, :, c].shape) * sigma
            noisy_img[:, :, c] = self.img[:, :, c] + noise

        return noisy_img

    def add_true_poisson_gaussian_noise(
        self,
        photon_scale: float | None = None,
        read_noise_std: float | None = None,
        shot_noise: float | None = None,
        read_noise: float | None = None,
        **kwargs: Any,
    ) -> np.ndarray:
        """Applies true discrete Poisson photon shot noise and Gaussian read noise.

        When `shot_noise` and `read_noise` are provided, computes equivalent
        `photon_scale = 1 / shot_noise` and `read_noise_std = sqrt(read_noise)`.

        Args:
            photon_scale: Expected peak photon count scaling factor (default: 1000.0).
            read_noise_std: Standard deviation of additive Gaussian read noise.
            shot_noise: Optional parameter to derive photon_scale (1 / shot_noise).
            read_noise: Optional parameter to derive read_noise_std (sqrt(read_noise)).
            **kwargs: Ignored compatibility arguments.

        Returns:
            np.ndarray: Noisy image array generated through discrete Poisson sampling.
        """
        self._check_image_loaded()
        assert self.img is not None

        if photon_scale is None:
            if shot_noise is not None and shot_noise > 0.0:
                photon_scale = 1.0 / shot_noise
            else:
                photon_scale = 1000.0

        if read_noise_std is None:
            if read_noise is not None and read_noise >= 0.0:
                read_noise_std = float(np.sqrt(read_noise))
            else:
                read_noise_std = 0.01

        expected_photons = np.clip(self.img, 0.0, None) * photon_scale
        photon_counts = np.random.poisson(expected_photons).astype(np.float32)
        noise = np.random.normal(0.0, read_noise_std * photon_scale, self.img.shape)

        noisy_img = (photon_counts + noise) / photon_scale
        return noisy_img
