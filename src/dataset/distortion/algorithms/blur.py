"""Optical and synthetic blur distortion algorithms."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np


class BlurDistortion:
    """Simulates realistic optical, directional, spatial, and chromatic blur distortions.

    Supports multiple blur models:
    - Isotropic / Anisotropic Gaussian blur
    - Directional / Motion Gaussian blur with arbitrary orientation angle
    - Radial spatially varying blur simulating lens defocus falloff towards edges
    - Chromatic aberration Gaussian blur with channel-dependent dispersion

    Provides a sequential `apply` method that round-robins across all blur
    algorithms across successive calls, using an adaptive argument resolution
    strategy to map generic parameters to method-specific arguments.
    """

    DISTORTION_METHODS: tuple[str, ...] = (
        "add_gaussian_blur",
        "add_directional_gaussian_blur",
        "add_radial_spatially_varying_blur",
        "add_chromatic_gaussian_blur",
    )

    def __init__(self, image: np.ndarray | str | Path | None = None) -> None:
        """Initializes the BlurDistortion engine.

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
    def resolve_blur_arguments(
        method_name: str,
        sigma: float = 2.0,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Resolves and adapts arguments for a target blur algorithm.

        Maps unified/generic parameters (such as `sigma`) into algorithm-specific
        parameters, while prioritizing explicitly provided keyword arguments.

        Args:
            method_name: Target blur method name.
            sigma: General blur scaling factor (default: 2.0).
            **kwargs: Explicit keyword arguments provided by caller.

        Returns:
            dict[str, Any]: Resolved keyword arguments tailored for the method.
        """
        resolved: dict[str, Any] = dict(kwargs)

        if method_name == "add_gaussian_blur":
            if "sigma_x" not in resolved:
                resolved["sigma_x"] = resolved.get("sigma", sigma)
            if "sigma_y" not in resolved:
                resolved["sigma_y"] = resolved.get("sigma_y", None)

        elif method_name == "add_directional_gaussian_blur":
            base_sigma = resolved.get("sigma", sigma)
            if "sigma_x" not in resolved:
                resolved["sigma_x"] = base_sigma
            if "sigma_y" not in resolved:
                resolved["sigma_y"] = max(0.2, float(base_sigma) * 0.25)
            if "angle_deg" not in resolved:
                resolved["angle_deg"] = resolved.get("angle", 45.0)

        elif method_name == "add_radial_spatially_varying_blur":
            base_sigma = resolved.get("sigma", sigma)
            if "center_sigma" not in resolved:
                resolved["center_sigma"] = max(0.1, float(base_sigma) * 0.25)
            if "edge_sigma" not in resolved:
                resolved["edge_sigma"] = max(0.5, float(base_sigma) * 1.75)

        elif method_name == "add_chromatic_gaussian_blur":
            base_sigma = resolved.get("sigma", sigma)
            if "sigmas" not in resolved:
                s = float(base_sigma)
                resolved["sigmas"] = (s * 1.1, s * 0.9, s * 1.4)

        return resolved

    def apply(
        self,
        sigma: float = 2.0,
        image: np.ndarray | None = None,
        method: str | None = None,
        step: int | None = None,
        **kwargs: Any,
    ) -> np.ndarray:
        """Sequentially applies the next blur distortion in the sequence.

        Cycles round-robin across all 4 blur algorithms on successive invocations.
        Uses an adaptive argument resolution strategy to route and scale blur
        parameters according to the specific requirements of the selected algorithm.

        Args:
            sigma: General blur severity parameter used across algorithms.
            image: Optional image array to set before applying distortion.
            method: Explicit method name to run instead of the sequential pointer.
            step: Explicit step index (0-3) to run instead of the internal counter.
            **kwargs: Extra parameters passed to the selected blur algorithm
                     (e.g., sigma_x, sigma_y, angle_deg, center_sigma, edge_sigma, sigmas).

        Returns:
            np.ndarray: Blurred image array as float32 in [0.0, 1.0].

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

        resolved_kwargs = self.resolve_blur_arguments(
            selected_method,
            sigma=sigma,
            **kwargs,
        )

        return method_fn(**resolved_kwargs)

    def add_gaussian_blur(
        self,
        sigma_x: float = 2.0,
        sigma_y: float | None = None,
        force_kernel: int | tuple[int, int] | list[int] | None = None,
        **kwargs: Any,
    ) -> np.ndarray:
        """Applies standard isotropic or anisotropic Gaussian blur.

        Args:
            sigma_x: Standard deviation in the horizontal direction.
            sigma_y: Standard deviation in the vertical direction (defaults to sigma_x).
            force_kernel: Optional explicit odd kernel size (e.g. 5 or (5, 7)).
            **kwargs: Ignored compatibility arguments.

        Returns:
            np.ndarray: Gaussian blurred image array in float32.
        """
        self._check_image_loaded()
        assert self.img is not None

        sx = max(1e-4, float(sigma_x))
        sy = sx if sigma_y is None else max(1e-4, float(sigma_y))

        if force_kernel is not None:
            if isinstance(force_kernel, (tuple, list)):
                kx, ky = int(force_kernel[0]), int(force_kernel[1])
            else:
                kx = ky = int(force_kernel)

            kx = kx if kx % 2 != 0 else kx + 1
            ky = ky if ky % 2 != 0 else ky + 1
            kernel_size = (max(1, kx), max(1, ky))
        else:
            kernel_size = (0, 0)

        blurred = cv2.GaussianBlur(self.img, kernel_size, sigmaX=sx, sigmaY=sy)
        return blurred.astype(np.float32)

    def add_directional_gaussian_blur(
        self,
        sigma_x: float = 3.0,
        sigma_y: float = 0.8,
        angle_deg: float = 45.0,
        **kwargs: Any,
    ) -> np.ndarray:
        """Applies oriented directional Gaussian blur simulating linear motion blur.

        Synthesizes an oriented 2D Gaussian kernel rotated by `angle_deg`.

        Args:
            sigma_x: Major axis standard deviation along the direction of motion.
            sigma_y: Minor axis standard deviation transverse to motion direction.
            angle_deg: Angle of motion blur in degrees counter-clockwise from horizontal.
            **kwargs: Ignored compatibility arguments.

        Returns:
            np.ndarray: Directionally blurred image array in float32.
        """
        self._check_image_loaded()
        assert self.img is not None

        sx = max(1e-4, float(sigma_x))
        sy = max(1e-4, float(sigma_y))

        max_sigma = max(sx, sy)
        k_radius = max(1, int(np.ceil(3.0 * max_sigma)))

        y, x = np.mgrid[-k_radius : k_radius + 1, -k_radius : k_radius + 1]

        theta = np.deg2rad(angle_deg)
        cos_t = np.cos(theta)
        sin_t = np.sin(theta)

        x_rot = x * cos_t + y * sin_t
        y_rot = -x * sin_t + y * cos_t

        kernel = np.exp(
            -0.5 * ((x_rot**2 / (sx**2 + 1e-6)) + (y_rot**2 / (sy**2 + 1e-6)))
        )
        k_sum = kernel.sum()
        if k_sum > 0.0:
            kernel /= k_sum

        blurred = cv2.filter2D(self.img, -1, kernel)
        return blurred.astype(np.float32)

    def add_radial_spatially_varying_blur(
        self,
        center_sigma: float = 0.5,
        edge_sigma: float = 3.5,
        **kwargs: Any,
    ) -> np.ndarray:
        """Applies spatially varying radial blur modeling lens defocus falloff.

        Blur smoothly increases from `center_sigma` at the optical center
        to `edge_sigma` at the outer corners following a quadratic radial curve.

        Args:
            center_sigma: Blur standard deviation at image center.
            edge_sigma: Blur standard deviation at outermost corners.
            **kwargs: Ignored compatibility arguments.

        Returns:
            np.ndarray: Spatially varying blurred image array in float32.
        """
        self._check_image_loaded()
        assert self.img is not None

        c_sig = max(1e-4, float(center_sigma))
        e_sig = max(1e-4, float(edge_sigma))

        sharp_img = cv2.GaussianBlur(self.img, (0, 0), sigmaX=c_sig, sigmaY=c_sig)
        blurred_img = cv2.GaussianBlur(self.img, (0, 0), sigmaX=e_sig, sigmaY=e_sig)

        h, w = self.img.shape[:2]
        center_y, center_x = h / 2.0, w / 2.0
        y, x = np.ogrid[:h, :w]
        max_radius = np.sqrt(center_x**2 + center_y**2)
        if max_radius == 0.0:
            max_radius = 1.0

        radial_mask = np.sqrt((x - center_x) ** 2 + (y - center_y) ** 2) / max_radius
        alpha = np.clip(radial_mask**2, 0.0, 1.0)
        if self.img.ndim == 3:
            alpha = alpha[:, :, np.newaxis]

        result = (1.0 - alpha) * sharp_img + alpha * blurred_img
        return result.astype(np.float32)

    def add_chromatic_gaussian_blur(
        self,
        sigmas: tuple[float, float, float] | list[float] = (2.2, 1.0, 2.8),
        **kwargs: Any,
    ) -> np.ndarray:
        """Applies chromatic aberration blur with independent dispersion per color channel.

        Args:
            sigmas: Sequence of 3 blur sigmas corresponding to (R, G, B) channels.
            **kwargs: Ignored compatibility arguments.

        Returns:
            np.ndarray: RGB image with channel-specific blur.

        Raises:
            ValueError: If image is not a 3-channel RGB image or sigmas length != 3.
        """
        self._check_image_loaded()
        assert self.img is not None

        if self.img.ndim != 3 or self.img.shape[2] != 3:
            raise ValueError("RGB image required for chromatic Gaussian blur.")

        if len(sigmas) != 3:
            raise ValueError(
                f"Expected 3 sigma values for (R, G, B) channels, got {len(sigmas)}"
            )

        blurred_channels = [
            cv2.GaussianBlur(
                self.img[:, :, c],
                (0, 0),
                sigmaX=max(1e-4, float(sigmas[c])),
                sigmaY=max(1e-4, float(sigmas[c])),
            )
            for c in range(3)
        ]

        result = np.stack(blurred_channels, axis=-1)
        return result.astype(np.float32)

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

