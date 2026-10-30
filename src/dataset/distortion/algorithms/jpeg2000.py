"""JPEG 2000 wavelet compression and quantization distortion algorithms."""

from __future__ import annotations

import io
from typing import Any

import cv2
import numpy as np
import pywt
from PIL import Image

from dataset.distortion.algorithms.base import BaseDistortion


class Jpeg2000Distortion(BaseDistortion):
    """Simulates realistic JPEG 2000 wavelet compression artifacts and DWT degradations.

    Supports multiple JPEG 2000 compression and wavelet degradation models:
    - In-memory OpenCV libjasper / OpenJPEG compression with variable ratio
    - Tiled JPEG 2000 compression inducing boundary tile edge artifacts
    - Multi-resolution decomposition rate-limited compression (Pillow / OpenJPEG)
    - Distortion-constrained (target PSNR in dB) JPEG 2000 encoding
    - Discrete Wavelet Transform (DWT) subband coefficient deadzone quantization

    Provides a sequential `apply` method that round-robins across all JPEG 2000
    algorithms across successive calls, using an adaptive argument resolution
    strategy to map generic parameters to method-specific arguments.
    """

    DISTORTION_METHODS: tuple[str, ...] = (
        "add_jpeg2000_opencv_memory",
        "add_jpeg2000_tiled",
        "add_jpeg2000_resolutions",
        "add_jpeg2000_target_psnr",
        "add_wavelet_quantization_distortion",
    )

    @staticmethod
    def resolve_jpeg2000_arguments(
        method_name: str,
        compression_rate: float = 20.0,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Resolves and adapts arguments for a target JPEG 2000 algorithm.

        Maps unified/generic parameters (such as `compression_rate`, `compression_ratio`, or `rate`)
        into algorithm-specific parameters, prioritizing explicitly provided keyword arguments.

        Args:
            method_name: Target JPEG 2000 method name.
            compression_rate: General compression factor / ratio (default: 20.0).
            **kwargs: Explicit keyword arguments provided by caller.

        Returns:
            dict[str, Any]: Resolved keyword arguments tailored for the method.
        """
        resolved: dict[str, Any] = dict(kwargs)
        base_rate = float(
            resolved.get(
                "compression_rate",
                resolved.get(
                    "compression_ratio",
                    resolved.get("rate", resolved.get("ratio", compression_rate)),
                ),
            )
        )

        if method_name == "add_jpeg2000_opencv_memory":
            if "compression_ratio" not in resolved:
                resolved["compression_ratio"] = base_rate

        elif method_name == "add_jpeg2000_tiled":
            if "compression_rate" not in resolved:
                resolved["compression_rate"] = base_rate
            if "tile_size" not in resolved:
                resolved["tile_size"] = (128, 128)

        elif method_name == "add_jpeg2000_resolutions":
            if "compression_rate" not in resolved:
                resolved["compression_rate"] = base_rate
            if "num_resolutions" not in resolved:
                resolved["num_resolutions"] = 3

        elif method_name == "add_jpeg2000_target_psnr":
            if "target_psnr_db" not in resolved:
                psnr = resolved.get("psnr", resolved.get("target_psnr"))
                if psnr is not None:
                    resolved["target_psnr_db"] = float(psnr)
                else:
                    # Inversely map compression rate to target PSNR in dB:
                    # rate=20 -> 30.0 dB, rate=10 -> 37.5 dB, rate=40 -> 15.0 dB
                    resolved["target_psnr_db"] = max(
                        15.0, min(50.0, 45.0 - 0.75 * base_rate)
                    )

        elif method_name == "add_wavelet_quantization_distortion":
            if "wavelet" not in resolved:
                resolved["wavelet"] = "bior4.4"
            if "level" not in resolved:
                resolved["level"] = 3
            if "quant_step" not in resolved:
                # Map compression rate: rate=20 -> quant_step=0.15
                resolved["quant_step"] = max(0.01, min(1.0, base_rate * 0.0075))

        return resolved

    def apply(
        self,
        compression_rate: float = 20.0,
        image: np.ndarray | None = None,
        method: str | None = None,
        step: int | None = None,
        **kwargs: Any,
    ) -> np.ndarray:
        """Sequentially applies the next JPEG 2000 distortion in the sequence.

        Cycles round-robin across all 5 JPEG 2000 distortion methods on successive invocations.
        Uses an adaptive argument resolution strategy to route and scale compression
        parameters according to the specific requirements of the selected method.

        Args:
            compression_rate: General compression factor / ratio used across methods.
            image: Optional image array to set before applying distortion.
            method: Explicit method name to run instead of the sequential pointer.
            step: Explicit step index (0-4) to run instead of the internal counter.
            **kwargs: Extra parameters passed to the selected JPEG 2000 method
                     (e.g., compression_ratio, tile_size, num_resolutions, target_psnr_db, wavelet, quant_step).

        Returns:
            np.ndarray: Distorted image array as float32 in [0.0, 1.0].

        Raises:
            ValueError: If no image is loaded or if specified method is invalid.
        """
        if image is not None:
            self.set_image(image)

        self._check_image_loaded()
        selected_method, method_fn = self._select_method(method=method, step=step)

        resolved_kwargs = self.resolve_jpeg2000_arguments(
            selected_method,
            compression_rate=compression_rate,
            **kwargs,
        )

        return method_fn(**resolved_kwargs)

    def add_jpeg2000_opencv_memory(
        self,
        compression_ratio: float = 20.0,
        **kwargs: Any,
    ) -> np.ndarray:
        """Compresses image using OpenCV's in-memory JPEG 2000 encoder.

        Args:
            compression_ratio: Target compression ratio factor (default: 20.0).
            **kwargs: Extra ignored keyword arguments.

        Returns:
            np.ndarray: Distorted image array as float32 in [0.0, 1.0].
        """
        self._check_image_loaded()
        assert self.img is not None

        img_u8 = np.clip(self.img * 255.0, 0, 255).astype(np.uint8)

        if img_u8.ndim == 3 and img_u8.shape[2] == 3:
            img_u8 = cv2.cvtColor(img_u8, cv2.COLOR_RGB2BGR)

        # In OpenCV, IMWRITE_JPEG2000_COMPRESSION_X1000 represents (target_size / original_size) * 1000.
        # A compression ratio of R (e.g. 20:1) corresponds to target size 1/R (e.g. 5% -> 50 / 1000).
        ratio = max(1.0, float(compression_ratio))
        target_size_x1000 = int(np.clip(round(1000.0 / ratio), 1, 1000))
        encode_params = [
            int(cv2.IMWRITE_JPEG2000_COMPRESSION_X1000),
            target_size_x1000,
        ]
        success, encoded_buffer = cv2.imencode(".jp2", img_u8, encode_params)
        if not success:
            return self.img.copy()

        decoded_img = cv2.imdecode(encoded_buffer, cv2.IMREAD_UNCHANGED)
        if decoded_img is None:
            return self.img.copy()

        if decoded_img.ndim == 3 and decoded_img.shape[2] == 3:
            decoded_img = cv2.cvtColor(decoded_img, cv2.COLOR_BGR2RGB)

        return decoded_img.astype(np.float32) / 255.0

    def add_jpeg2000_tiled(
        self,
        compression_rate: float = 20.0,
        tile_size: tuple[int, int] = (128, 128),
        **kwargs: Any,
    ) -> np.ndarray:
        """Compresses image using tiled JPEG 2000 encoding via Pillow.

        Args:
            compression_rate: Target compression factor (default: 20.0).
            tile_size: Dimensions (width, height) of compression tiles.
            **kwargs: Extra ignored keyword arguments.

        Returns:
            np.ndarray: Distorted image array as float32 in [0.0, 1.0].
        """
        self._check_image_loaded()
        assert self.img is not None

        img_u8 = np.clip(self.img * 255.0, 0, 255).astype(np.uint8)
        pil_img = Image.fromarray(img_u8)

        buffer = io.BytesIO()
        # tile_size defines (tile_width, tile_height)
        # quality_mode='rates': compression_rate corresponds to compression factor
        pil_img.save(
            buffer,
            format="JPEG2000",
            tile_size=tile_size,
            quality_mode="rates",
            quality_layers=[compression_rate],
            irreversible=True,  # CDF 9/7 floating-point wavelet (lossy mode)
        )
        buffer.seek(0)

        jp2_img = Image.open(buffer)
        return np.array(jp2_img, dtype=np.float32) / 255.0

    def add_jpeg2000_resolutions(
        self,
        compression_rate: float = 20.0,
        num_resolutions: int = 3,
        **kwargs: Any,
    ) -> np.ndarray:
        """Compresses image with multi-resolution wavelet decomposition layers.

        Args:
            compression_rate: Target compression factor (default: 20.0).
            num_resolutions: Number of decomposition resolution levels (default: 3).
            **kwargs: Extra ignored keyword arguments.

        Returns:
            np.ndarray: Distorted image array as float32 in [0.0, 1.0].
        """
        self._check_image_loaded()
        assert self.img is not None

        img_u8 = np.clip(self.img * 255.0, 0, 255).astype(np.uint8)
        pil_img = Image.fromarray(img_u8)

        buffer = io.BytesIO()
        pil_img.save(
            buffer,
            format="JPEG2000",
            num_resolutions=int(num_resolutions),
            quality_mode="rates",
            quality_layers=[compression_rate],
            irreversible=True,
        )
        buffer.seek(0)

        jp2_img = Image.open(buffer)
        return np.array(jp2_img, dtype=np.float32) / 255.0

    def add_jpeg2000_target_psnr(
        self,
        target_psnr_db: float = 30.0,
        **kwargs: Any,
    ) -> np.ndarray:
        """Compresses image targeting a specific distortion PSNR bound in decibels.

        Args:
            target_psnr_db: Target peak signal-to-noise ratio in dB (default: 30.0).
            **kwargs: Extra ignored keyword arguments.

        Returns:
            np.ndarray: Distorted image array as float32 in [0.0, 1.0].
        """
        self._check_image_loaded()
        assert self.img is not None

        img_u8 = np.clip(self.img * 255.0, 0, 255).astype(np.uint8)
        pil_img = Image.fromarray(img_u8)

        buffer = io.BytesIO()
        # quality_mode='dB' sets target distortion levels directly
        pil_img.save(
            buffer,
            format="JPEG2000",
            quality_mode="dB",
            quality_layers=[float(target_psnr_db)],
            irreversible=True,
        )
        buffer.seek(0)

        jp2_img = Image.open(buffer)
        return np.array(jp2_img, dtype=np.float32) / 255.0

    def add_wavelet_quantization_distortion(
        self,
        wavelet: str = "bior4.4",
        level: int = 3,
        quant_step: float = 0.15,
        **kwargs: Any,
    ) -> np.ndarray:
        """Applies 2D Discrete Wavelet Transform and deadzone quantization to high-frequency subbands.

        Args:
            wavelet: Wavelet family name (default: "bior4.4").
            level: Number of decomposition levels (clamped to max supported).
            quant_step: Quantization step size applied to detail subbands (LH, HL, HH).
            **kwargs: Extra ignored keyword arguments.

        Returns:
            np.ndarray: Distorted image array as float32 in [0.0, 1.0].
        """
        self._check_image_loaded()
        assert self.img is not None

        w_obj = pywt.Wavelet(wavelet)
        min_dim = min(self.height or 0, self.width or 0)
        max_level = pywt.dwt_max_level(min_dim, w_obj.dec_len)
        effective_level = max(1, min(int(level), max(1, max_level)))

        def quantize_subbands(channel: np.ndarray) -> np.ndarray:
            coeffs = pywt.wavedec2(channel, wavelet=wavelet, level=effective_level)
            quantized_coeffs = [coeffs[0]]  # Keep approximation subband LL unchanged

            # Quantize detail subbands: (LH, HL, HH)
            for detail_level in coeffs[1:]:
                quantized_level = tuple(
                    np.round(subband / float(quant_step)) * float(quant_step)
                    for subband in detail_level
                )
                quantized_coeffs.append(quantized_level)

            reconstructed = pywt.waverec2(quantized_coeffs, wavelet=wavelet)
            return reconstructed[: channel.shape[0], : channel.shape[1]]

        if self.img.ndim == 3:
            channels = [
                quantize_subbands(self.img[:, :, c]) for c in range(self.img.shape[2])
            ]
            distorted = np.stack(channels, axis=-1)
        else:
            distorted = quantize_subbands(self.img)

        return np.clip(distorted, 0.0, 1.0).astype(np.float32)
