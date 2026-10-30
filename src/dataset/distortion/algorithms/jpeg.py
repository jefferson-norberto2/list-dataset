"""JPEG compression and quantization distortion algorithms."""

from __future__ import annotations

import io
from typing import Any

import cv2
import numpy as np
from PIL import Image

from dataset.distortion.algorithms.base import BaseDistortion


class JpegDistortion(BaseDistortion):
    """Simulates realistic JPEG compression artifacts and DCT quantization distortions.

    Supports multiple JPEG compression and degradation models:
    - JPEG with explicit chroma subsampling (Pillow)
    - Standard OpenCV libjpeg compression
    - Grid-shifted JPEG compression to induce block boundary misalignments
    - Double JPEG compression with differing quantization qualities
    - Custom DCT luminance quantization table scaling (high-frequency amplification)

    Provides a sequential `apply` method that round-robins across all JPEG
    algorithms across successive calls, using an adaptive argument resolution
    strategy to map generic parameters to method-specific arguments.
    """

    DISTORTION_METHODS: tuple[str, ...] = (
        "add_jpeg_with_subsampling",
        "add_jpeg_opencv",
        "add_jpeg_grid_shift",
        "add_double_jpeg_compression",
        "add_custom_quantized_jpeg",
    )

    @staticmethod
    def resolve_jpeg_arguments(
        method_name: str,
        quality: int = 10,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Resolves and adapts arguments for a target JPEG compression algorithm.

        Maps unified/generic parameters (such as `quality`, `q`, or `factor`) into
        algorithm-specific parameters, while prioritizing explicitly provided keyword arguments.

        Args:
            method_name: Target JPEG method name.
            quality: General JPEG compression quality factor (1-95, default: 10).
            **kwargs: Explicit keyword arguments provided by caller.

        Returns:
            dict[str, Any]: Resolved keyword arguments tailored for the method.
        """
        resolved: dict[str, Any] = dict(kwargs)
        base_quality = int(
            resolved.get(
                "quality",
                resolved.get("q", quality),
            )
        )

        if method_name == "add_jpeg_with_subsampling":
            if "quality" not in resolved:
                resolved["quality"] = base_quality
            if "subsampling" not in resolved:
                resolved["subsampling"] = 2

        elif method_name == "add_jpeg_opencv":
            if "quality" not in resolved:
                resolved["quality"] = base_quality

        elif method_name == "add_jpeg_grid_shift":
            if "quality" not in resolved:
                resolved["quality"] = base_quality
            if "shift_x" not in resolved:
                resolved["shift_x"] = 4
            if "shift_y" not in resolved:
                resolved["shift_y"] = 4

        elif method_name == "add_double_jpeg_compression":
            if "quality_2" not in resolved:
                resolved["quality_2"] = max(1, min(95, base_quality))
            if "quality_1" not in resolved:
                resolved["quality_1"] = max(5, min(95, int(base_quality * 2)))

        elif method_name == "add_custom_quantized_jpeg":
            if "high_freq_quant_factor" not in resolved:
                factor = resolved.get("quant_factor", resolved.get("factor"))
                if factor is not None:
                    resolved["high_freq_quant_factor"] = float(factor)
                else:
                    # Inversely map quality (1-95) to high-frequency quantization factor:
                    # quality=10 -> 8.0, quality=20 -> 4.0, quality=80 -> 1.0
                    resolved["high_freq_quant_factor"] = max(
                        1.0, 80.0 / max(1.0, float(base_quality))
                    )

        return resolved

    def apply(
        self,
        quality: int = 10,
        image: np.ndarray | None = None,
        method: str | None = None,
        step: int | None = None,
        **kwargs: Any,
    ) -> np.ndarray:
        """Sequentially applies the next JPEG compression distortion in the sequence.

        Cycles round-robin across all 5 JPEG distortion methods on successive invocations.
        Uses an adaptive argument resolution strategy to route and scale compression
        parameters according to the specific requirements of the selected method.

        Args:
            quality: General JPEG quality parameter (1-95) used across methods.
            image: Optional image array to set before applying distortion.
            method: Explicit method name to run instead of the sequential pointer.
            step: Explicit step index (0-4) to run instead of the internal counter.
            **kwargs: Extra parameters passed to the selected JPEG method
                     (e.g., subsampling, shift_x, shift_y, quality_1, quality_2, high_freq_quant_factor).

        Returns:
            np.ndarray: Distorted image array as float32 in [0.0, 1.0].

        Raises:
            ValueError: If no image is loaded or if specified method is invalid.
        """
        if image is not None:
            self.set_image(image)

        self._check_image_loaded()
        selected_method, method_fn = self._select_method(method=method, step=step)

        resolved_kwargs = self.resolve_jpeg_arguments(
            selected_method,
            quality=quality,
            **kwargs,
        )

        return method_fn(**resolved_kwargs)

    def add_jpeg_with_subsampling(
        self,
        quality: int = 10,
        subsampling: int = 2,
        **kwargs: Any,
    ) -> np.ndarray:
        """Compresses image using Pillow with explicit chroma subsampling.

        Args:
            quality: JPEG compression quality factor (1-95, default: 10).
            subsampling: Chroma subsampling setting: 0 (4:4:4), 1 (4:2:2), 2 (4:2:0).
            **kwargs: Extra ignored keyword arguments.

        Returns:
            np.ndarray: Distorted image array as float32 in [0.0, 1.0].
        """
        self._check_image_loaded()
        assert self.img is not None

        uint8_img = np.clip(self.img * 255.0, 0, 255).astype(np.uint8)
        pil_img = Image.fromarray(uint8_img)
        buffer = io.BytesIO()

        # subsampling: 0 (4:4:4), 1 (4:2:2), 2 (4:2:0)
        save_kwargs: dict[str, Any] = {"quality": int(quality)}
        if pil_img.mode in ("RGB", "YCbCr"):
            save_kwargs["subsampling"] = int(subsampling)

        pil_img.save(buffer, "JPEG", **save_kwargs)
        buffer.seek(0)

        jpeg_img = Image.open(buffer)
        return np.array(jpeg_img, dtype=np.float32) / 255.0

    def add_jpeg_opencv(
        self,
        quality: int = 10,
        **kwargs: Any,
    ) -> np.ndarray:
        """Compresses image using OpenCV's libjpeg implementation.

        Args:
            quality: JPEG compression quality factor (1-100, default: 10).
            **kwargs: Extra ignored keyword arguments.

        Returns:
            np.ndarray: Distorted image array as float32 in [0.0, 1.0].

        Raises:
            RuntimeError: If encoding or decoding fails.
        """
        self._check_image_loaded()
        assert self.img is not None

        # Scale to 8-bit uint8
        uint8_img = np.clip(self.img * 255.0, 0, 255).astype(np.uint8)

        # OpenCV uses BGR ordering
        if uint8_img.ndim == 3 and uint8_img.shape[2] == 3:
            uint8_img = cv2.cvtColor(uint8_img, cv2.COLOR_RGB2BGR)

        encode_params = [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)]
        success, encoded_buffer = cv2.imencode(".jpg", uint8_img, encode_params)

        if not success:
            raise RuntimeError("JPEG encoding failed.")

        decoded_img = cv2.imdecode(encoded_buffer, cv2.IMREAD_UNCHANGED)
        if decoded_img is None:
            raise RuntimeError("JPEG decoding failed.")

        if decoded_img.ndim == 3 and decoded_img.shape[2] == 3:
            decoded_img = cv2.cvtColor(decoded_img, cv2.COLOR_RGB2BGR)

        return decoded_img.astype(np.float32) / 255.0

    def add_jpeg_grid_shift(
        self,
        quality: int = 10,
        shift_x: int = 4,
        shift_y: int = 4,
        **kwargs: Any,
    ) -> np.ndarray:
        """Compresses image after padding to offset 8x8 DCT grid boundaries.

        Args:
            quality: JPEG compression quality factor (1-95, default: 10).
            shift_x: Horizontal grid shift in pixels (modulo 8).
            shift_y: Vertical grid shift in pixels (modulo 8).
            **kwargs: Extra ignored keyword arguments.

        Returns:
            np.ndarray: Distorted image array as float32 in [0.0, 1.0].
        """
        self._check_image_loaded()
        assert self.img is not None

        shift_x = int(shift_x) % 8
        shift_y = int(shift_y) % 8

        h, w = self.img.shape[:2]
        uint8_img = np.clip(self.img * 255.0, 0, 255).astype(np.uint8)

        # Pad borders to avoid boundary wrapping
        padded_img = np.pad(
            uint8_img,
            (
                ((shift_y, 8 - shift_y), (shift_x, 8 - shift_x), (0, 0))
                if uint8_img.ndim == 3
                else ((shift_y, 8 - shift_y), (shift_x, 8 - shift_x))
            ),
            mode="edge",
        )

        # Compress padded image
        pil_img = Image.fromarray(padded_img)
        buffer = io.BytesIO()
        pil_img.save(buffer, "JPEG", quality=int(quality))
        buffer.seek(0)
        compressed_padded = np.array(Image.open(buffer), dtype=np.float32) / 255.0

        # Crop back to original dimensions
        cropped_img = compressed_padded[shift_y : shift_y + h, shift_x : shift_x + w]
        return cropped_img

    def add_double_jpeg_compression(
        self,
        quality_1: int = 30,
        quality_2: int = 15,
        **kwargs: Any,
    ) -> np.ndarray:
        """Applies two successive rounds of JPEG compression.

        Args:
            quality_1: Quality factor for first compression pass (default: 30).
            quality_2: Quality factor for second compression pass (default: 15).
            **kwargs: Extra ignored keyword arguments.

        Returns:
            np.ndarray: Distorted image array as float32 in [0.0, 1.0].
        """
        self._check_image_loaded()
        assert self.img is not None

        uint8_img = np.clip(self.img * 255.0, 0, 255).astype(np.uint8)

        # First pass
        buffer1 = io.BytesIO()
        Image.fromarray(uint8_img).save(buffer1, "JPEG", quality=int(quality_1))
        buffer1.seek(0)
        intermediate_img = Image.open(buffer1)

        # Second pass
        buffer2 = io.BytesIO()
        intermediate_img.save(buffer2, "JPEG", quality=int(quality_2))
        buffer2.seek(0)
        final_img = Image.open(buffer2)

        return np.array(final_img, dtype=np.float32) / 255.0

    def add_custom_quantized_jpeg(
        self,
        high_freq_quant_factor: float = 8.0,
        **kwargs: Any,
    ) -> np.ndarray:
        """Compresses image using a custom luminance quantization table.

        Amplifies high-frequency quantization steps towards the bottom-right of
        the 8x8 DCT matrix to induce pronounced ringing and edge blurring.

        Args:
            high_freq_quant_factor: Factor scaling high-frequency quantization steps.
            **kwargs: Extra ignored keyword arguments.

        Returns:
            np.ndarray: Distorted image array as float32 in [0.0, 1.0].
        """
        self._check_image_loaded()
        assert self.img is not None

        # Standard base luminance quantization table (ITU-R BT.601)
        base_luma_table = [
            16,
            11,
            10,
            16,
            24,
            40,
            51,
            61,
            12,
            12,
            14,
            19,
            26,
            58,
            60,
            55,
            14,
            13,
            16,
            24,
            40,
            57,
            69,
            56,
            14,
            17,
            22,
            29,
            51,
            87,
            80,
            62,
            18,
            22,
            37,
            56,
            68,
            109,
            103,
            77,
            24,
            35,
            55,
            64,
            81,
            104,
            113,
            92,
            49,
            64,
            78,
            87,
            103,
            121,
            120,
            101,
            72,
            92,
            95,
            98,
            112,
            100,
            103,
            99,
        ]

        # Apply custom amplification to higher frequencies (bottom-right of 8x8 block)
        modified_table = []
        for idx, val in enumerate(base_luma_table):
            row, col = idx // 8, idx % 8
            weight = 1.0 + (row + col) / 14.0 * (float(high_freq_quant_factor) - 1.0)
            quant_val = int(np.clip(round(val * weight), 1, 255))
            modified_table.append(quant_val)

        pil_img = Image.fromarray(np.clip(self.img * 255.0, 0, 255).astype(np.uint8))
        buffer = io.BytesIO()

        # qtables accepts a list of 64-element sequences (up to 4 tables)
        pil_img.save(buffer, "JPEG", qtables=[modified_table])
        buffer.seek(0)

        jpeg_img = Image.open(buffer)
        return np.array(jpeg_img, dtype=np.float32) / 255.0
