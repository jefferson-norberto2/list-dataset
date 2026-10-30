"""Multi-level synthetic image distortion dataset generator service."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from tqdm import tqdm

from dataset.distortion.algorithms.blur import BlurDistortion
from dataset.distortion.algorithms.contrast import ContrastDistortion
from dataset.distortion.algorithms.fnoise import FnoiseDistortion
from dataset.distortion.algorithms.jpeg import JpegDistortion
from dataset.distortion.algorithms.jpeg2000 import Jpeg2000Distortion
from dataset.distortion.algorithms.poisson import PoissonDistortion

# Valid image file extensions
SUPPORTED_EXTENSIONS: frozenset[str] = frozenset(
    {".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff"}
)

# Standard progression levels across 5 severity stages
DEFAULT_LEVELS: dict[str, list[Any]] = {
    "blur_sigma": [1.0, 1.5, 2.0, 3.0, 4.0],
    "pgn_shot": [0.005, 0.015, 0.03, 0.05, 0.08],
    "pgn_read": [0.0005, 0.001, 0.002, 0.003, 0.005],
    "jpeg_quality": [20, 15, 10, 8, 3],  # Quality decreases with level
    "jp2k_ratio": [
        10,
        25,
        50,
        100,
        200,
    ],  # Compression ratio increases with level (10x to 200x)
    "contrast_alpha": [0.6, 0.5, 0.4, 0.3, 0.2],  # Contrast decreases with level
    "pink_intensity": [0.03, 0.07, 0.12, 0.18, 0.25],  # Noise increases with level
}


class DistortionGenerator:
    """Orchestrates multi-level image distortion generation across dataset directories.

    Delegates distortion operations to modular algorithm engines:
    - `BlurDistortion`: Optical and synthetic blur degradation
    - `PoissonDistortion`: Sensor photon-shot and electronic read noise
    - `JpegDistortion`: DCT quantization and JPEG compression artifacts
    - `Jpeg2000Distortion`: Wavelet subband quantization and JPEG 2000 compression
    - `ContrastDistortion`: Linear and non-linear dynamic range scaling
    - `FnoiseDistortion`: 1/f spatial pink noise degradations
    """

    def __init__(
        self,
        image: np.ndarray | str | Path | None = None,
        filename_format: str = "dist_{level}_{img_name}.png",
    ) -> None:
        """Initializes the distortion generator and its internal algorithm engines.

        Args:
            image: Optional input image as array or file path.
            filename_format: Default output filename template.
        """
        self.img_uint8: np.ndarray | None = None
        self.img: np.ndarray | None = None

        self.height: int | None = None
        self.width: int | None = None
        self.channels: int | None = None

        self.img_name: str = "image"
        self.folder_path: str | None = None
        self.root_path: str | None = None
        self.filename_format = filename_format

        # Algorithm engines
        self.blur_distortion = BlurDistortion()
        self.poisson_distortion = PoissonDistortion()
        self.jpeg_distortion = JpegDistortion()
        self.jpeg2000_distortion = Jpeg2000Distortion()
        self.contrast_distortion = ContrastDistortion()
        self.fnoise_distortion = FnoiseDistortion()

        if image is not None:
            if isinstance(image, (str, Path)):
                self.change_image(str(image))
            elif isinstance(image, np.ndarray):
                self.set_image(image)
            else:
                raise TypeError(f"Unsupported image type: {type(image).__name__}")

    def _sync_algorithms(self) -> None:
        """Synchronizes internal image and metadata across all algorithm engines."""
        if self.img is not None:
            self.blur_distortion.set_image(self.img)
            self.poisson_distortion.set_image(self.img)
            self.jpeg_distortion.set_image(self.img)
            self.jpeg2000_distortion.set_image(self.img)
            self.contrast_distortion.set_image(self.img)
            self.fnoise_distortion.set_image(self.img)

            self.blur_distortion.img_name = self.img_name
            self.poisson_distortion.img_name = self.img_name
            self.jpeg_distortion.img_name = self.img_name
            self.jpeg2000_distortion.img_name = self.img_name
            self.contrast_distortion.img_name = self.img_name
            self.fnoise_distortion.img_name = self.img_name

    def _check_image_loaded(self) -> None:
        """Ensures an image is loaded before performing operations.

        Raises:
            ValueError: If no image has been loaded.
        """
        if self.img is None or self.img_uint8 is None:
            raise ValueError("No image loaded. Use change_image() to load an image.")

    def _get_image_name(self, image_path: str | Path) -> str:
        """Extracts the base stem name of an image path."""
        return Path(image_path).stem

    def change_image(self, new_image_path: str | Path) -> None:
        """Loads and normalizes a new image from disk, syncing all algorithm engines.

        Args:
            new_image_path: Path to the image file to load.

        Raises:
            ValueError: If the file cannot be found or decoded.
        """
        path_obj = Path(new_image_path)
        img_bgr = cv2.imread(str(path_obj))

        if img_bgr is None:
            raise ValueError(f"Image not found at path: {new_image_path}.")

        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        self.set_image(img_rgb, img_name=path_obj.stem)

        self.folder_path = str(path_obj.parent)
        self.root_path = str(path_obj.parent.parent)

    def set_image(self, image: np.ndarray, img_name: str = "image") -> None:
        """Sets the working image from an in-memory NumPy array.

        Converts input to float32 normalized in [0.0, 1.0].

        Args:
            image: Grayscale or RGB image array (uint8 or float).
            img_name: Base identifier name for output naming.

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
        self.img_name = img_name

        self._sync_algorithms()

    def save_output(
        self,
        distorted_img: np.ndarray,
        distortion_name: str,
        level: int | str,
        output_dir: str | Path | None = None,
        filename_format: str | None = None,
    ) -> Path:
        """Saves the resulting image converting back to uint8 using lossless PNG encoding.

        Args:
            distorted_img: Distorted image array in float32 [0.0, 1.0] or uint8.
            distortion_name: Distortion category folder name (e.g. 'blur', 'jpeg').
            level: Distortion severity level (e.g. 1, 2, 3, 4, 5).
            output_dir: Optional explicit directory. If None, saves in '{self.root_path}/{distortion_name}'.
            filename_format: Optional filename format string. Defaults to self.filename_format.

        Returns:
            Path: Resolved path to the saved PNG image.
        """
        self._check_image_loaded()

        img_clipped = np.clip(distorted_img, 0.0, 1.0)
        img_out = (img_clipped * 255.0).astype(np.uint8)

        if output_dir is not None:
            save_path = Path(output_dir)
        elif self.root_path is not None:
            save_path = Path(self.root_path) / distortion_name
        else:
            save_path = Path(distortion_name)

        save_path.mkdir(parents=True, exist_ok=True)

        fmt = filename_format or self.filename_format
        filename = fmt.format(
            img_name=self.img_name,
            distortion=distortion_name,
            level=level,
        )
        file_path = save_path / filename

        # Convert RGB to BGR for OpenCV export
        if img_out.ndim == 3 and img_out.shape[2] == 3:
            img_to_write = cv2.cvtColor(img_out, cv2.COLOR_RGB2BGR)
        else:
            img_to_write = img_out

        cv2.imwrite(str(file_path), img_to_write, [cv2.IMWRITE_PNG_COMPRESSION, 0])
        return file_path.resolve()

    # --- INDIVIDUAL DELEGATION METHODS ---

    def add_gaussian_blur(
        self,
        sigma: float = 2.0,
        force_kernel: int | tuple[int, int] | list[int] | None = None,
        **kwargs: Any,
    ) -> np.ndarray:
        """Applies Gaussian blur using BlurDistortion."""
        self._check_image_loaded()
        return self.blur_distortion.add_gaussian_blur(
            sigma_x=sigma,
            sigma_y=sigma,
            force_kernel=force_kernel,
            **kwargs,
        )

    def add_poisson_gaussian_noise(
        self,
        shot_noise: float = 0.01,
        read_noise: float = 0.001,
        **kwargs: Any,
    ) -> np.ndarray:
        """Applies Poisson-Gaussian noise using PoissonDistortion."""
        self._check_image_loaded()
        return self.poisson_distortion.add_poisson_gaussian_noise(
            shot_noise=shot_noise,
            read_noise=read_noise,
            **kwargs,
        )

    def add_jpeg_compression(
        self,
        quality: int = 10,
        **kwargs: Any,
    ) -> np.ndarray:
        """Applies JPEG compression using JpegDistortion."""
        self._check_image_loaded()
        return self.jpeg_distortion.add_jpeg_opencv(quality=quality, **kwargs)

    def add_jpeg2000_compression(
        self,
        compression_ratio: float = 20.0,
        **kwargs: Any,
    ) -> np.ndarray:
        """Applies JPEG 2000 compression using Jpeg2000Distortion."""
        self._check_image_loaded()
        return self.jpeg2000_distortion.add_jpeg2000_opencv_memory(
            compression_ratio=compression_ratio, **kwargs
        )

    def change_contrast(
        self,
        alpha: float = 0.5,
        **kwargs: Any,
    ) -> np.ndarray:
        """Applies contrast scaling using ContrastDistortion."""
        self._check_image_loaded()
        return self.contrast_distortion.change_contrast_mean_adaptive(
            alpha=alpha, **kwargs
        )

    def add_pink_noise(
        self,
        intensity: float = 0.05,
        spatial_scale: float = 1.0,
        **kwargs: Any,
    ) -> np.ndarray:
        """Applies 1/f spatial pink noise using FnoiseDistortion."""
        self._check_image_loaded()
        return self.fnoise_distortion.add_correlated_pink_noise(
            intensity=intensity,
            spatial_scale=spatial_scale,
            **kwargs,
        )

    # --- GENERAL APPLY PIPELINE METHODS ---

    def apply_blur(
        self,
        sigma: float = 2.0,
        method: str | None = None,
        step: int | None = None,
        **kwargs: Any,
    ) -> np.ndarray:
        """Applies blur distortion via BlurDistortion.apply."""
        self._check_image_loaded()
        return self.blur_distortion.apply(
            sigma=sigma, method=method, step=step, **kwargs
        )

    def apply_poisson(
        self,
        shot_noise: float = 0.01,
        read_noise: float = 0.005,
        method: str | None = None,
        step: int | None = None,
        **kwargs: Any,
    ) -> np.ndarray:
        """Applies noise distortion via PoissonDistortion.apply."""
        self._check_image_loaded()
        return self.poisson_distortion.apply(
            shot_noise=shot_noise,
            read_noise=read_noise,
            method=method,
            step=step,
            **kwargs,
        )

    def apply_jpeg(
        self,
        quality: int = 10,
        method: str | None = None,
        step: int | None = None,
        **kwargs: Any,
    ) -> np.ndarray:
        """Applies JPEG compression via JpegDistortion.apply."""
        self._check_image_loaded()
        return self.jpeg_distortion.apply(
            quality=quality, method=method, step=step, **kwargs
        )

    def apply_jpeg2000(
        self,
        compression_rate: float = 20.0,
        method: str | None = None,
        step: int | None = None,
        **kwargs: Any,
    ) -> np.ndarray:
        """Applies JPEG 2000 compression via Jpeg2000Distortion.apply."""
        self._check_image_loaded()
        return self.jpeg2000_distortion.apply(
            compression_rate=compression_rate, method=method, step=step, **kwargs
        )

    def apply_contrast(
        self,
        alpha: float = 0.5,
        method: str | None = None,
        step: int | None = None,
        **kwargs: Any,
    ) -> np.ndarray:
        """Applies contrast adjustment via ContrastDistortion.apply."""
        self._check_image_loaded()
        return self.contrast_distortion.apply(
            alpha=alpha, method=method, step=step, **kwargs
        )

    def apply_fnoise(
        self,
        intensity: float = 0.05,
        method: str | None = None,
        step: int | None = None,
        **kwargs: Any,
    ) -> np.ndarray:
        """Applies pink noise distortion via FnoiseDistortion.apply."""
        self._check_image_loaded()
        return self.fnoise_distortion.apply(
            intensity=intensity, method=method, step=step, **kwargs
        )

    # --- BATCH DATASET GENERATION ---

    def generate_all_distortions_for_image(
        self,
        image_path: str | Path,
        output_root: str | Path | None = None,
        levels: dict[str, Any] | None = None,
        distortions: Sequence[str] | None = None,
        filename_format: str = "{img_name}.{distortion}.{level}.png",
        use_apply: bool = False,
    ) -> dict[str, list[Path]]:
        """Generates all 5 degradation levels for each distortion category for a single image.

        Args:
            image_path: Path to the pristine reference image.
            output_root: Root destination directory where distortion subfolders are created.
                         If None, defaults to the parent directory of image's folder.
            levels: Dictionary of severity parameters per level (defaults to DEFAULT_LEVELS).
            distortions: Selected distortions to generate (e.g. ['blur', 'jpeg']). Defaults to all 6.
            filename_format: Formatting template for filenames (default: '{img_name}.{distortion}.{level}.png').
            use_apply: If True, uses the round-robin apply() pipeline of each engine.

        Returns:
            dict[str, list[Path]]: Map of distortion category to generated file paths.
        """
        self.change_image(str(image_path))

        target_levels = levels or DEFAULT_LEVELS
        selected_distortions = set(
            distortions or ["blur", "awgn", "jpeg", "jpeg2000", "contrast", "fnoise"]
        )

        dest_root = (
            Path(output_root)
            if output_root is not None
            else Path(self.root_path or ".")
        )

        results: dict[str, list[Path]] = {}

        for lvl in range(5):
            level_name = lvl + 1

            # 1. Blur
            if "blur" in selected_distortions:
                sigma_val = target_levels["blur_sigma"][lvl]
                if use_apply:
                    res_blur = self.apply_blur(sigma=sigma_val)
                else:
                    res_blur = self.add_gaussian_blur(sigma=sigma_val)

                p = self.save_output(
                    res_blur,
                    "blur",
                    level_name,
                    output_dir=dest_root / "blur",
                    filename_format=filename_format,
                )
                results.setdefault("blur", []).append(p)

            # 2. AWGN / Poisson-Gaussian Noise
            if "awgn" in selected_distortions or "poisson" in selected_distortions:
                dist_key = "awgn" if "awgn" in selected_distortions else "poisson"
                shot_val = target_levels["pgn_shot"][lvl]
                read_val = target_levels["pgn_read"][lvl]
                if use_apply:
                    res_noise = self.apply_poisson(
                        shot_noise=shot_val, read_noise=read_val
                    )
                else:
                    res_noise = self.add_poisson_gaussian_noise(
                        shot_noise=shot_val, read_noise=read_val
                    )

                p = self.save_output(
                    res_noise,
                    dist_key,
                    level_name,
                    output_dir=dest_root / dist_key,
                    filename_format=filename_format,
                )
                results.setdefault(dist_key, []).append(p)

            # 3. JPEG
            if "jpeg" in selected_distortions:
                q_val = target_levels["jpeg_quality"][lvl]
                if use_apply:
                    res_jpg = self.apply_jpeg(quality=q_val)
                else:
                    res_jpg = self.add_jpeg_compression(quality=q_val)

                p = self.save_output(
                    res_jpg,
                    "jpeg",
                    level_name,
                    output_dir=dest_root / "jpeg",
                    filename_format=filename_format,
                )
                results.setdefault("jpeg", []).append(p)

            # 4. JPEG 2000
            if "jpeg2000" in selected_distortions:
                ratio_val = target_levels["jp2k_ratio"][lvl]
                if use_apply:
                    res_jp2 = self.apply_jpeg2000(compression_rate=ratio_val)
                else:
                    res_jp2 = self.add_jpeg2000_compression(compression_ratio=ratio_val)

                p = self.save_output(
                    res_jp2,
                    "jpeg2000",
                    level_name,
                    output_dir=dest_root / "jpeg2000",
                    filename_format=filename_format,
                )
                results.setdefault("jpeg2000", []).append(p)

            # 5. Contrast Decrement
            if "contrast" in selected_distortions:
                alpha_val = target_levels["contrast_alpha"][lvl]
                if use_apply:
                    res_contrast = self.apply_contrast(alpha=alpha_val)
                else:
                    res_contrast = self.change_contrast(alpha=alpha_val)

                p = self.save_output(
                    res_contrast,
                    "contrast",
                    level_name,
                    output_dir=dest_root / "contrast",
                    filename_format=filename_format,
                )
                results.setdefault("contrast", []).append(p)

            # 6. Pink Noise
            if "fnoise" in selected_distortions:
                intensity_val = target_levels["pink_intensity"][lvl]
                if use_apply:
                    res_pink = self.apply_fnoise(intensity=intensity_val)
                else:
                    res_pink = self.add_pink_noise(
                        intensity=intensity_val, spatial_scale=1.0
                    )

                p = self.save_output(
                    res_pink,
                    "fnoise",
                    level_name,
                    output_dir=dest_root / "fnoise",
                    filename_format=filename_format,
                )
                results.setdefault("fnoise", []).append(p)

        return results

    def process_folder(
        self,
        folder_path: str | Path,
        output_root: str | Path | None = None,
        levels: dict[str, Any] | None = None,
        distortions: Sequence[str] | None = None,
        filename_format: str = "{img_name}.{distortion}.{level}.png",
        use_apply: bool = False,
        limit: int | None = None,
        show_progress: bool = True,
    ) -> dict[str, int]:
        """Processes all pristine images in a folder and generates distortion subfolders.

        Args:
            folder_path: Path to directory containing pristine source images.
            output_root: Root directory for distortion folders. Defaults to parent of folder_path.
            levels: Custom distortion severity parameter dictionary.
            distortions: Subset of distortions to apply.
            filename_format: Template string for filenames.
            use_apply: Whether to use round-robin apply() across methods.
            limit: Optional maximum number of source images to process.
            show_progress: If True, displays a tqdm progress bar.

        Returns:
            dict[str, int]: Number of generated artifacts per distortion category.
        """
        src_dir = Path(folder_path)
        if not src_dir.is_dir():
            raise FileNotFoundError(f"Source directory not found: {folder_path}")

        all_files = sorted(
            [
                f
                for f in src_dir.iterdir()
                if f.is_file() and f.suffix.lower() in SUPPORTED_EXTENSIONS
            ]
        )

        if limit is not None and limit > 0:
            all_files = all_files[:limit]

        if not all_files:
            return {}

        dest_root = Path(output_root) if output_root is not None else src_dir.parent

        counts: dict[str, int] = {}
        iterator = (
            tqdm(all_files, desc="Generating distortions")
            if show_progress
            else all_files
        )

        for file_path in iterator:
            try:
                gen_dict = self.generate_all_distortions_for_image(
                    image_path=file_path,
                    output_root=dest_root,
                    levels=levels,
                    distortions=distortions,
                    filename_format=filename_format,
                    use_apply=use_apply,
                )
                for dist_name, paths in gen_dict.items():
                    counts[dist_name] = counts.get(dist_name, 0) + len(paths)
            except Exception as e:  # noqa: BLE001
                print(f"Error processing {file_path.name}: {e}")

        return counts


def main(argv: list[str] | None = None) -> int:
    """CLI entry point for running multi-level synthetic distortion generation."""
    parser = argparse.ArgumentParser(
        description="Generates 5-level synthetic distortions organized in subfolders."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("Datasets/LISTV2/images/test/src")
        if Path("Datasets/LISTV2/images/test/src").is_dir()
        else Path("Datasets/LIVE_RGB/src"),
        help="Input folder containing pristine reference images.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output root folder where distortion subfolders are created (defaults to parent of input-dir).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional limit on number of source images to process.",
    )
    parser.add_argument(
        "--format",
        type=str,
        default="{img_name}.{distortion}.{level}.png",
        help="Filename formatting string (default: '{img_name}.{distortion}.{level}.png').",
    )
    parser.add_argument(
        "--use-apply",
        action="store_true",
        help="Rotate through algorithm variants using round-robin apply() pipeline.",
    )
    parser.add_argument(
        "--distortions",
        nargs="+",
        default=None,
        help="Subset of distortions to generate (e.g. blur awgn jpeg jpeg2000 contrast fnoise).",
    )

    args = parser.parse_args(argv)

    if not args.input_dir.is_dir():
        print(f"Error: input directory '{args.input_dir}' does not exist.")
        return 1

    generator = DistortionGenerator()
    print(f"Starting distortion generation from: {args.input_dir}")
    counts = generator.process_folder(
        folder_path=args.input_dir,
        output_root=args.output_dir,
        filename_format=args.format,
        use_apply=args.use_apply,
        limit=args.limit,
        distortions=args.distortions,
        show_progress=True,
    )

    print("\nGeneration summary:")
    for dist_name, count in sorted(counts.items()):
        print(f"  - {dist_name}: {count} images generated")

    return 0


if __name__ == "__main__":
    sys.exit(main())
