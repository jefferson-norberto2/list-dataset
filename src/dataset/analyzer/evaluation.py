"""Dataset-wide image similarity and distortion level evaluation module."""

from __future__ import annotations

import csv
import logging
import math
import re
from collections import defaultdict
from collections.abc import Sequence
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from dataset.analyzer.similarity import compare_images_ssim

logger = logging.getLogger(__name__)

# Valid image file extensions
SUPPORTED_IMAGE_EXTENSIONS: frozenset[str] = frozenset(
    {".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff"}
)

# Regex matching <ref_stem>.<distortion_name>.<level>.<ext>
# Example: 1600.JPEG.1.png -> ref='1600', distortion='JPEG', level='1'
DISTORTED_NAME_REGEX = re.compile(
    r"^(?P<ref>.+)\.(?P<distortion>[^.]+)\.(?P<level>[^.]+)$"
)


@dataclass(frozen=True)
class DistortionPair:
    """Pair of reference image and its distorted candidate image.

    Attributes:
        ref_path: Path to reference image on disk.
        dist_path: Path to distorted image on disk.
        ref_stem: Stem name of reference image.
        distortion: Name or category of distortion (e.g. 'JPEG', 'AWGN').
        level: Severity level (e.g. 1, 2, 3, 4, 5).
    """

    ref_path: Path
    dist_path: Path
    ref_stem: str
    distortion: str
    level: int | str


@dataclass(frozen=True)
class PairSimilarityResult:
    """SSIM evaluation result for a single reference/distorted pair.

    Attributes:
        ref_stem: Stem name of reference image.
        ref_path: Path to reference image.
        dist_path: Path to distorted image.
        distortion: Distortion name.
        level: Severity level.
        ssim: Computed SSIM score in [-1.0, 1.0] or NaN.
        difference: Difference from reference (1.0 - SSIM) or NaN.
    """

    ref_stem: str
    ref_path: Path
    dist_path: Path
    distortion: str
    level: int | str
    ssim: float
    difference: float


@dataclass(frozen=True)
class LevelStatistics:
    """Aggregated statistics for a specific distortion level or (distortion, level) group.

    Attributes:
        level: Severity level.
        count: Number of evaluated image pairs.
        mean_ssim: Mean SSIM score across pairs.
        median_ssim: Median SSIM score across pairs.
        std_ssim: Standard deviation of SSIM scores.
        min_ssim: Minimum SSIM score.
        max_ssim: Maximum SSIM score.
        mean_difference: Mean difference (1.0 - SSIM).
        median_difference: Median difference.
    """

    level: int | str
    count: int
    mean_ssim: float
    median_ssim: float
    std_ssim: float
    min_ssim: float
    max_ssim: float
    mean_difference: float
    median_difference: float


@dataclass
class DatasetSimilarityReport:
    """Comprehensive similarity assessment report for a dataset across distortion levels.

    Attributes:
        results: List of all pairwise evaluation results.
        level_stats: Aggregate statistics mapped by level.
        distortion_level_stats: Aggregate statistics mapped by (distortion, level).
        unmatched_files: Distorted image files that could not be matched to reference images.
    """

    results: list[PairSimilarityResult]
    level_stats: dict[int | str, LevelStatistics]
    distortion_level_stats: dict[tuple[str, int | str], LevelStatistics]
    unmatched_files: list[Path]

    def format_summary(self) -> str:
        """Generates a human-readable summary table and descriptive analysis.

        Returns:
            str: Multi-line formatted summary string.
        """
        header_title = (
            f"{'Nível':^7} | {'Qtd':^6} | {'SSIM Médio':^11} | {'Mediana':^9} | "
            f"{'Dif. Média (1-SSIM)':^20} | {'Dif. Mediana':^13} | {'Desv. Padrão':^12}"
        )
        lines: list[str] = [
            "=" * 92,
            "              RESUMO DE SIMILARIDADE POR NÍVEL DE DISTORÇÃO (SSIM)",
            "=" * 92,
            header_title,
            "-" * 92,
        ]

        sorted_levels = sorted(self.level_stats.keys(), key=_sort_key_level)
        for lvl in sorted_levels:
            stat = self.level_stats[lvl]
            diff_mean_pct = (
                stat.mean_difference * 100.0
                if not math.isnan(stat.mean_difference)
                else float("nan")
            )
            diff_med_pct = (
                stat.median_difference * 100.0
                if not math.isnan(stat.median_difference)
                else float("nan")
            )

            lines.append(
                f"{lvl!s:^7} | {stat.count:^6} | {stat.mean_ssim:^11.4f} | {stat.median_ssim:^9.4f} | "
                f"{stat.mean_difference:>7.4f} ({diff_mean_pct:>5.2f}%) | "
                f"{stat.median_difference:>6.4f} ({diff_med_pct:>5.2f}%) | "
                f"{stat.std_ssim:^12.4f}"
            )

        lines.extend(
            [
                "-" * 92,
                "",
                "Conclusão Descritiva:",
            ]
        )

        for lvl in sorted_levels:
            stat = self.level_stats[lvl]
            diff_mean_pct = (
                stat.mean_difference * 100.0
                if not math.isnan(stat.mean_difference)
                else float("nan")
            )
            lines.append(
                f" • Nível {lvl}: As imagens originais diferem em média de {stat.mean_difference:.4f} "
                f"({diff_mean_pct:.2f}%) [mediana: {stat.median_difference:.4f}] "
                f"(SSIM médio: {stat.mean_ssim:.4f}, mediana: {stat.median_ssim:.4f})"
            )

        # Distortions breakdown if available
        distortions = sorted({d for d, _ in self.distortion_level_stats})
        if distortions:
            lines.extend(
                [
                    "",
                    "=" * 92,
                    "                    DETALHAMENTO POR DISTORÇÃO E NÍVEL",
                    "=" * 92,
                    f"{'Distorção':<16} | {'Nível':^7} | {'Qtd':^5} | {'SSIM Médio':^11} | {'SSIM Mediana':^12} | {'Dif. Média':^11}",
                    "-" * 92,
                ]
            )
            for dist in distortions:
                for lvl in sorted_levels:
                    key = (dist, lvl)
                    if key in self.distortion_level_stats:
                        d_stat = self.distortion_level_stats[key]
                        lines.append(
                            f"{dist:<16} | {lvl!s:^7} | {d_stat.count:^5} | {d_stat.mean_ssim:^11.4f} | "
                            f"{d_stat.median_ssim:^12.4f} | {d_stat.mean_difference:^11.4f}"
                        )
            lines.append("-" * 92)

        if self.unmatched_files:
            lines.append(
                f"\n[Aviso] {len(self.unmatched_files)} arquivos distorcidos não puderam ser pareados."
            )

        return "\n".join(lines)

    def save_csv(self, output_path: str | Path) -> None:
        """Saves detailed pairwise comparison results to a CSV file.

        Args:
            output_path: File path where CSV will be stored.
        """
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        fieldnames = [
            "reference_stem",
            "reference_path",
            "distorted_path",
            "distortion",
            "level",
            "ssim",
            "difference",
        ]

        with path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for r in self.results:
                writer.writerow(
                    {
                        "reference_stem": r.ref_stem,
                        "reference_path": str(r.ref_path),
                        "distorted_path": str(r.dist_path),
                        "distortion": r.distortion,
                        "level": r.level,
                        "ssim": f"{r.ssim:.6f}" if not math.isnan(r.ssim) else "NaN",
                        "difference": f"{r.difference:.6f}"
                        if not math.isnan(r.difference)
                        else "NaN",
                    }
                )


def _sort_key_level(lvl: int | str) -> tuple[int, int | str]:
    """Helper sort key ordering numeric levels first, then string levels."""
    if isinstance(lvl, int):
        return (0, lvl)
    if isinstance(lvl, str) and lvl.isdigit():
        return (0, int(lvl))
    return (1, str(lvl))


def parse_distorted_filename(filename: str | Path) -> tuple[str, str, int | str] | None:
    """Parses a distorted filename in the format `<ref_stem>.<distortion>.<level>.<ext>`.

    Args:
        filename: Filename or Path of the distorted image.

    Returns:
        tuple[str, str, int | str] | None: (ref_stem, distortion, level) or None if format does not match.
    """
    stem = Path(filename).stem
    match = DISTORTED_NAME_REGEX.match(stem)
    if not match:
        return None

    ref_stem = match.group("ref")
    distortion = match.group("distortion")
    level_raw = match.group("level")
    level: int | str = int(level_raw) if level_raw.isdigit() else level_raw

    return ref_stem, distortion, level


def discover_distortion_pairs(
    ref_dir: str | Path,
    dist_dir: str | Path,
    *,
    recursive: bool = True,
) -> tuple[list[DistortionPair], list[Path]]:
    """Scans reference and distorted directories to build matched comparison pairs.

    Args:
        ref_dir: Path to directory containing pristine reference images.
        dist_dir: Path to directory (or tree of directories) containing distorted images.
        recursive: Whether to scan distorted directory recursively.

    Returns:
        tuple[list[DistortionPair], list[Path]]: List of matched pairs and list of unmatched files.

    Raises:
        FileNotFoundError: If ref_dir or dist_dir does not exist.
        ValueError: If ref_dir contains no supported image files.
    """
    ref_path = Path(ref_dir)
    dist_path = Path(dist_dir)

    if not ref_path.is_dir():
        raise FileNotFoundError(f"Reference directory not found: {ref_path}")
    if not dist_path.is_dir():
        raise FileNotFoundError(f"Distorted directory not found: {dist_path}")

    # Build reference lookup mapping stem -> Path
    ref_map: dict[str, Path] = {}
    for p in ref_path.iterdir():
        if p.is_file() and p.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS:
            ref_map[p.stem] = p

    if not ref_map:
        raise ValueError(f"No valid reference images found in: {ref_path}")

    # Discover distorted images
    dist_iterator = dist_path.rglob("*") if recursive else dist_path.glob("*")
    matched_pairs: list[DistortionPair] = []
    unmatched_files: list[Path] = []
    resolved_ref = ref_path.resolve()

    for file_path in dist_iterator:
        if (
            not file_path.is_file()
            or file_path.suffix.lower() not in SUPPORTED_IMAGE_EXTENSIONS
        ):
            continue

        # Skip pristine reference files if ref_dir is located inside dist_dir
        try:
            if file_path.resolve().is_relative_to(resolved_ref):
                continue
        except (ValueError, AttributeError):
            pass

        parsed = parse_distorted_filename(file_path)
        if parsed is None:
            unmatched_files.append(file_path)
            continue

        ref_stem, distortion, level = parsed
        ref_file = ref_map.get(ref_stem)
        if ref_file is not None:
            matched_pairs.append(
                DistortionPair(
                    ref_path=ref_file,
                    dist_path=file_path,
                    ref_stem=ref_stem,
                    distortion=distortion,
                    level=level,
                )
            )
        else:
            unmatched_files.append(file_path)

    return matched_pairs, unmatched_files


def _evaluate_single_pair(pair: DistortionPair) -> PairSimilarityResult:
    """Worker function to compute SSIM similarity for a single image pair.

    Args:
        pair: DistortionPair containing reference and candidate file paths.

    Returns:
        PairSimilarityResult: Result data object with computed SSIM and difference.
    """
    try:
        score = compare_images_ssim(pair.ref_path, pair.dist_path, auto_resize=False)
    except Exception as exc:  # noqa: BLE001 - return NaN on corrupted image decoding
        logger.warning(
            "Failed computing SSIM between %s and %s: %s",
            pair.ref_path.name,
            pair.dist_path.name,
            exc,
        )
        score = float("nan")

    diff = 1.0 - score if not math.isnan(score) else float("nan")

    return PairSimilarityResult(
        ref_stem=pair.ref_stem,
        ref_path=pair.ref_path,
        dist_path=pair.dist_path,
        distortion=pair.distortion,
        level=pair.level,
        ssim=score,
        difference=diff,
    )


def compute_level_statistics(
    results: Sequence[PairSimilarityResult],
) -> dict[int | str, LevelStatistics]:
    """Computes mean, median, standard deviation, and difference statistics per level.

    Args:
        results: Sequence of pairwise comparison results.

    Returns:
        dict[int | str, LevelStatistics]: Map from level to aggregated statistics.
    """
    scores_by_level: dict[int | str, list[float]] = defaultdict(list)
    for r in results:
        scores_by_level[r.level].append(r.ssim)

    stats: dict[int | str, LevelStatistics] = {}
    for lvl, raw_scores in scores_by_level.items():
        valid_scores = [s for s in raw_scores if not math.isnan(s)]
        count = len(raw_scores)

        if not valid_scores:
            stats[lvl] = LevelStatistics(
                level=lvl,
                count=count,
                mean_ssim=float("nan"),
                median_ssim=float("nan"),
                std_ssim=float("nan"),
                min_ssim=float("nan"),
                max_ssim=float("nan"),
                mean_difference=float("nan"),
                median_difference=float("nan"),
            )
            continue

        arr = np.array(valid_scores, dtype=np.float64)
        mean_s = float(np.mean(arr))
        median_s = float(np.median(arr))
        std_s = float(np.std(arr))
        min_s = float(np.min(arr))
        max_s = float(np.max(arr))
        diff_arr = 1.0 - arr
        mean_d = float(np.mean(diff_arr))
        median_d = float(np.median(diff_arr))

        stats[lvl] = LevelStatistics(
            level=lvl,
            count=count,
            mean_ssim=mean_s,
            median_ssim=median_s,
            std_ssim=std_s,
            min_ssim=min_s,
            max_ssim=max_s,
            mean_difference=mean_d,
            median_difference=median_d,
        )

    return stats


def compute_distortion_level_statistics(
    results: Sequence[PairSimilarityResult],
) -> dict[tuple[str, int | str], LevelStatistics]:
    """Computes statistics grouped by (distortion, level).

    Args:
        results: Sequence of pairwise comparison results.

    Returns:
        dict[tuple[str, int | str], LevelStatistics]: Map from (distortion, level) to statistics.
    """
    grouped: dict[tuple[str, int | str], list[float]] = defaultdict(list)
    for r in results:
        grouped[(r.distortion, r.level)].append(r.ssim)

    stats: dict[tuple[str, int | str], LevelStatistics] = {}
    for (dist, lvl), raw_scores in grouped.items():
        valid_scores = [s for s in raw_scores if not math.isnan(s)]
        count = len(raw_scores)

        if not valid_scores:
            stats[(dist, lvl)] = LevelStatistics(
                level=lvl,
                count=count,
                mean_ssim=float("nan"),
                median_ssim=float("nan"),
                std_ssim=float("nan"),
                min_ssim=float("nan"),
                max_ssim=float("nan"),
                mean_difference=float("nan"),
                median_difference=float("nan"),
            )
            continue

        arr = np.array(valid_scores, dtype=np.float64)
        mean_s = float(np.mean(arr))
        median_s = float(np.median(arr))
        std_s = float(np.std(arr))
        min_s = float(np.min(arr))
        max_s = float(np.max(arr))
        diff_arr = 1.0 - arr
        mean_d = float(np.mean(diff_arr))
        median_d = float(np.median(diff_arr))

        stats[(dist, lvl)] = LevelStatistics(
            level=lvl,
            count=count,
            mean_ssim=mean_s,
            median_ssim=median_s,
            std_ssim=std_s,
            min_ssim=min_s,
            max_ssim=max_s,
            mean_difference=mean_d,
            median_difference=median_d,
        )

    return stats


def evaluate_dataset_similarity(
    ref_dir: str | Path,
    dist_dir: str | Path,
    *,
    max_workers: int | None = None,
    show_progress: bool = True,
    recursive: bool = True,
) -> DatasetSimilarityReport:
    """Evaluates SSIM similarity between reference images and their distorted counterparts.

    Parses distortion level from filename `<ref_stem>.<distortion>.<level>.<ext>`
    and aggregates mean and median similarity and difference for each level.

    Args:
        ref_dir: Directory containing original reference images.
        dist_dir: Directory containing distorted images across levels.
        max_workers: Maximum number of worker processes (defaults to CPU count).
        show_progress: Whether to display a progress bar in the terminal.
        recursive: Whether to search for distorted images recursively in subdirectories.

    Returns:
        DatasetSimilarityReport: Contains all pairwise scores, level statistics, and summary methods.
    """
    pairs, unmatched = discover_distortion_pairs(ref_dir, dist_dir, recursive=recursive)

    if not pairs:
        logger.warning("No matched reference/distorted image pairs found to evaluate.")
        return DatasetSimilarityReport(
            results=[],
            level_stats={},
            distortion_level_stats={},
            unmatched_files=unmatched,
        )

    # Optional tqdm progress bar
    progress_bar: Any = None
    if show_progress:
        try:
            from tqdm import tqdm

            progress_bar = tqdm(
                total=len(pairs), desc="Avaliando Similaridade", unit="img"
            )
        except ImportError:
            progress_bar = None

    results: list[PairSimilarityResult] = []

    # Run execution (parallel if max_workers != 1 and pairs > 1)
    if max_workers == 1 or len(pairs) == 1:
        for pair in pairs:
            res = _evaluate_single_pair(pair)
            results.append(res)
            if progress_bar:
                progress_bar.update(1)
    else:
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            future_to_pair = {
                executor.submit(_evaluate_single_pair, p): p for p in pairs
            }
            for future in as_completed(future_to_pair):
                res = future.result()
                results.append(res)
                if progress_bar:
                    progress_bar.update(1)

    if progress_bar:
        progress_bar.close()

    level_stats = compute_level_statistics(results)
    distortion_level_stats = compute_distortion_level_statistics(results)

    return DatasetSimilarityReport(
        results=results,
        level_stats=level_stats,
        distortion_level_stats=distortion_level_stats,
        unmatched_files=unmatched,
    )
