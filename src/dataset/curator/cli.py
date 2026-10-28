"""Command-line interface (CLI) for the dataset curation tool."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from dataset.curator.config import CurateConfig
from dataset.curator.curator import DatasetCurator


def build_parser() -> argparse.ArgumentParser:
    """Constructs the command-line argument parser with default quality thresholds."""
    parser = argparse.ArgumentParser(
        prog="curate-dataset",
        description="Filter and curate a pristine reference image dataset based on cascaded objective quality metrics.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Path arguments
    parser.add_argument(
        "--input-dir",
        "-i",
        type=Path,
        required=True,
        help="Input directory containing raw high-resolution images.",
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        type=Path,
        required=True,
        help="Output directory to save accepted pristine image crops.",
    )
    parser.add_argument(
        "--audit-csv",
        "-a",
        type=Path,
        default=None,
        help="Destination path for structured CSV audit log (default: <output-dir>/curation_audit.csv).",
    )

    # Geometric parameters
    parser.add_argument(
        "--target-dim",
        type=int,
        default=512,
        help="Target dimension for the shorter edge after Lanczos4 downsampling.",
    )
    parser.add_argument(
        "--crop-size",
        type=int,
        nargs=2,
        metavar=("HEIGHT", "WIDTH"),
        default=[512, 512],
        help="Dimensions for the center crop (height width).",
    )

    # Cascaded Quality Gate Thresholds
    parser.add_argument(
        "--max-clipping-ratio",
        type=float,
        default=0.01,
        help="Max allowable clipping ratio (proportion <= 1 or >= 254). 0.01 = 1.0%%.",
    )
    parser.add_argument(
        "--min-luminance-std",
        type=float,
        default=40.0,
        help="Min allowable standard deviation of grayscale luminance for dynamic range/contrast.",
    )
    parser.add_argument(
        "--min-entropy",
        type=float,
        default=7.0,
        help="Min allowable Shannon entropy in bits/pixel.",
    )
    parser.add_argument(
        "--min-laplacian-var",
        type=float,
        default=120.0,
        help="Min allowable variance of Laplacian operator for sharpness.",
    )
    parser.add_argument(
        "--max-niqe",
        type=float,
        default=3.5,
        help="Max allowable NIQE score (lower indicates higher perceptual quality).",
    )
    parser.add_argument(
        "--csv-score-path",
        "--quality-csv",
        dest="csv_score_path",
        type=Path,
        default=None,
        help="Optional path to external CSV containing image quality scores.",
    )
    parser.add_argument(
        "--csv-score-column",
        "--quality-csv-column",
        dest="csv_score_column",
        type=str,
        default=None,
        help="Column name in external CSV containing the quality scores.",
    )
    parser.add_argument(
        "--min-csv-score",
        "--min-quality-score",
        dest="min_csv_score",
        type=float,
        default=0.7,
        help="Minimum allowable quality score from external CSV (default: 0.7).",
    )

    # Execution & System parameters
    parser.add_argument(
        "--device",
        choices=["auto", "cuda", "cpu"],
        default="auto",
        help="Device to use for tensor calculations in pyiqa ('auto' prefers CUDA if available).",
    )
    parser.add_argument(
        "--png-compression",
        type=int,
        default=0,
        choices=range(10),
        help="PNG compression level (0 is uncompressed lossless, 9 is maximum compression).",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Disable interactive progress bar.",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable detailed debug logging output.",
    )

    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI entry point for dataset curation.

    Args:
        argv: Command-line arguments. If None, reads from sys.argv[1:].

    Returns:
        int: Exit status code (0 for success, 1 on failure).
    """
    parser = build_parser()
    args = parser.parse_args(argv)

    # Logging setup
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    logger = logging.getLogger("dataset")

    audit_csv = args.audit_csv or (args.output_dir / "curation_audit.csv")

    try:
        config = CurateConfig(
            input_dir=args.input_dir,
            output_dir=args.output_dir,
            audit_csv_path=audit_csv,
            target_dim=args.target_dim,
            crop_size=tuple(args.crop_size),  # type: ignore[arg-type]
            max_clipping_ratio=args.max_clipping_ratio,
            min_luminance_std=args.min_luminance_std,
            min_entropy=args.min_entropy,
            min_laplacian_var=args.min_laplacian_var,
            max_niqe=args.max_niqe,
            device=args.device,
            png_compression=args.png_compression,
            csv_score_path=args.csv_score_path,
            csv_score_column=args.csv_score_column,
            min_csv_score=args.min_csv_score,
        )

        logger.info("Initializing dataset curation pipeline...")
        logger.info("Input directory: %s", config.input_dir)
        logger.info("Output directory: %s", config.output_dir)
        logger.info("Audit log: %s", config.audit_csv_path)

        curator = DatasetCurator(config=config)
        summary = curator.run_curation(show_progress=not args.quiet)

        print("\n=== Dataset Curation Summary ===")
        print(f"Total processed files : {summary['total_images']}")
        print(
            f"Accepted pristine     : {summary['accepted_pristine']} ({summary['acceptance_rate_pct']:.2f}%)"
        )
        print(f"Rejected non-pristine : {summary['rejected']}")
        if summary["rejections_by_reason"]:
            print("Rejections breakdown  :")
            for reason, count in sorted(
                summary["rejections_by_reason"].items(), key=lambda x: -x[1]
            ):
                print(f"  - {reason:25s}: {count}")
        print(f"Audit log saved to    : {summary['audit_file']}")
        print("================================\n")
        return 0

    except Exception:
        logger.exception("Curation pipeline terminated with error")
        return 1


if __name__ == "__main__":
    sys.exit(main())
