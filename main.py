from pathlib import Path
from dataset_curator.config import CurateConfig
from dataset_curator.curator import DatasetCurator

# 1. Custom Configuration
config = CurateConfig(
    input_dir=Path("Datasets/UHD-IQA-database/validation"),
    output_dir=Path("Datasets/pristine2"),
    audit_csv_path=Path("Datasets/pristine2/audit_validation.csv"),
    csv_score_column="quality_mos",
    csv_score_path=Path("Datasets/UHD-IQA-database/uhd-iqa-metadata.csv"),
    target_dim=1800,
    crop_size=(1800, 1800),
    max_clipping_ratio=0.01,
    min_luminance_std=40.0,
    min_entropy=7.0,
    min_laplacian_var=120.0,
    max_niqe=3.5,
    device="auto",
)

# 2. Run Batch Curation Service
curator = DatasetCurator(config=config)
summary = curator.run_curation()
print(f"Accepted {summary['accepted_pristine']} out of {summary['total_images']} images.")
