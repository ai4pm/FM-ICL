#!/usr/bin/env python
from __future__ import annotations

import argparse
import os
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_ROOT = REPO_ROOT / "data" / "tcga"

CANCERS = ("BRCA", "UCEC", "KIRC")
OMICS = ("mRNA", "MicroRNA", "Methylation")
TARGETS = ("OS", "PFI")
EXPECTED_SPLITS_PER_TASK = 30


def bytes_to_human(size: int) -> str:
    value = float(size)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if value < 1024 or unit == "TB":
            return f"{value:.1f}{unit}"
        value /= 1024
    return f"{value:.1f}TB"


def directory_size(path: Path) -> int:
    total = 0
    for item in path.rglob("*"):
        if item.is_file() and not item.is_symlink():
            total += item.stat().st_size
    return total


def expected_tasks() -> list[str]:
    return [
        f"{cancer}_AFR_{omics}_{target}_2"
        for cancer in CANCERS
        for omics in OMICS
        for target in TARGETS
    ]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate the lean TCGA data package required by FM-ICL pca100 experiments."
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path(os.environ.get("FM_ICL_DATA_ROOT", DEFAULT_DATA_ROOT)),
        help="TCGA data root. Defaults to FM_ICL_DATA_ROOT or data/tcga under this repo.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_root = args.data_root.expanduser().resolve()
    split_root = data_root / "Sample_Splits" / "raw_split_tcga_pca100d_ctxae20d"
    ancestry_root = data_root / "ancestral_info"

    errors: list[str] = []

    for filename in ("black_ids.pkl", "white_ids.pkl"):
        path = ancestry_root / filename
        if not path.is_file():
            errors.append(f"missing ancestry file: {path}")

    task_names = expected_tasks()
    for task_name in task_names:
        task_dir = split_root / task_name
        if not task_dir.is_dir():
            errors.append(f"missing task directory: {task_dir}")
            continue
        split_files = sorted(task_dir.glob("3_folds_sample_split_*.pkl"))
        if len(split_files) != EXPECTED_SPLITS_PER_TASK:
            errors.append(
                f"{task_name}: expected {EXPECTED_SPLITS_PER_TASK} split files, found {len(split_files)}"
            )

    if split_root.exists():
        unexpected = sorted(
            path.name
            for path in split_root.iterdir()
            if path.is_dir() and path.name not in task_names
        )
        if unexpected:
            errors.append(
                "unexpected task directories under raw_split_tcga_pca100d_ctxae20d: "
                + ", ".join(unexpected[:20])
                + (" ..." if len(unexpected) > 20 else "")
            )

    total_size = directory_size(data_root) if data_root.exists() else 0

    print(f"data_root: {data_root}")
    print(f"expected_tasks: {len(task_names)}")
    print(f"expected_split_files: {len(task_names) * EXPECTED_SPLITS_PER_TASK}")
    print(f"data_size: {bytes_to_human(total_size)}")

    if errors:
        print("status: FAILED")
        for error in errors:
            print(f"error: {error}")
        raise SystemExit(1)

    print("status: OK")


if __name__ == "__main__":
    main()
