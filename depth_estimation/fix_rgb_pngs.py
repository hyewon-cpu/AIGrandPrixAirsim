from __future__ import annotations

"""Fix already-collected RGB PNG files for TAO DepthNet.

The earlier collector path wrote RGB arrays directly with OpenCV. To make the
saved PNGs match TAO's loader, this utility swaps the channels once on disk.
It only touches ``rgb/*.png`` files under dataset run directories.
"""

from argparse import ArgumentParser
from pathlib import Path

import cv2


DEFAULT_DATASET_ROOT = Path(__file__).resolve().parent / "datasets"


def iter_rgb_pngs(dataset_root: Path):
    for run_dir in sorted(dataset_root.iterdir()):
        rgb_dir = run_dir / "rgb"
        if not run_dir.is_dir() or not rgb_dir.is_dir():
            continue
        for path in sorted(rgb_dir.glob("*.png")):
            yield path


def build_args():
    parser = ArgumentParser(description="Fix RGB PNGs that were saved with swapped channels.")
    parser.add_argument(
        "--dataset_root",
        type=str,
        default=str(DEFAULT_DATASET_ROOT),
        help="Root directory containing dataset run folders.",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        default=False,
        help="List files that would be fixed without modifying them.",
    )
    parser.add_argument(
        "--backup_ext",
        type=str,
        default=".bak",
        help="Backup extension to use before overwriting each file. Use an empty string to disable backups.",
    )
    return parser.parse_args()


def main() -> None:
    args = build_args()
    dataset_root = Path(args.dataset_root).expanduser().resolve()
    if not dataset_root.exists():
        raise FileNotFoundError(f"Dataset root does not exist: {dataset_root}")

    png_files = list(iter_rgb_pngs(dataset_root))
    if not png_files:
        print(f"No RGB PNG files found under {dataset_root}")
        return

    print(f"Found {len(png_files)} RGB PNG files under {dataset_root}")
    for path in png_files:
        if args.dry_run:
            print(f"[dry-run] would fix {path}")
            continue

        img = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError(f"Failed to read image: {path}")

        fixed = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        backup_ext = args.backup_ext
        if backup_ext:
            backup_path = path.with_name(path.name + backup_ext)
            backup_path.write_bytes(path.read_bytes())

        if not cv2.imwrite(str(path), fixed):
            raise IOError(f"Failed to write fixed image: {path}")

        print(f"fixed {path}")


if __name__ == "__main__":
    main()
