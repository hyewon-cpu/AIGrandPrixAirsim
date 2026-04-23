#!/usr/bin/env python3

"""Merge one or more gate datasets and create train/val/test splits.

This script reads `pairs.txt` files from dataset folders inside `gate_detection/datasets`,
shuffles the combined entries once, and writes:

- `train.txt` 85%
- `val.txt` 10%
- `test.txt` 5%

The output files are written into `gate_detection/dataforuse/` by default.
"""

from __future__ import annotations

from argparse import ArgumentParser
from pathlib import Path
import random


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET_ROOT = REPO_ROOT / "gate_detection" / "datasets"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "gate_detection" / "dataforuse"


def build_args():
    parser = ArgumentParser(description="Create train/val/test splits from gate datasets.")
    parser.add_argument(
        "--dataset_root",
        type=str,
        default=str(DEFAULT_DATASET_ROOT),
        help="Folder that contains dataset subfolders, each with a pairs.txt file.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=str(DEFAULT_OUTPUT_DIR),
        help="Where to write train.txt, val.txt, and test.txt.",
    )
    parser.add_argument(
        "--datasets",
        nargs="*",
        default=[],
        help="Optional dataset folder names to include. If omitted, all valid datasets are used.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1234,
        help="Random seed used before shuffling the combined pairs.",
    )
    return parser.parse_args()


def read_pairs_file(pairs_path: Path) -> list[str]:
    lines: list[str] = []
    for raw_line in pairs_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        lines.append(line)
    return lines


def resolve_dataset_dirs(dataset_root: Path, selected_names: list[str]) -> list[Path]:
    if selected_names:
        dataset_dirs = [dataset_root / name for name in selected_names]
    else:
        dataset_dirs = sorted(
            path for path in dataset_root.iterdir() if path.is_dir()
        )

    valid_dirs: list[Path] = []
    missing = []
    for dataset_dir in dataset_dirs:
        pairs_path = dataset_dir / "pairs.txt"
        if pairs_path.exists():
            valid_dirs.append(dataset_dir)
        else:
            missing.append(dataset_dir)

    if missing:
        missing_text = "\n".join(f"- {path}" for path in missing)
        raise FileNotFoundError(
            "The following dataset folders do not contain pairs.txt:\n"
            f"{missing_text}"
        )

    if not valid_dirs:
        raise FileNotFoundError(
            f"No dataset folders with pairs.txt found under {dataset_root}"
        )

    return valid_dirs


def split_entries(entries: list[str]) -> tuple[list[str], list[str], list[str]]:
    total = len(entries)
    train_count = int(total * 0.85)
    val_count = int(total * 0.10)
    test_count = total - train_count - val_count

    train_entries = entries[:train_count]
    val_entries = entries[train_count : train_count + val_count]
    test_entries = entries[train_count + val_count : train_count + val_count + test_count]
    return train_entries, val_entries, test_entries


def write_split(path: Path, entries: list[str]) -> None:
    path.write_text("\n".join(entries) + ("\n" if entries else ""), encoding="utf-8")


def main():
    args = build_args()
    dataset_root = Path(args.dataset_root).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()

    if not dataset_root.exists():
        raise FileNotFoundError(f"Dataset root does not exist: {dataset_root}")

    dataset_dirs = resolve_dataset_dirs(dataset_root, list(args.datasets))

    all_entries: list[str] = []
    for dataset_dir in dataset_dirs:
        pairs_path = dataset_dir / "pairs.txt"
        entries = read_pairs_file(pairs_path)
        all_entries.extend(entries)

    if not all_entries:
        raise FileNotFoundError(
            f"No pair entries were found in the selected datasets under {dataset_root}"
        )

    rng = random.Random(int(args.seed))
    rng.shuffle(all_entries)

    train_entries, val_entries, test_entries = split_entries(all_entries)

    output_dir.mkdir(parents=True, exist_ok=True)
    write_split(output_dir / "train.txt", train_entries)
    write_split(output_dir / "val.txt", val_entries)
    write_split(output_dir / "test.txt", test_entries)

    print(f"Selected datasets: {', '.join(path.name for path in dataset_dirs)}")
    print(f"Total pairs: {len(all_entries)}")
    print(f"train: {len(train_entries)}")
    print(f"val:   {len(val_entries)}")
    print(f"test:  {len(test_entries)}")
    print(f"Wrote splits to: {output_dir}")


if __name__ == "__main__":
    main()
