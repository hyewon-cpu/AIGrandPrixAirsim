from __future__ import annotations

"""Split collected dataset pairs into train/val/test text files."""

from argparse import ArgumentParser
from datetime import datetime
from pathlib import Path
import random
import sys

DEPTH_ROOT = Path(__file__).resolve().parent
DATASETS_ROOT = DEPTH_ROOT / "datasets"
OUTPUT_DIR = DEPTH_ROOT / "dataforuse"

if str(DEPTH_ROOT) not in sys.path:
    sys.path.insert(0, str(DEPTH_ROOT))


def read_pairs_file(path: Path) -> list[str]:
    pairs: list[str] = []
    with open(path, "r", encoding="utf-8") as f:
        for line_number, raw_line in enumerate(f, start=1):
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 2:
                raise ValueError(
                    f"Invalid pair on line {line_number} of {path}: "
                    "expected 'rgb_path depth_path'"
                )
            pairs.append(f"{parts[0]} {parts[1]}")
    return pairs


def split_pairs(
    pairs: list[str],
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    shuffle: bool = True,
    seed: int = 1234,
) -> tuple[list[str], list[str], list[str]]:
    total_ratio = train_ratio + val_ratio + test_ratio
    if abs(total_ratio - 1.0) > 1e-6:
        raise ValueError(
            f"Split ratios must sum to 1.0, got "
            f"train={train_ratio}, val={val_ratio}, test={test_ratio}"
        )

    items = list(pairs)
    if shuffle:
        random.Random(seed).shuffle(items)

    total = len(items)
    train_count = int(total * train_ratio)
    val_count = int(total * val_ratio)
    test_count = total - train_count - val_count

    train_pairs = items[:train_count]
    val_pairs = items[train_count : train_count + val_count]
    test_pairs = items[train_count + val_count : train_count + val_count + test_count]
    return train_pairs, val_pairs, test_pairs


def write_lines(path: Path, lines: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for line in lines:
            f.write(line.rstrip() + "\n")


def _dataset_sort_key(dataset_dir: Path) -> tuple[datetime, str]:
    """Prefer the timestamp embedded in the run directory name, then mtime."""

    timestamp_text = dataset_dir.name[:22]
    try:
        timestamp = datetime.strptime(timestamp_text, "%Y%m%d_%H%M%S_%f")
    except ValueError:
        timestamp = datetime.fromtimestamp(dataset_dir.stat().st_mtime)
    return timestamp, dataset_dir.name


def find_latest_dataset_dir(datasets_root: Path) -> Path:
    datasets_root = Path(datasets_root).expanduser().resolve()
    if not datasets_root.exists():
        raise FileNotFoundError(f"Datasets directory does not exist: {datasets_root}")

    candidates = [
        child
        for child in datasets_root.iterdir()
        if child.is_dir() and (child / "pairs.txt").exists()
    ]
    if not candidates:
        raise FileNotFoundError(
            f"No dataset directories with pairs.txt were found under {datasets_root}"
        )

    return max(candidates, key=_dataset_sort_key)


def _resolve_dataset_dir_input(dataset_dir: Path | str) -> Path:
    raw = Path(dataset_dir).expanduser()
    if raw.is_absolute():
        return raw.resolve()

    cwd_candidate = (Path.cwd() / raw).resolve()
    if cwd_candidate.exists():
        return cwd_candidate

    datasets_candidate = (DATASETS_ROOT / raw).resolve()
    if datasets_candidate.exists():
        return datasets_candidate

    if len(raw.parts) == 1:
        return datasets_candidate
    return cwd_candidate


def _validate_dataset_dir(dataset_dir: Path) -> Path:
    dataset_dir = _resolve_dataset_dir_input(dataset_dir)
    if not dataset_dir.exists():
        raise FileNotFoundError(f"Dataset directory does not exist: {dataset_dir}")
    if not (dataset_dir / "pairs.txt").exists():
        raise FileNotFoundError(f"pairs.txt does not exist in: {dataset_dir}")
    return dataset_dir


def read_pairs_from_dirs(dataset_dirs: list[Path]) -> tuple[list[str], list[Path]]:
    resolved_dirs: list[Path] = []
    merged_pairs: list[str] = []
    for dataset_dir in dataset_dirs:
        valid_dir = _validate_dataset_dir(dataset_dir)
        resolved_dirs.append(valid_dir)
        merged_pairs.extend(read_pairs_file(valid_dir / "pairs.txt"))
    return merged_pairs, resolved_dirs


def build_args():
    parser = ArgumentParser(
        description="Split AirSim dataset pairs into train/val/test files."
    )
    parser.add_argument(
        "--data_type",
        type=str,
        default="latest",
        help=(
            "Directory containing dataset run folders, or keyword 'latest' to use "
            f"the latest run under {DATASETS_ROOT}, or keyword 'multiple' to merge "
            "pairs from --dataset_dirs."
        ),
    )
    parser.add_argument(
        "--dataset_dirs",
        type=str,
        nargs="+",
        default=None,
        help=(
            "Used when --data_type multiple. Provide one or more dataset run "
            "folders that each contain pairs.txt. Dataset folder names like "
            "'20260420_111350_260747' are also accepted."
        ),
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=str(OUTPUT_DIR),
        help="Directory where split files will be written.",
    )
    parser.add_argument("--train_ratio", type=float, default=0.8)
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--test_ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument(
        "--no_shuffle",
        action="store_true",
        default=False,
        help="Keep the original line order instead of shuffling before splitting.",
    )
    return parser.parse_args()


def main():
    args = build_args()

    selected_dataset_dirs: list[Path]

    if args.data_type == "latest":
        selected_dataset_dirs = [find_latest_dataset_dir(DATASETS_ROOT)]
    elif args.data_type == "multiple":
        if not args.dataset_dirs:
            raise ValueError(
                "--data_type multiple requires at least one path in --dataset_dirs"
            )
        selected_dataset_dirs = [Path(path) for path in args.dataset_dirs]
    else:
        selected_dataset_dirs = [_validate_dataset_dir(Path(args.data_type))]

    output_dir = Path(args.output_dir).expanduser().resolve()

    pairs, resolved_dataset_dirs = read_pairs_from_dirs(selected_dataset_dirs)
    if not pairs:
        raise ValueError("No pairs were found in the selected dataset directories.")

    train_pairs, val_pairs, test_pairs = split_pairs(
        pairs,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        shuffle=not args.no_shuffle,
        seed=args.seed,
    )

    output_dir.mkdir(parents=True, exist_ok=True)

    train_path = output_dir / "train.txt"
    val_path = output_dir / "val.txt"
    test_path = output_dir / "test.txt"

    write_lines(train_path, train_pairs)
    write_lines(val_path, val_pairs)
    write_lines(test_path, test_pairs)

    if len(resolved_dataset_dirs) == 1:
        print(f"Using dataset: {resolved_dataset_dirs[0]}")
    else:
        print("Using multiple datasets:")
        for dataset_dir in resolved_dataset_dirs:
            print(f"  - {dataset_dir}")
        print(f"Merged total pairs: {len(pairs)}")
    print(f"Wrote {len(train_pairs)} train pairs to {train_path}")
    print(f"Wrote {len(val_pairs)} val pairs to {val_path}")
    print(f"Wrote {len(test_pairs)} test pairs to {test_path}")


if __name__ == "__main__":
    main()
