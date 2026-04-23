from __future__ import annotations

"""Flip depth PFM files in one or more dataset runs.

By default this script walks the dataset tree, finds every ``.pfm`` file under a
``depth`` directory, flips the image vertically, and writes it back in place.
You can also pass one or more specific PFM files or directories to flip only
those targets.
"""

from argparse import ArgumentParser
from pathlib import Path
from typing import Iterable
import re

import numpy as np


DEFAULT_DATASET_ROOT = Path(__file__).resolve().parent / "datasets"


def read_pfm(path: Path) -> tuple[np.ndarray, float]:
    """Read a PFM file and return the image data and scale."""
    with open(path, "rb") as f:
        header = f.readline().decode("ascii").rstrip()
        if header not in ("Pf", "PF"):
            raise ValueError(f"Not a PFM file: {path}")

        dim_line = f.readline().decode("ascii").strip()
        dims = re.findall(r"\d+", dim_line)
        if len(dims) != 2:
            raise ValueError(f"Invalid PFM dimension line in {path}: {dim_line}")
        width, height = map(int, dims)

        scale = float(f.readline().decode("ascii").strip())
        endian = "<" if scale < 0 else ">"
        channels = 1 if header == "Pf" else 3

        data = np.fromfile(f, endian + "f")
        expected = width * height * channels
        if data.size != expected:
            raise ValueError(
                f"PFM payload size mismatch for {path}: expected {expected}, got {data.size}"
            )

        shape = (height, width) if channels == 1 else (height, width, 3)
        data = np.reshape(data, shape)
        return data, scale


def write_pfm(path: Path, image: np.ndarray, scale: float) -> None:
    """Write a PFM file without changing row order."""
    if image.dtype != np.float32:
        image = image.astype(np.float32, copy=False)

    if image.ndim == 2:
        header = "Pf"
    elif image.ndim == 3 and image.shape[2] == 3:
        header = "PF"
    else:
        raise ValueError(f"PFM image must be HxW or HxWx3, got {image.shape}")

    with open(path, "wb") as f:
        f.write(f"{header}\n".encode("ascii"))
        f.write(f"{image.shape[1]} {image.shape[0]}\n".encode("ascii"))

        endian = image.dtype.byteorder
        scale_to_write = scale
        if endian == "<" or (endian == "=" and np.little_endian):
            scale_to_write = -abs(scale_to_write)
        else:
            scale_to_write = abs(scale_to_write)
        f.write(f"{scale_to_write}\n".encode("ascii"))
        image.tofile(f)


def iter_pfm_files(dataset_root: Path) -> Iterable[Path]:
    """Yield PFM files found under depth folders."""
    for path in sorted(dataset_root.rglob("*.pfm")):
        if "depth" in path.parts:
            yield path


def build_args():
    parser = ArgumentParser(description="Flip PFM depth maps in dataset runs.")
    parser.add_argument(
        "--dataset_root",
        type=str,
        default=str(DEFAULT_DATASET_ROOT),
        help="Root directory containing dataset run folders.",
    )
    parser.add_argument(
        "--pfm_path",
        type=str,
        nargs="*",
        default=None,
        help=(
            "Optional specific PFM file(s) or directory(ies) to flip. "
            "If omitted, all .pfm files under dataset_root are processed."
        ),
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        default=False,
        help="List files that would be flipped without modifying them.",
    )
    parser.add_argument(
        "--backup_ext",
        type=str,
        default=".bak",
        help="Backup extension to use before overwriting each file. Use an empty string to disable backups.",
    )
    return parser.parse_args()


def resolve_pfm_targets(dataset_root: Path, pfm_paths: list[str] | None) -> list[Path]:
    """Resolve the files to flip.

    If specific paths are provided, each item may be a .pfm file or a directory.
    Directories are searched recursively for .pfm files under depth folders.
    """
    if not pfm_paths:
        return list(iter_pfm_files(dataset_root))

    targets: list[Path] = []
    for raw_path in pfm_paths:
        path = Path(raw_path).expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(f"PFM target does not exist: {path}")
        if path.is_dir():
            targets.extend(iter_pfm_files(path))
        else:
            if path.suffix.lower() != ".pfm":
                raise ValueError(f"Not a .pfm file: {path}")
            targets.append(path)

    # Preserve order but remove duplicates.
    unique_targets = list(dict.fromkeys(targets))
    return sorted(unique_targets)


def main() -> None:
    args = build_args()
    dataset_root = Path(args.dataset_root).expanduser().resolve()
    if not dataset_root.exists():
        raise FileNotFoundError(f"Dataset root does not exist: {dataset_root}")

    pfm_files = resolve_pfm_targets(dataset_root, args.pfm_path)
    if not pfm_files:
        print(f"No PFM depth files found under {dataset_root}")
        return

    if args.pfm_path:
        print(f"Found {len(pfm_files)} selected PFM file(s)")
    else:
        print(f"Found {len(pfm_files)} PFM files under {dataset_root}")

    for path in pfm_files:
        if args.dry_run:
            print(f"[dry-run] would flip {path}")
            continue

        data, scale = read_pfm(path)
        flipped = np.flipud(data)

        backup_ext = args.backup_ext
        if backup_ext:
            backup_path = path.with_name(path.name + backup_ext)
            backup_path.write_bytes(path.read_bytes())

        write_pfm(path, flipped, scale=scale)
        print(f"flipped {path}")


if __name__ == "__main__":
    main()
