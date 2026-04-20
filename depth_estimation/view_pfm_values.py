from __future__ import annotations

from argparse import ArgumentParser
from pathlib import Path
import re

import numpy as np


def read_pfm(path: Path) -> np.ndarray:
    with open(path, "rb") as f:
        header = f.readline().decode("ascii").rstrip()
        if header not in ("Pf", "PF"):
            raise ValueError(f"Not a PFM file: {path}")

        dim_line = f.readline().decode("ascii").strip()
        while dim_line.startswith("#"):
            dim_line = f.readline().decode("ascii").strip()

        dims = re.findall(r"\d+", dim_line)
        if len(dims) != 2:
            raise ValueError(f"Invalid PFM dimension line: {dim_line}")
        width, height = map(int, dims)

        scale = float(f.readline().decode("ascii").strip())
        endian = "<" if scale < 0 else ">"
        channels = 1 if header == "Pf" else 3

        data = np.fromfile(f, endian + "f")
        expected = width * height * channels
        if data.size != expected:
            raise ValueError(
                f"PFM payload size mismatch: expected {expected}, got {data.size}"
            )

        shape = (height, width) if channels == 1 else (height, width, 3)
        data = np.reshape(data, shape)
        # PFM rows are stored from bottom to top.
        return np.flipud(data)


def build_args():
    parser = ArgumentParser(description="Inspect raw float values in a PFM file.")
    parser.add_argument("pfm_path", type=str, help="Path to a .pfm file")
    parser.add_argument(
        "--x",
        type=int,
        default=None,
        help="Optional x coordinate (column) to print.",
    )
    parser.add_argument(
        "--y",
        type=int,
        default=None,
        help="Optional y coordinate (row) to print.",
    )
    return parser.parse_args()


def main():
    args = build_args()
    path = Path(args.pfm_path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"PFM does not exist: {path}")

    depth = read_pfm(path)
    if depth.ndim != 2:
        raise ValueError(f"Expected single-channel depth PFM, got shape {depth.shape}")

    h, w = depth.shape
    finite = np.isfinite(depth)
    finite_ratio = float(finite.mean() * 100.0)

    print(f"path: {path}")
    print(f"shape: {depth.shape}, dtype: {depth.dtype}")
    print(f"finite_ratio: {finite_ratio:.2f}%")
    if np.any(finite):
        print(f"min: {float(np.nanmin(depth))}")
        print(f"max: {float(np.nanmax(depth))}")
        print(f"mean: {float(np.nanmean(depth))}")
    else:
        print("min/max/mean: no finite values")

    cy, cx = h // 2, w // 2
    print(f"center ({cx}, {cy}): {float(depth[cy, cx])}")

    if args.x is not None and args.y is not None:
        if not (0 <= args.x < w and 0 <= args.y < h):
            raise ValueError(
                f"Requested pixel ({args.x}, {args.y}) is out of bounds for {w}x{h}"
            )
        print(f"pixel ({args.x}, {args.y}): {float(depth[args.y, args.x])}")


if __name__ == "__main__":
    main()

