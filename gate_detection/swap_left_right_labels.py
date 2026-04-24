#!/usr/bin/env python3

"""Swap left/right corner labels (TL<->TR, BL<->BR) in LabelMe JSON files.

This edits only the `shapes[*].label` strings and leaves point coordinates,
group IDs, and other metadata unchanged.

Typical usage (dry-run first):
  python3 gate_detection/swap_left_right_labels.py --dataset_root gate_detection/datasets

Apply changes in place:
  python3 gate_detection/swap_left_right_labels.py --dataset_root gate_detection/datasets --in_place
"""

from __future__ import annotations

from argparse import ArgumentParser
from pathlib import Path
import json


SWAP = {
    "TL": "TR",
    "TR": "TL",
    "BL": "BR",
    "BR": "BL",
}


def _match_case(text: str, template: str) -> str:
    if template.islower():
        return text.lower()
    if template.isupper():
        return text.upper()
    return text


def swap_label(label: str) -> tuple[str, bool]:
    """Return (new_label, changed)."""
    raw = str(label)
    parts = raw.split("_")
    if len(parts) == 1:
        token = raw.strip()
        key = token.upper()
        if key in SWAP:
            return _match_case(SWAP[key], token), True
        return raw, False

    token = parts[-1]
    key = token.upper()
    if key not in SWAP:
        return raw, False

    parts[-1] = _match_case(SWAP[key], token)
    return "_".join(parts), True


def process_json(path: Path, *, in_place: bool) -> tuple[int, bool]:
    """Return (num_labels_changed, file_changed)."""
    record = json.loads(path.read_text(encoding="utf-8"))
    shapes = record.get("shapes", [])
    if not isinstance(shapes, list):
        return 0, False

    changed = 0
    for shape in shapes:
        if not isinstance(shape, dict) or "label" not in shape:
            continue
        new_label, did_change = swap_label(shape.get("label", ""))
        if did_change:
            shape["label"] = new_label
            changed += 1

    if changed and in_place:
        path.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
        return changed, True
    return changed, False


def build_args():
    parser = ArgumentParser(description="Swap TL/TR and BL/BR in LabelMe JSON files.")
    parser.add_argument(
        "--dataset_root",
        type=str,
        default=str(Path(__file__).resolve().parent / "datasets"),
        help="Folder to scan recursively for .json files.",
    )
    parser.add_argument(
        "--in_place",
        action="store_true",
        default=False,
        help="Write changes back to disk (default: dry-run only).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Optional max number of JSON files to process (0 means no limit).",
    )
    return parser.parse_args()


def main() -> None:
    args = build_args()
    dataset_root = Path(args.dataset_root).expanduser().resolve()
    if not dataset_root.exists():
        raise FileNotFoundError(f"dataset_root does not exist: {dataset_root}")

    json_paths = sorted(dataset_root.rglob("*.json"))
    if args.limit and args.limit > 0:
        json_paths = json_paths[: int(args.limit)]

    total_files = 0
    changed_files = 0
    changed_labels = 0

    for path in json_paths:
        total_files += 1
        n_changed, file_changed = process_json(path, in_place=bool(args.in_place))
        if n_changed:
            changed_labels += n_changed
        if file_changed:
            changed_files += 1

    mode = "IN-PLACE" if args.in_place else "DRY-RUN"
    print(f"[{mode}] scanned_files={total_files} changed_files={changed_files} changed_labels={changed_labels}")


if __name__ == "__main__":
    main()

