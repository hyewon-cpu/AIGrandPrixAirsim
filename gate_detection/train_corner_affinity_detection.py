#!/usr/bin/env python3

"""Train a gate corner + Part Affinity Field (PAF) detector from LabelMe annotations.

This script follows the pipeline described in the paper excerpt you shared:

- Predict 4 corner confidence maps (TL/TR/BL/BR) as Gaussian heatmaps.
- Predict 4 PAFs (one per gate edge type), each a 2D vector field:
  (TL->TR), (TR->BR), (BR->BL), (BL->TL) => 8 channels total.
- Supervise both map types with MSE.
- At inference time, extract corner candidates via NMS + thresholding, score
  edge candidates using a line integral over the predicted PAF, match edges, and
  assemble corners into per-gate detections.
"""

from __future__ import annotations

from argparse import ArgumentParser
from dataclasses import dataclass
from pathlib import Path
import json
import math
import random
import re

import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from corner_unet import CornerUNet


CORNER_NAMES = ("TL", "TR", "BL", "BR")
CORNER_TO_INDEX = {name: idx for idx, name in enumerate(CORNER_NAMES)}
EDGE_TYPES = (("TL", "TR"), ("TR", "BR"), ("BR", "BL"), ("BL", "TL"))
EDGE_TO_INDEX = {edge: idx for idx, edge in enumerate(EDGE_TYPES)}

DEFAULT_SPLIT_DIR = Path(__file__).resolve().parent / "dataforuse"
DEFAULT_CHECKPOINT_DIR = Path(__file__).resolve().parent / "corner_affinity_checkpoints"


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def normalize_corner_label(label: str) -> str | None:
    text = str(label).strip().lower()
    if re.search(r"\btl\b", text) or text.endswith("_tl") or text.startswith("tl"):
        return "TL"
    if re.search(r"\btr\b", text) or text.endswith("_tr") or text.startswith("tr"):
        return "TR"
    if re.search(r"\bbl\b", text) or text.endswith("_bl") or text.startswith("bl"):
        return "BL"
    if re.search(r"\bbr\b", text) or text.endswith("_br") or text.startswith("br"):
        return "BR"
    return None


def resolve_image_path(json_path: Path, record: dict) -> Path:
    image_path = record.get("imagePath")
    if not image_path:
        raise ValueError(f"Missing imagePath in {json_path}")

    candidate = (json_path.parent / image_path).resolve()
    if candidate.exists():
        return candidate

    candidate = Path(image_path).expanduser().resolve()
    if candidate.exists():
        return candidate

    raise FileNotFoundError(f"Could not resolve image for {json_path}: {image_path}")


def load_pairs_file(pairs_file: str | Path) -> list[tuple[Path, Path]]:
    pairs_path = Path(pairs_file).expanduser().resolve()
    if not pairs_path.exists():
        raise FileNotFoundError(f"pairs file does not exist: {pairs_path}")

    pairs: list[tuple[Path, Path]] = []
    for raw_line in pairs_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        image_candidates = [
            part
            for part in parts
            if part.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".webp"))
        ]
        json_candidates = [part for part in parts if part.lower().endswith(".json")]
        if not image_candidates:
            raise ValueError(f"Could not find image path in pairs line: {raw_line}")
        if not json_candidates:
            raise ValueError(f"Could not find JSON path in pairs line: {raw_line}")
        pairs.append(
            (
                Path(image_candidates[0]).expanduser().resolve(),
                Path(json_candidates[-1]).expanduser().resolve(),
            )
        )
    return pairs


def resolve_split_path(split_dir: str | Path, split_name: str) -> Path:
    split_path = Path(split_dir).expanduser().resolve() / f"{split_name}.txt"
    if not split_path.exists():
        raise FileNotFoundError(f"Split file does not exist: {split_path}")
    return split_path


def affine_matrix(image_size: tuple[int, int], angle_deg: float) -> np.ndarray:
    width, height = image_size
    center = (width * 0.5, height * 0.5)
    return cv2.getRotationMatrix2D(center, angle_deg, 1.0)


def warp_points(points: list[tuple[float, float]], matrix: np.ndarray) -> list[tuple[float, float]]:
    warped = []
    for x, y in points:
        px = matrix[0, 0] * x + matrix[0, 1] * y + matrix[0, 2]
        py = matrix[1, 0] * x + matrix[1, 1] * y + matrix[1, 2]
        warped.append((px, py))
    return warped


def apply_color_jitter_rgb(
    image: np.ndarray,
    brightness: float,
    contrast: float,
    saturation: float,
    hue: float,
) -> np.ndarray:
    out = image.astype(np.float32) / 255.0

    if brightness > 0:
        factor = random.uniform(max(0.0, 1.0 - brightness), 1.0 + brightness)
        out = out * factor

    if contrast > 0:
        factor = random.uniform(max(0.0, 1.0 - contrast), 1.0 + contrast)
        mean = out.mean(axis=(0, 1), keepdims=True)
        out = (out - mean) * factor + mean

    if saturation > 0 or hue > 0:
        hsv = cv2.cvtColor(np.clip(out * 255.0, 0, 255).astype(np.uint8), cv2.COLOR_RGB2HSV)
        h = hsv[:, :, 0].astype(np.float32)
        s = hsv[:, :, 1].astype(np.float32)
        v = hsv[:, :, 2].astype(np.float32)

        if saturation > 0:
            s_factor = random.uniform(max(0.0, 1.0 - saturation), 1.0 + saturation)
            s = np.clip(s * s_factor, 0, 255)

        if hue > 0:
            h_shift = random.uniform(-hue, hue) * 180.0
            h = (h + h_shift) % 180.0

        hsv = np.stack([h, s, v], axis=2).astype(np.uint8)
        out = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB).astype(np.float32) / 255.0

    return np.clip(out, 0.0, 1.0)


def draw_gaussian_heatmap(heatmap: np.ndarray, point: tuple[float, float], sigma: float) -> None:
    x, y = point
    if not np.isfinite(x) or not np.isfinite(y):
        return

    height, width = heatmap.shape[:2]
    xi = int(round(x))
    yi = int(round(y))
    if xi < 0 or yi < 0 or xi >= width or yi >= height:
        return

    radius = max(1, int(math.ceil(3.0 * sigma)))
    left = max(0, xi - radius)
    right = min(width, xi + radius + 1)
    top = max(0, yi - radius)
    bottom = min(height, yi + radius + 1)
    if left >= right or top >= bottom:
        return

    grid_x = np.arange(left, right, dtype=np.float32)
    grid_y = np.arange(top, bottom, dtype=np.float32)
    xx, yy = np.meshgrid(grid_x, grid_y)
    gauss = np.exp(-((xx - x) ** 2 + (yy - y) ** 2) / (2.0 * sigma * sigma))
    heatmap[top:bottom, left:right] = np.maximum(heatmap[top:bottom, left:right], gauss)


def _segment_distance_mask(
    height: int,
    width: int,
    p0: tuple[float, float],
    p1: tuple[float, float],
    max_dist_px: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (mask, xs, ys) for pixels within max_dist_px of segment p0->p1.

    Coordinates are in image pixel space: x=col, y=row.
    """
    x0, y0 = p0
    x1, y1 = p1
    if not (np.isfinite(x0) and np.isfinite(y0) and np.isfinite(x1) and np.isfinite(y1)):
        return np.zeros((0, 0), dtype=bool), np.zeros((0,), dtype=np.int32), np.zeros((0,), dtype=np.int32)

    d = float(max_dist_px)
    xmin = max(0, int(math.floor(min(x0, x1) - d)))
    xmax = min(width - 1, int(math.ceil(max(x0, x1) + d)))
    ymin = max(0, int(math.floor(min(y0, y1) - d)))
    ymax = min(height - 1, int(math.ceil(max(y0, y1) + d)))
    if xmin > xmax or ymin > ymax:
        return np.zeros((0, 0), dtype=bool), np.zeros((0,), dtype=np.int32), np.zeros((0,), dtype=np.int32)

    xs = np.arange(xmin, xmax + 1, dtype=np.float32)
    ys = np.arange(ymin, ymax + 1, dtype=np.float32)
    xx, yy = np.meshgrid(xs, ys)

    vx = float(x1 - x0)
    vy = float(y1 - y0)
    seg_len2 = vx * vx + vy * vy
    if seg_len2 < 1e-6:
        dist2 = (xx - x0) ** 2 + (yy - y0) ** 2
        mask = dist2 <= (d * d)
        return mask, xs.astype(np.int32), ys.astype(np.int32)

    # Project points to segment and compute distances.
    t = ((xx - x0) * vx + (yy - y0) * vy) / seg_len2
    t = np.clip(t, 0.0, 1.0)
    proj_x = x0 + t * vx
    proj_y = y0 + t * vy
    dist2 = (xx - proj_x) ** 2 + (yy - proj_y) ** 2
    mask = dist2 <= (d * d)
    return mask, xs.astype(np.int32), ys.astype(np.int32)


def add_paf_edge(
    paf_accum: np.ndarray,
    paf_count: np.ndarray,
    edge_index: int,
    p0: tuple[float, float],
    p1: tuple[float, float],
    *,
    max_dist_px: float,
) -> None:
    """Accumulate a PAF vector field for one gate edge into (accum,count)."""
    height, width = paf_accum.shape[1], paf_accum.shape[2]
    x0, y0 = p0
    x1, y1 = p1
    dx = float(x1 - x0)
    dy = float(y1 - y0)
    norm = math.hypot(dx, dy)
    if not math.isfinite(norm) or norm < 1e-6:
        return
    vx = dx / norm
    vy = dy / norm

    mask, xs, ys = _segment_distance_mask(height, width, p0, p1, max_dist_px=max_dist_px)
    if mask.size == 0:
        return

    # Mask is (len(ys), len(xs)) aligned with grid.
    y_grid, x_grid = np.nonzero(mask)
    if y_grid.size == 0:
        return
    x_pix = xs[x_grid].astype(np.int32)
    y_pix = ys[y_grid].astype(np.int32)

    ch0 = edge_index * 2 + 0
    ch1 = edge_index * 2 + 1
    paf_accum[ch0, y_pix, x_pix] += vx
    paf_accum[ch1, y_pix, x_pix] += vy
    paf_count[edge_index, y_pix, x_pix] += 1.0


def finalize_pafs(paf_accum: np.ndarray, paf_count: np.ndarray) -> np.ndarray:
    """Convert (accum,count) to averaged PAF maps shaped (8,H,W)."""
    paf = paf_accum.copy()
    for edge_idx in range(paf_count.shape[0]):
        count = paf_count[edge_idx]
        mask = count > 0
        if not np.any(mask):
            continue
        paf[edge_idx * 2 + 0][mask] /= count[mask]
        paf[edge_idx * 2 + 1][mask] /= count[mask]
    return paf


class LabelMeCornerPafDataset(Dataset):
    def __init__(
        self,
        pairs_file: str | Path,
        *,
        image_height: int = 112,
        image_width: int = 224,
        sigma: float = 7.0,
        paf_distance_px: float = 10.0,
        train: bool = True,
        rotation_degrees: float = 30.0,
        brightness: float = 0.4,
        contrast: float = 0.4,
        saturation: float = 0.4,
        hue: float = 0.1,
    ):
        self.image_height = int(image_height)
        self.image_width = int(image_width)
        self.sigma = float(sigma)
        self.paf_distance_px = float(paf_distance_px)
        self.train = bool(train)
        self.rotation_degrees = float(rotation_degrees)
        self.brightness = float(brightness)
        self.contrast = float(contrast)
        self.saturation = float(saturation)
        self.hue = float(hue)

        self.json_paths = [json_path for _img_path, json_path in load_pairs_file(pairs_file)]
        if not self.json_paths:
            raise FileNotFoundError(f"No JSON files found in pairs file: {Path(pairs_file).expanduser().resolve()}")

    def __len__(self) -> int:
        return len(self.json_paths)

    def _load_record(self, json_path: Path) -> tuple[np.ndarray, dict]:
        record = json.loads(json_path.read_text(encoding="utf-8"))
        image_path = resolve_image_path(json_path, record)
        image_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image_bgr is None:
            raise FileNotFoundError(f"Failed to read image: {image_path}")
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        return image_rgb, record

    def _extract_gates(self, record: dict) -> dict[str, dict[str, tuple[float, float]]]:
        """Return mapping gate_id -> {corner_name -> (x,y)}."""
        gates: dict[str, dict[str, tuple[float, float]]] = {}
        for shape in record.get("shapes", []):
            label_text = str(shape.get("label", ""))
            corner = normalize_corner_label(label_text)
            if corner is None:
                continue
            pts = shape.get("points", [])
            if not pts:
                continue
            x, y = pts[0]

            group_id = shape.get("group_id")
            gate_id = str(group_id) if group_id not in (None, "") else ""
            if not gate_id:
                if "_" in label_text:
                    gate_id = label_text.split("_", 1)[0]
                else:
                    gate_id = "gate_0"
            gates.setdefault(gate_id, {})[corner] = (float(x), float(y))
        return gates

    def __getitem__(self, index: int):
        json_path = self.json_paths[index]
        image_rgb, record = self._load_record(json_path)

        orig_h, orig_w = image_rgb.shape[:2]
        if orig_h != self.image_height or orig_w != self.image_width:
            raise ValueError(
                f"Expected image size {(self.image_height, self.image_width)}, "
                f"got {(orig_h, orig_w)} for {json_path}"
            )

        gates = self._extract_gates(record)

        # Optional rotation augmentation: apply same warp to image and corner points.
        if self.train:
            angle = random.uniform(-self.rotation_degrees, self.rotation_degrees)
            matrix = affine_matrix((orig_w, orig_h), angle)
            image_rgb = cv2.warpAffine(
                image_rgb,
                matrix,
                (orig_w, orig_h),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=(0, 0, 0),
            )

            for gate_id, corners in list(gates.items()):
                labels = list(corners.keys())
                coords = [corners[name] for name in labels]
                warped = warp_points(coords, matrix)
                gates[gate_id] = {name: pt for name, pt in zip(labels, warped)}

            image_rgb = apply_color_jitter_rgb(
                (image_rgb * 255.0).astype(np.uint8),
                brightness=self.brightness,
                contrast=self.contrast,
                saturation=self.saturation,
                hue=self.hue,
            )
        else:
            image_rgb = image_rgb.astype(np.float32)

        # Corner heatmaps (4,H,W): pixel-wise max across gates.
        corner_maps = np.zeros((4, self.image_height, self.image_width), dtype=np.float32)
        for corners in gates.values():
            for corner_name, (x, y) in corners.items():
                ch = CORNER_TO_INDEX[corner_name]
                draw_gaussian_heatmap(corner_maps[ch], (x, y), self.sigma)

        # PAFs (8,H,W): per-gate edge fields, aggregated by averaging where overlapping.
        paf_accum = np.zeros((len(EDGE_TYPES) * 2, self.image_height, self.image_width), dtype=np.float32)
        paf_count = np.zeros((len(EDGE_TYPES), self.image_height, self.image_width), dtype=np.float32)
        for corners in gates.values():
            if any(name not in corners for name in CORNER_NAMES):
                continue
            for edge_idx, (a, b) in enumerate(EDGE_TYPES):
                add_paf_edge(
                    paf_accum,
                    paf_count,
                    edge_index=edge_idx,
                    p0=corners[a],
                    p1=corners[b],
                    max_dist_px=self.paf_distance_px,
                )
        paf_maps = finalize_pafs(paf_accum, paf_count)

        image_rgb = np.clip(image_rgb.astype(np.float32), 0.0, 1.0)
        image_tensor = torch.from_numpy(image_rgb).permute(2, 0, 1).contiguous()
        corner_tensor = torch.from_numpy(corner_maps)
        paf_tensor = torch.from_numpy(paf_maps)
        return image_tensor, corner_tensor, paf_tensor


def collate_batch(batch):
    images, corners, pafs = zip(*batch)
    return (
        torch.stack(list(images), dim=0),
        torch.stack(list(corners), dim=0),
        torch.stack(list(pafs), dim=0),
    )


class CornerPafMSELoss(nn.Module):
    def __init__(self, *, corner_positive_weight: float = 0.0):
        super().__init__()
        self.corner_positive_weight = float(corner_positive_weight)
        self.mse = nn.MSELoss(reduction="mean")

    def forward(
        self,
        corner_preds: torch.Tensor,
        corner_targets: torch.Tensor,
        paf_preds: torch.Tensor,
        paf_targets: torch.Tensor,
    ) -> torch.Tensor:
        if self.corner_positive_weight > 0:
            weights = 1.0 + self.corner_positive_weight * corner_targets
            corner_loss = ((corner_preds - corner_targets) ** 2 * weights).sum() / weights.sum().clamp_min(1e-6)
        else:
            corner_loss = self.mse(corner_preds, corner_targets)
        paf_loss = self.mse(paf_preds, paf_targets)
        return corner_loss + paf_loss


@dataclass
class EpochResult:
    train_loss: float
    val_loss: float


@torch.no_grad()
def evaluate_split(
    model: nn.Module,
    dataset: Dataset,
    criterion: CornerPafMSELoss,
    device: torch.device,
    batch_size: int,
    num_workers: int,
    *,
    use_amp: bool,
) -> float:
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        collate_fn=collate_batch,
        drop_last=False,
    )
    return validate(model, loader, criterion, device, use_amp=use_amp)


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: CornerPafMSELoss,
    device: torch.device,
    *,
    use_amp: bool,
    scaler: torch.cuda.amp.GradScaler | None,
) -> float:
    model.train()
    total_loss = 0.0
    total_count = 0

    for images, corner_targets, paf_targets in loader:
        images = images.to(device, non_blocking=True)
        corner_targets = corner_targets.to(device, non_blocking=True)
        paf_targets = paf_targets.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, enabled=use_amp):
            logits = model(images)
            corner_preds = torch.sigmoid(logits[:, :4])
            paf_preds = torch.tanh(logits[:, 4:])
            loss = criterion(corner_preds, corner_targets, paf_preds, paf_targets)

        if use_amp and scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        batch_size = images.size(0)
        total_loss += float(loss.item()) * batch_size
        total_count += batch_size

    return total_loss / max(1, total_count)


@torch.no_grad()
def validate(
    model: nn.Module,
    loader: DataLoader,
    criterion: CornerPafMSELoss,
    device: torch.device,
    *,
    use_amp: bool,
) -> float:
    model.eval()
    total_loss = 0.0
    total_count = 0

    for images, corner_targets, paf_targets in loader:
        images = images.to(device, non_blocking=True)
        corner_targets = corner_targets.to(device, non_blocking=True)
        paf_targets = paf_targets.to(device, non_blocking=True)

        with torch.autocast(device_type=device.type, enabled=use_amp):
            logits = model(images)
            corner_preds = torch.sigmoid(logits[:, :4])
            paf_preds = torch.tanh(logits[:, 4:])
            loss = criterion(corner_preds, corner_targets, paf_preds, paf_targets)

        batch_size = images.size(0)
        total_loss += float(loss.item()) * batch_size
        total_count += batch_size

    return total_loss / max(1, total_count)


def _bilinear_sample_2ch(paf_xy: np.ndarray, x: float, y: float) -> tuple[float, float]:
    """Bilinear sample a 2-channel PAF at float coords, with edge clamping."""
    height, width = paf_xy.shape[1], paf_xy.shape[2]
    x = float(np.clip(x, 0.0, width - 1.0))
    y = float(np.clip(y, 0.0, height - 1.0))
    x0 = int(math.floor(x))
    y0 = int(math.floor(y))
    x1 = min(width - 1, x0 + 1)
    y1 = min(height - 1, y0 + 1)
    dx = x - x0
    dy = y - y0

    v00 = paf_xy[:, y0, x0]
    v10 = paf_xy[:, y0, x1]
    v01 = paf_xy[:, y1, x0]
    v11 = paf_xy[:, y1, x1]
    v0 = v00 * (1.0 - dx) + v10 * dx
    v1 = v01 * (1.0 - dx) + v11 * dx
    v = v0 * (1.0 - dy) + v1 * dy
    return float(v[0]), float(v[1])


def edge_score_line_integral(
    paf_xy: np.ndarray,
    p0: tuple[float, float],
    p1: tuple[float, float],
    *,
    samples: int = 10,
) -> float:
    x0, y0 = p0
    x1, y1 = p1
    dx = float(x1 - x0)
    dy = float(y1 - y0)
    norm = math.hypot(dx, dy)
    if not math.isfinite(norm) or norm < 1e-6:
        return float("-inf")
    ux = dx / norm
    uy = dy / norm

    samples = max(2, int(samples))
    acc = 0.0
    for i in range(samples):
        u = float(i) / float(samples - 1)
        x = float(x0 + u * dx)
        y = float(y0 + u * dy)
        vx, vy = _bilinear_sample_2ch(paf_xy, x, y)
        acc += float(vx * ux + vy * uy)
    return acc / float(samples)


def extract_corner_candidates(
    corner_heatmaps: np.ndarray,
    *,
    threshold: float,
    topk: int,
    nms_radius: int,
) -> dict[str, list[dict]]:
    """Return per-corner-type candidate points extracted from heatmaps."""
    if corner_heatmaps.ndim != 3 or corner_heatmaps.shape[0] != 4:
        raise ValueError(f"Expected corner heatmaps with shape (4,H,W), got {corner_heatmaps.shape}")
    threshold = float(threshold)
    topk = max(1, int(topk))
    nms_radius = max(0, int(nms_radius))

    hmaps_t = torch.from_numpy(corner_heatmaps).float()
    out: dict[str, list[dict]] = {name: [] for name in CORNER_NAMES}
    for idx, name in enumerate(CORNER_NAMES):
        hmap = hmaps_t[idx]
        if nms_radius > 0:
            pooled = torch.nn.functional.max_pool2d(
                hmap[None, None, ...],
                kernel_size=2 * nms_radius + 1,
                stride=1,
                padding=nms_radius,
            )[0, 0]
            mask = (hmap >= pooled) & (hmap >= threshold)
        else:
            mask = hmap >= threshold

        ys, xs = torch.nonzero(mask, as_tuple=True)
        if ys.numel() == 0:
            continue
        scores = hmap[ys, xs]
        order = torch.argsort(scores, descending=True)
        if order.numel() > topk:
            order = order[:topk]

        candidates: list[dict] = []
        for rank, k in enumerate(order.tolist()):
            candidates.append(
                {
                    "id": int(rank),  # local id per corner type (stable enough for matching)
                    "x": float(xs[k].item()),
                    "y": float(ys[k].item()),
                    "score": float(scores[k].item()),
                }
            )
        out[name] = candidates
    return out


def match_edges_greedy(score_matrix: np.ndarray, *, min_score: float) -> list[tuple[int, int, float]]:
    """Greedy bipartite matching maximizing score with one-to-one constraint."""
    min_score = float(min_score)
    if score_matrix.size == 0:
        return []
    nk, nl = score_matrix.shape
    entries: list[tuple[float, int, int]] = []
    for i in range(nk):
        for j in range(nl):
            s = float(score_matrix[i, j])
            if not np.isfinite(s) or s < min_score:
                continue
            entries.append((s, i, j))
    entries.sort(key=lambda t: t[0], reverse=True)
    used_i: set[int] = set()
    used_j: set[int] = set()
    matches: list[tuple[int, int, float]] = []
    for s, i, j in entries:
        if i in used_i or j in used_j:
            continue
        used_i.add(i)
        used_j.add(j)
        matches.append((i, j, float(s)))
    return matches


def score_and_match_edges(
    candidates: dict[str, list[dict]],
    paf_maps: np.ndarray,
    *,
    edge_min_score: float,
    integral_samples: int,
) -> dict[tuple[str, str], list[tuple[dict, dict, float]]]:
    """Compute edge scores and return matched edges per edge type."""
    if paf_maps.ndim != 3 or paf_maps.shape[0] != 8:
        raise ValueError(f"Expected paf maps with shape (8,H,W), got {paf_maps.shape}")

    edge_matches: dict[tuple[str, str], list[tuple[dict, dict, float]]] = {}
    for edge_idx, (k, l) in enumerate(EDGE_TYPES):
        left = candidates.get(k, [])
        right = candidates.get(l, [])
        if not left or not right:
            edge_matches[(k, l)] = []
            continue
        paf_xy = paf_maps[edge_idx * 2 : edge_idx * 2 + 2]
        score_matrix = np.full((len(left), len(right)), -np.inf, dtype=np.float32)
        for i, ck in enumerate(left):
            p0 = (float(ck["x"]), float(ck["y"]))
            for j, cl in enumerate(right):
                p1 = (float(cl["x"]), float(cl["y"]))
                score_matrix[i, j] = float(
                    edge_score_line_integral(paf_xy, p0, p1, samples=integral_samples)
                )
        matches = match_edges_greedy(score_matrix, min_score=edge_min_score)
        edge_matches[(k, l)] = [(left[i], right[j], s) for i, j, s in matches]
    return edge_matches


def assemble_gates_from_edges(
    edge_matches: dict[tuple[str, str], list[tuple[dict, dict, float]]]
) -> list[dict]:
    """Assemble TL/TR/BR/BL quads by finding consistent cycles in matched edges."""
    tl_tr = {(id(a), id(b)): (a, b, s) for a, b, s in edge_matches.get(("TL", "TR"), [])}
    tr_br = {(id(a), id(b)): (a, b, s) for a, b, s in edge_matches.get(("TR", "BR"), [])}
    br_bl = {(id(a), id(b)): (a, b, s) for a, b, s in edge_matches.get(("BR", "BL"), [])}
    bl_tl = {(id(a), id(b)): (a, b, s) for a, b, s in edge_matches.get(("BL", "TL"), [])}

    # Build forward maps keyed by start-candidate identity.
    tl_to_tr = {id(a): (b, s) for a, b, s in edge_matches.get(("TL", "TR"), [])}
    tr_to_br = {id(a): (b, s) for a, b, s in edge_matches.get(("TR", "BR"), [])}
    br_to_bl = {id(a): (b, s) for a, b, s in edge_matches.get(("BR", "BL"), [])}
    bl_to_tl = {id(a): (b, s) for a, b, s in edge_matches.get(("BL", "TL"), [])}

    gates: list[dict] = []
    for tl_id, (tr, s1) in tl_to_tr.items():
        tr_id = id(tr)
        if tr_id not in tr_to_br:
            continue
        br, s2 = tr_to_br[tr_id]
        br_id = id(br)
        if br_id not in br_to_bl:
            continue
        bl, s3 = br_to_bl[br_id]
        bl_id = id(bl)
        if bl_id not in bl_to_tl:
            continue
        tl2, s4 = bl_to_tl[bl_id]
        if id(tl2) != tl_id:
            continue

        # Recover the actual tl object via any stored tuple.
        tl_obj = None
        for a, b, _s in edge_matches.get(("TL", "TR"), []):
            if id(a) == tl_id:
                tl_obj = a
                break
        if tl_obj is None:
            continue

        points = {
            "TL": (float(tl_obj["x"]), float(tl_obj["y"])),
            "TR": (float(tr["x"]), float(tr["y"])),
            "BR": (float(br["x"]), float(br["y"])),
            "BL": (float(bl["x"]), float(bl["y"])),
        }
        scores = {
            "TL": float(tl_obj["score"]),
            "TR": float(tr["score"]),
            "BR": float(br["score"]),
            "BL": float(bl["score"]),
        }
        edge_scores = {"TL_TR": float(s1), "TR_BR": float(s2), "BR_BL": float(s3), "BL_TL": float(s4)}
        gate_score = float(np.mean(list(scores.values()))) + 0.5 * float(np.mean(list(edge_scores.values())))
        gates.append({"points": points, "scores": scores, "edge_scores": edge_scores, "gate_score": gate_score})

    gates.sort(key=lambda g: float(g.get("gate_score", 0.0)), reverse=True)
    return gates


def save_inference_overlay(
    output_dir: Path,
    image_stem: str,
    image_bgr: np.ndarray,
    gates: list[dict],
) -> None:
    overlays_dir = output_dir / "overlays"
    overlays_dir.mkdir(parents=True, exist_ok=True)
    overlay = image_bgr.copy()
    palette = [
        (0, 0, 255),
        (0, 165, 255),
        (0, 255, 255),
        (0, 255, 0),
        (255, 0, 0),
    ]
    for gate_idx, gate in enumerate(gates[:5]):
        color = palette[gate_idx % len(palette)]
        pts = gate.get("points", {})
        for name, (x, y) in pts.items():
            cv2.circle(overlay, (int(round(x)), int(round(y))), 4, color, -1)
            cv2.putText(
                overlay,
                name,
                (int(round(x)) + 4, int(round(y)) - 4),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )
        # Draw edges.
        for a, b in EDGE_TYPES:
            if a in pts and b in pts:
                x0, y0 = pts[a]
                x1, y1 = pts[b]
                cv2.line(
                    overlay,
                    (int(round(x0)), int(round(y0))),
                    (int(round(x1)), int(round(y1))),
                    color,
                    1,
                )
    cv2.imwrite(str(overlays_dir / f"{image_stem}.png"), overlay)


def run_inference(
    model: nn.Module,
    pairs_file: str | Path,
    output_dir: str | Path,
    expected_height: int,
    expected_width: int,
    device: torch.device,
    *,
    corner_threshold: float,
    corner_topk: int,
    corner_nms_radius: int,
    edge_min_score: float,
    integral_samples: int,
) -> None:
    pairs = load_pairs_file(pairs_file)
    output_dir = Path(output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    for image_path, _json_path in pairs:
        image_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image_bgr is None:
            continue
        if image_bgr.shape[0] != expected_height or image_bgr.shape[1] != expected_width:
            raise ValueError(
                f"Expected image size {(expected_height, expected_width)}, got {image_bgr.shape[:2]} for {image_path}"
            )
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        image_tensor = torch.from_numpy(image_rgb).permute(2, 0, 1).unsqueeze(0).contiguous().to(device)

        model.eval()
        with torch.no_grad():
            logits = model(image_tensor)
            corner_maps = torch.sigmoid(logits[:, :4])[0].detach().float().cpu().numpy()
            paf_maps = torch.tanh(logits[:, 4:])[0].detach().float().cpu().numpy()

        candidates = extract_corner_candidates(
            corner_maps,
            threshold=corner_threshold,
            topk=corner_topk,
            nms_radius=corner_nms_radius,
        )
        edge_matches = score_and_match_edges(
            candidates,
            paf_maps,
            edge_min_score=edge_min_score,
            integral_samples=integral_samples,
        )
        gates = assemble_gates_from_edges(edge_matches)

        save_inference_overlay(output_dir, image_path.stem, image_bgr, gates)

    print(f"Inference overlays saved to: {output_dir}")


def build_args():
    parser = ArgumentParser(description="Train corner+PAF gate detector from LabelMe data.")
    script_dir = Path(__file__).resolve().parent
    parser.add_argument("--split_dir", type=str, default=None, help="Folder containing train.txt, val.txt, and test.txt.")
    parser.add_argument("--mode", type=str, choices=["train", "test", "inference"], default="train")
    parser.add_argument("--output_dir", type=str, default=str(script_dir / "corner_affinity_checkpoints"))
    parser.add_argument("--inference_output_dir", type=str, default=str(script_dir / "corner_affinity_inference"))
    parser.add_argument("--checkpoint", type=str, default=str(script_dir / "corner_affinity_checkpoints" / "best.pt"))
    parser.add_argument("--image_height", type=int, default=112)
    parser.add_argument("--image_width", type=int, default=224)
    parser.add_argument("--sigma", type=float, default=7.0)
    parser.add_argument("--paf_distance_px", type=float, default=10.0)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--rotation_degrees", type=float, default=30.0)
    parser.add_argument("--brightness", type=float, default=0.4)
    parser.add_argument("--contrast", type=float, default=0.4)
    parser.add_argument("--saturation", type=float, default=0.4)
    parser.add_argument("--hue", type=float, default=0.1)
    parser.add_argument(
        "--corner_positive_weight",
        type=float,
        default=0.0,
        help="Optional extra weight applied to nonzero corner heatmap regions (paper uses 0.0).",
    )
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--resume", type=str, default="")

    # Inference hyperparameters.
    parser.add_argument("--corner_threshold", type=float, default=0.6)
    parser.add_argument("--corner_topk", type=int, default=50)
    parser.add_argument("--corner_nms_radius", type=int, default=5)
    parser.add_argument("--edge_min_score", type=float, default=0.05)
    parser.add_argument("--integral_samples", type=int, default=10)

    return parser.parse_args()


def main():
    args = build_args()
    set_seed(args.seed)

    device = torch.device(args.device)
    model = CornerUNet(out_channels=12).to(device)

    split_dir = Path(args.split_dir).expanduser().resolve() if args.split_dir else DEFAULT_SPLIT_DIR
    split_dir = split_dir.expanduser().resolve()

    if args.mode == "train":
        output_dir = Path(args.output_dir).expanduser().resolve()
        output_dir.mkdir(parents=True, exist_ok=True)

        train_split = resolve_split_path(split_dir, "train")
        val_split = resolve_split_path(split_dir, "val")
        train_dataset = LabelMeCornerPafDataset(
            train_split,
            image_height=args.image_height,
            image_width=args.image_width,
            sigma=args.sigma,
            paf_distance_px=args.paf_distance_px,
            train=True,
            rotation_degrees=args.rotation_degrees,
            brightness=args.brightness,
            contrast=args.contrast,
            saturation=args.saturation,
            hue=args.hue,
        )
        val_dataset = LabelMeCornerPafDataset(
            val_split,
            image_height=args.image_height,
            image_width=args.image_width,
            sigma=args.sigma,
            paf_distance_px=args.paf_distance_px,
            train=False,
            rotation_degrees=args.rotation_degrees,
            brightness=args.brightness,
            contrast=args.contrast,
            saturation=args.saturation,
            hue=args.hue,
        )

        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=torch.cuda.is_available(),
            collate_fn=collate_batch,
            drop_last=False,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=torch.cuda.is_available(),
            collate_fn=collate_batch,
            drop_last=False,
        )

        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        criterion = CornerPafMSELoss(corner_positive_weight=args.corner_positive_weight)
        use_amp = device.type == "cuda"
        scaler = torch.cuda.amp.GradScaler(enabled=use_amp) if use_amp else None

        start_epoch = 0
        best_val_loss = float("inf")
        if args.resume:
            resume_path = Path(args.resume).expanduser().resolve()
            checkpoint = torch.load(resume_path, map_location=device)
            model.load_state_dict(checkpoint["model"])
            optimizer.load_state_dict(checkpoint["optimizer"])
            if use_amp and scaler is not None and checkpoint.get("scaler") is not None:
                scaler.load_state_dict(checkpoint["scaler"])
            start_epoch = int(checkpoint.get("epoch", 0)) + 1
            best_val_loss = float(checkpoint.get("best_val_loss", best_val_loss))

        for epoch in range(start_epoch, args.epochs):
            train_loss = train_one_epoch(
                model,
                train_loader,
                optimizer,
                criterion,
                device,
                use_amp=use_amp,
                scaler=scaler,
            )
            val_loss = validate(model, val_loader, criterion, device, use_amp=use_amp)

            checkpoint = {
                "epoch": epoch,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scaler": scaler.state_dict() if scaler is not None else None,
                "best_val_loss": best_val_loss,
                "args": vars(args),
            }
            torch.save(checkpoint, output_dir / "last.pt")
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                checkpoint["best_val_loss"] = best_val_loss
                torch.save(checkpoint, output_dir / "best.pt")

            print(
                f"epoch={epoch + 1}/{args.epochs} "
                f"train_loss={train_loss:.5f} val_loss={val_loss:.5f} best_val_loss={best_val_loss:.5f}"
            )

    elif args.mode == "test":
        checkpoint_path = Path(args.checkpoint).expanduser().resolve()
        checkpoint = torch.load(checkpoint_path, map_location=device)
        state_dict = checkpoint.get("model", checkpoint)
        model.load_state_dict(state_dict)
        criterion = CornerPafMSELoss(corner_positive_weight=args.corner_positive_weight)
        test_split = resolve_split_path(split_dir, "test")
        test_dataset = LabelMeCornerPafDataset(
            test_split,
            image_height=args.image_height,
            image_width=args.image_width,
            sigma=args.sigma,
            paf_distance_px=args.paf_distance_px,
            train=False,
        )
        test_loss = evaluate_split(
            model=model,
            dataset=test_dataset,
            criterion=criterion,
            device=device,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            use_amp=(device.type == "cuda"),
        )
        print(f"test_loss={test_loss:.5f}")

    else:
        checkpoint_path = Path(args.checkpoint).expanduser().resolve()
        checkpoint = torch.load(checkpoint_path, map_location=device)
        state_dict = checkpoint.get("model", checkpoint)
        model.load_state_dict(state_dict)
        test_split = resolve_split_path(split_dir, "test")
        run_inference(
            model=model,
            pairs_file=test_split,
            output_dir=args.inference_output_dir,
            expected_height=args.image_height,
            expected_width=args.image_width,
            device=device,
            corner_threshold=args.corner_threshold,
            corner_topk=args.corner_topk,
            corner_nms_radius=args.corner_nms_radius,
            edge_min_score=args.edge_min_score,
            integral_samples=args.integral_samples,
        )


if __name__ == "__main__":
    main()

