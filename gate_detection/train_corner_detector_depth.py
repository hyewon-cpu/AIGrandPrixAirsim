#!/usr/bin/env python3

"""Train a gate-corner detector from LabelMe annotations (depth input).

The model predicts four heatmaps, one for each gate corner class:
TL, TR, BL, BR.

This is a heatmap-regression setup inspired by the pipeline described in the
paper text you shared: each annotated corner becomes a Gaussian blob in the
target heatmap, and the network is trained with supervised regression.
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

from corner_unet_depth import CornerUNetDepth


CORNER_NAMES = ("TL", "TR", "BL", "BR")
CORNER_TO_INDEX = {name: idx for idx, name in enumerate(CORNER_NAMES)}
DEFAULT_SPLIT_DIR_DEPTH = Path(__file__).resolve().parent / "dataforuse_depth" #resolve() : turns to absolute path 
DEFAULT_CHECKPOINT_DIR_DEPTH = Path(__file__).resolve().parent / "corner_checkpoint_depth"
DEPTH_MIN_M = 0.01
DEPTH_MAX_M = 200.0


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

    # Fall back to the path stored in the JSON if it is already absolute.
    candidate = Path(image_path).expanduser().resolve()
    if candidate.exists():
        return candidate

    raise FileNotFoundError(f"Could not resolve image for {json_path}: {image_path}")


def load_pairs_file(pairs_file: str | Path) -> list[tuple[Path, Path]]: #read the pairs file and make it in a list of tuples, wchich contains the paths.  
    pairs_path = Path(pairs_file).expanduser().resolve()
    if not pairs_path.exists():
        raise FileNotFoundError(f"pairs file does not exist: {pairs_path}")

    pairs: list[tuple[Path, Path]] = []
    for raw_line in pairs_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip() #remove leading and trailing whitespace characters from the line
        if not line or line.startswith("#"): #skip empty or comment lines 
            continue
        parts = line.split() #split the line inot witespace separated pieces(any whitespace)
        image_candidates = [
            part
            for part in parts
            if part.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".webp", ".pfm"))
        ]
        json_candidates = [part for part in parts if part.lower().endswith(".json")]
        if not image_candidates:
            raise ValueError(f"Could not find image path in pairs line: {raw_line}")
        if not json_candidates:
            raise ValueError(f"Could not find JSON path in pairs line: {raw_line}")
        pairs.append(
            (
                Path(image_candidates[0]).expanduser().resolve(), #take the first image file found
                Path(json_candidates[-1]).expanduser().resolve(), #take the last JSOn file found 
            )
        )
    return pairs


def resolve_split_path(split_dir: str | Path, split_name: str) -> Path: #str | Path : accepts either string or path 
    split_path = Path(split_dir).expanduser().resolve() / f"{split_name}.txt"
    if not split_path.exists():
        raise FileNotFoundError(f"Split file does not exist: {split_path}")
    return split_path


def resolve_checkpoint_path(checkpoint: str | Path) -> Path:
    checkpoint_path = Path(checkpoint).expanduser().resolve()
    if checkpoint_path.exists():
        return checkpoint_path

    script_dir = Path(__file__).resolve().parent
    fallback_paths = [
        script_dir / "gate_detection" / "corner_checkpoints" / checkpoint_path.name,
        script_dir / "gate_detection" / "corner_checkpoint_depth" / checkpoint_path.name,
        script_dir / "corner_checkpoints" / checkpoint_path.name,
        script_dir / "corner_checkpoint_depth" / checkpoint_path.name,
    ]
    for candidate in fallback_paths:
        if candidate.exists():
            return candidate.resolve()

    raise FileNotFoundError(f"Checkpoint does not exist: {checkpoint_path}")


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


def read_pfm(path: Path) -> np.ndarray:
    """Read a single-channel PFM file written by `annotate_gate_corners_depth.py`."""
    with path.open("rb") as fh:
        header = fh.readline().decode("ascii", errors="replace").strip()
        if header != "Pf":
            raise ValueError(f"Unsupported PFM header {header!r} in {path}")

        dims = fh.readline().decode("ascii", errors="replace").strip().split()
        if len(dims) != 2:
            raise ValueError(f"Malformed PFM dimensions line in {path}: {dims!r}")
        width, height = map(int, dims)

        scale = float(fh.readline().decode("ascii", errors="replace").strip())
        dtype = "<f4" if scale < 0 else ">f4"

        data = np.frombuffer(fh.read(), dtype=dtype)
        expected = width * height
        if data.size != expected:
            raise ValueError(f"PFM data size mismatch in {path}: expected {expected}, got {data.size}")

        image = data.reshape((height, width))
        return image.astype(np.float32, copy=False)


def normalize_depth_to_01(
    depth_m: np.ndarray,
    *,
    min_depth_m: float = DEPTH_MIN_M,
    max_depth_m: float = DEPTH_MAX_M,
) -> np.ndarray:
    depth = np.asarray(depth_m, dtype=np.float32)
    depth = np.nan_to_num(
        depth,
        nan=float(max_depth_m),
        posinf=float(max_depth_m),
        neginf=float(min_depth_m),
    )
    depth = np.clip(depth, float(min_depth_m), float(max_depth_m))
    return depth / float(max_depth_m)


def apply_color_jitter_rgb(
    image: np.ndarray,
    brightness: float,
    contrast: float,
    saturation: float,
    hue: float,
) -> np.ndarray:
    """Apply simple color jitter in RGB space."""
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


def draw_gaussian_heatmap(
    heatmap: np.ndarray,
    point: tuple[float, float],
    sigma: float,
) -> None:
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

    patch_w = right - left
    patch_h = bottom - top
    grid_x = np.arange(left, right, dtype=np.float32)
    grid_y = np.arange(top, bottom, dtype=np.float32)
    xx, yy = np.meshgrid(grid_x, grid_y)
    gauss = np.exp(-((xx - x) ** 2 + (yy - y) ** 2) / (2.0 * sigma * sigma))
    heatmap[top:bottom, left:right] = np.maximum(heatmap[top:bottom, left:right], gauss)


class WeightedHeatmapMSELoss(nn.Module):
    """MSE that emphasizes positive heatmap regions.

    Plain MSE is dominated by the background because most pixels are zero.
    This loss keeps the same regression objective but upweights pixels near
    the Gaussian peaks so corner locations matter much more.
    """

    def __init__(self, positive_weight: float = 2000.0):
        super().__init__()
        self.positive_weight = float(positive_weight)

    def forward(self, preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        if preds.shape != targets.shape:
            raise ValueError(f"preds and targets must have the same shape, got {tuple(preds.shape)} vs {tuple(targets.shape)}")
        weights = 1.0 + self.positive_weight * targets
        squared_error = (preds - targets) ** 2
        return (weights * squared_error).sum() / weights.sum().clamp_min(1e-6)


class LabelMeCornerDataset(Dataset):
    def __init__(
        self,
        pairs_file: str | Path | None = None, #dataset split file path
        image_height: int = 112,
        image_width: int = 224,
        sigma: float = 7.0,
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
        self.train = bool(train)
        self.rotation_degrees = float(rotation_degrees)
        self.brightness = float(brightness)
        self.contrast = float(contrast)
        self.saturation = float(saturation)
        self.hue = float(hue)

    

        if pairs_file is None:
            raise ValueError("pairs_file is required.")
        self.json_paths = [json_path for _, json_path in load_pairs_file(pairs_file)] #throw away the png paths and keep the json paths
    
        if not self.json_paths:
            if pairs_file is not None:
                raise FileNotFoundError(f"No JSON files found in pairs file: {Path(pairs_file).expanduser().resolve()}")
            raise FileNotFoundError("No pairs file provided.")

    def __len__(self) -> int:
        return len(self.json_paths)

    def _load_record(self, json_path: Path) -> tuple[np.ndarray, dict]:
        record = json.loads(json_path.read_text(encoding="utf-8"))
        image_path = resolve_image_path(json_path, record)
        if image_path.suffix.lower() == ".pfm":
            depth_m = read_pfm(image_path)
            depth01 = normalize_depth_to_01(depth_m)
            return depth01.astype(np.float32, copy=False), record

        image = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
        if image is None:
            raise FileNotFoundError(f"Failed to read image: {image_path}")
        if image.ndim == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        if image.dtype == np.uint8:
            depth01 = image.astype(np.float32) / 255.0
        elif image.dtype == np.uint16:
            depth01 = image.astype(np.float32) / 65535.0
        else:
            depth01 = image.astype(np.float32)
            if depth01.max() > 1.0:
                depth01 = depth01 / max(1.0, float(depth01.max()))
        depth01 = np.clip(depth01, 0.0, 1.0)
        return depth01, record

    def _extract_points(self, record: dict) -> list[tuple[str, tuple[float, float]]]:
        points: list[tuple[str, tuple[float, float]]] = []
        for shape in record.get("shapes", []):
            label = normalize_corner_label(shape.get("label", ""))
            if label is None:
                continue
            shape_points = shape.get("points", [])
            if not shape_points:
                continue
            x, y = shape_points[0]
            points.append((label, (float(x), float(y))))
        return points

    def __getitem__(self, index: int):
        json_path = self.json_paths[index]
        depth01, record = self._load_record(json_path)
        points = self._extract_points(record)

        orig_h, orig_w = depth01.shape[:2]
        if orig_h != self.image_height or orig_w != self.image_width:
            raise ValueError(
                f"Expected image size {(self.image_height, self.image_width)}, "
                f"got {(orig_h, orig_w)} for {json_path}"
            )

        if self.train:
            angle = random.uniform(-self.rotation_degrees, self.rotation_degrees)
            matrix = affine_matrix((orig_w, orig_h), angle)
            depth01 = cv2.warpAffine(
                depth01,
                matrix,
                (orig_w, orig_h),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=float(DEPTH_MIN_M / DEPTH_MAX_M),
            )
            labels = [label for label, _ in points]
            coords = [pt for _, pt in points]
            points = list(zip(labels, warp_points(coords, matrix)))
        else:
            depth01 = depth01.astype(np.float32)

        heatmaps = np.zeros((4, self.image_height, self.image_width), dtype=np.float32)
        for label, (x, y) in points:
            ch = CORNER_TO_INDEX[label]
            draw_gaussian_heatmap(heatmaps[ch], (x, y), self.sigma)

        depth01 = depth01.astype(np.float32)
        depth01 = np.clip(depth01, float(DEPTH_MIN_M / DEPTH_MAX_M), 1.0)

        image_tensor = torch.from_numpy(depth01).unsqueeze(0).contiguous()
        heatmap_tensor = torch.from_numpy(heatmaps)
        return image_tensor, heatmap_tensor


@dataclass
class EpochResult:
    train_loss: float
    val_loss: float


def build_dataset_from_split( #wrapper around LabelMeCornerDataset to build labeled dataset 
    split_path: str | Path,
    image_height: int,
    image_width: int,
    sigma: float,
    train: bool,
    rotation_degrees: float,
    brightness: float,
    contrast: float,
    saturation: float,
    hue: float,
) -> LabelMeCornerDataset: #return type annotation. This function returns a LabelMeCornerDataset object that is initialized with the given parameters.
    return LabelMeCornerDataset( #makes the labeled heat map and input image data(augmentation if needed)
        pairs_file=split_path,
        image_height=image_height,
        image_width=image_width,
        sigma=sigma, #width of Gaussian blob 
        train=train, #whether to apply data augmentation
        rotation_degrees=rotation_degrees, #max rotation angle for image
        brightness=brightness, 
        contrast=contrast,
        saturation=saturation,
        hue=hue,
    )


def collate_batch(batch):
    images, heatmaps = zip(*batch)
    return torch.stack(list(images), dim=0), torch.stack(list(heatmaps), dim=0)


def load_image_for_inference(
    image_path: Path, expected_height: int, expected_width: int
) -> tuple[np.ndarray, torch.Tensor, bool]:
    if image_path.suffix.lower() == ".pfm":
        depth_m = read_pfm(image_path)
        depth01 = normalize_depth_to_01(depth_m)
        depth01_vis = np.flipud(depth01)
        depth_uint8 = np.clip(depth01_vis * 255.0, 0.0, 255.0).astype(np.uint8)
        image_bgr = cv2.applyColorMap(depth_uint8, cv2.COLORMAP_JET)
        orig_h, orig_w = depth01.shape[:2]
        depth_tensor = torch.from_numpy(depth01.astype(np.float32)).unsqueeze(0).contiguous()
        flip_overlay_y = True
    else:
        image = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
        if image is None:
            raise FileNotFoundError(f"Failed to read image: {image_path}")
        if image.ndim == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        if image.dtype == np.uint8:
            depth01 = image.astype(np.float32) / 255.0
        elif image.dtype == np.uint16:
            depth01 = image.astype(np.float32) / 65535.0
        else:
            depth01 = image.astype(np.float32)
            if depth01.max() > 1.0:
                depth01 = depth01 / max(1.0, float(depth01.max()))
        depth01 = np.clip(depth01, 0.0, 1.0)
        image_bgr = cv2.cvtColor((depth01 * 255.0).astype(np.uint8), cv2.COLOR_GRAY2BGR)
        orig_h, orig_w = depth01.shape[:2]
        depth_tensor = torch.from_numpy(depth01.astype(np.float32)).unsqueeze(0).contiguous()
        flip_overlay_y = False
    if orig_h != expected_height or orig_w != expected_width:
        raise ValueError(
            f"Expected image size {(expected_height, expected_width)}, "
            f"got {(orig_h, orig_w)} for {image_path}"
        )
    return image_bgr, depth_tensor, flip_overlay_y


def extract_corner_predictions(
    heatmaps: torch.Tensor,
    *,
    threshold: float = 0.25,
    topk: int = 50,
    nms_radius: int = 5,
) -> list[dict]:
    """Extract multiple corner candidates per channel from heatmaps.

    The model predicts dense heatmaps; decoding with argmax collapses each channel
    to a single point. For scenes with multiple gates/corners visible, we want
    multiple local maxima above a confidence threshold.
    """
    if heatmaps.ndim != 3 or heatmaps.shape[0] != 4:
        raise ValueError(f"Expected heatmaps with shape (4, H, W), got {tuple(heatmaps.shape)}")
    if topk <= 0:
        return []

    threshold = float(threshold)
    topk = int(topk)
    nms_radius = int(nms_radius)
    if nms_radius < 0:
        nms_radius = 0

    preds: list[dict] = []
    heatmaps_t = heatmaps.detach().float().cpu()
    for idx, corner_name in enumerate(CORNER_NAMES):
        hmap = heatmaps_t[idx]
        if nms_radius > 0:
            # Local maxima via max-pooling (NMS). A pixel is a peak if it equals the
            # max in its neighborhood and exceeds the confidence threshold.
            pooled = torch.nn.functional.max_pool2d(
                hmap[None, None, ...],
                kernel_size=2 * nms_radius + 1,
                stride=1,
                padding=nms_radius,
            )[0, 0]
            peak_mask = (hmap >= pooled) & (hmap >= threshold)
        else:
            peak_mask = hmap >= threshold

        ys, xs = torch.nonzero(peak_mask, as_tuple=True)
        if ys.numel() == 0:
            continue

        scores = hmap[ys, xs]
        order = torch.argsort(scores, descending=True)
        if order.numel() > topk:
            order = order[:topk]

        for rank, k in enumerate(order.tolist()):
            y = int(ys[k].item())
            x = int(xs[k].item())
            score = float(scores[k].item())
            preds.append(
                {
                    "label": corner_name,
                    "x": float(x),
                    "y": float(y),
                    "score": score,
                    "rank": int(rank),
                }
            )

    # Sort across channels so the JSON is stable and the strongest points come first.
    preds.sort(key=lambda item: float(item.get("score", 0.0)), reverse=True)
    return preds


def save_prediction_record(
    output_dir: Path,
    image_stem: str,
    source_image_path: Path,
    image_bgr: np.ndarray,
    predicted_corners: list[dict],
    *,
    debug: dict | None = None,
    flip_overlay_y: bool = False,
) -> None:
    predictions_dir = output_dir / "predictions"
    overlays_dir = output_dir / "overlays"
    predictions_dir.mkdir(parents=True, exist_ok=True)
    overlays_dir.mkdir(parents=True, exist_ok=True)

    orig_h, orig_w = image_bgr.shape[:2]
    shapes: list[dict] = []
    overlay = image_bgr.copy()
    for corner in predicted_corners:
        x = float(corner["x"])
        y = float(corner["y"])
        score = float(corner["score"])
        label = str(corner["label"])
        y_draw = float(orig_h - 1) - y if flip_overlay_y else y
        shapes.append(
            {
                "label": label,
                "points": [[x, y]],
                "group_id": "predicted_gate_0",
                "shape_type": "point",
                "flags": {},
                "score": score,
            }
        )
        cv2.circle(overlay, (int(round(x)), int(round(y_draw))), 4, (0, 0, 255), -1)
        cv2.putText(
            overlay,
            f"{label}:{score:.2f}",
            (int(round(x)) + 4, int(round(y_draw)) - 4),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (0, 255, 0),
            1,
            cv2.LINE_AA,
        )

    record = {
        "version": "5.4.1",
        "flags": {},
        "shapes": shapes,
        "imagePath": str(source_image_path),
        "imageData": None,
        "imageHeight": int(orig_h),
        "imageWidth": int(orig_w),
    }
    if debug is not None:
        record["debug"] = debug

    json_path = predictions_dir / f"{image_stem}.json"
    json_path.write_text(json.dumps(record, indent=2), encoding="utf-8")
    cv2.imwrite(str(overlays_dir / f"{image_stem}.png"), overlay)


def run_inference(
    model: nn.Module,
    pairs_file: str | Path,
    output_dir: str | Path,
    expected_height: int,
    expected_width: int,
    device: torch.device,
    *,
    threshold: float = 0.25,
    topk: int = 50,
    nms_radius: int = 5,
) -> None:
    pairs = load_pairs_file(pairs_file)
    output_dir = Path(output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    pairs_out_path = output_dir / "pairs.txt"
    with pairs_out_path.open("w", encoding="utf-8") as pairs_out:
        model.eval()
        for image_path, json_path in pairs:
            image_bgr, image_tensor, flip_overlay_y = load_image_for_inference(
                image_path, expected_height, expected_width
            )
            image_tensor = image_tensor.unsqueeze(0).to(device)
            with torch.no_grad():
                logits = model(image_tensor)
                heatmaps = torch.sigmoid(logits)[0]
            heatmap_max = [float(heatmaps[i].max().item()) for i in range(int(heatmaps.shape[0]))]
            predicted_corners = extract_corner_predictions(
                heatmaps,
                threshold=threshold,
                topk=topk,
                nms_radius=nms_radius,
            )

            image_stem = image_path.stem
            save_prediction_record(
                output_dir=output_dir,
                image_stem=image_stem,
                source_image_path=image_path,
                image_bgr=image_bgr,
                predicted_corners=predicted_corners,
                debug={
                    "heatmap_max": heatmap_max,
                    "threshold": float(threshold),
                    "topk": int(topk),
                    "nms_radius": int(nms_radius),
                    "num_predictions": int(len(predicted_corners)),
                },
                flip_overlay_y=flip_overlay_y,
            )
            pred_json_path = output_dir / "predictions" / f"{image_stem}.json"
            pairs_out.write(f"{image_path} {pred_json_path}\n")

    print(f"Inference results saved to: {output_dir}")


@torch.no_grad()
def evaluate_split(
    model: nn.Module,
    dataset: Dataset,
    criterion: nn.Module,
    device: torch.device,
    batch_size: int,
    num_workers: int,
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
    return validate(model, loader, criterion, device)


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    use_amp: bool,
    scaler: torch.cuda.amp.GradScaler | None,
) -> float:
    model.train()
    total_loss = 0.0
    total_count = 0

    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, enabled=use_amp):
            logits = model(images)
            preds = torch.sigmoid(logits)
            loss = criterion(preds, targets)

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
    criterion: nn.Module,
    device: torch.device,
) -> float:
    model.eval()
    total_loss = 0.0
    total_count = 0
    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        logits = model(images)
        preds = torch.sigmoid(logits)
        loss = criterion(preds, targets)
        batch_size = images.size(0)
        total_loss += float(loss.item()) * batch_size
        total_count += batch_size
    return total_loss / max(1, total_count)


def build_args():
    parser = ArgumentParser(description="Train a corner heatmap detector from LabelMe data.")
    script_dir = Path(__file__).resolve().parent
    parser.add_argument(
        "--split_dir",
        type=str,
        default=None,
        help="Folder containing train.txt, val.txt, and test.txt.",
    )
    parser.add_argument("--mode", type=str, choices=["train", "test", "inference"], default="train")
    parser.add_argument("--output_dir", type=str, default=str(script_dir / "corner_checkpoint_depth"))
    parser.add_argument("--inference_output_dir", type=str, default=str(script_dir / "corner_inference_depth"))
    parser.add_argument("--checkpoint", type=str, default=str(script_dir / "corner_checkpoint_depth" / "best.pt"))
    parser.add_argument("--image_height", type=int, default=112)
    parser.add_argument("--image_width", type=int, default=224)
    parser.add_argument("--sigma", type=float, default=7.0)
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
        "--positive_weight",
        type=float,
        default=2000.0,
        help="Extra weight applied to nonzero heatmap regions in the loss.",
    )
    parser.add_argument(
        "--inference_threshold",
        type=float,
        default=0.25,
        help="Confidence threshold for inference peak extraction.",
    )
    parser.add_argument(
        "--inference_topk",
        type=int,
        default=50,
        help="Max number of peaks to keep per corner channel during inference.",
    )
    parser.add_argument(
        "--inference_nms_radius",
        type=int,
        default=5,
        help="Neighborhood radius (in pixels) for local-maximum filtering during inference.",
    )
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--resume", type=str, default="")
    return parser.parse_args()


def main():
    args = build_args()
    set_seed(args.seed)

    device = torch.device(args.device)
    model = CornerUNetDepth().to(device)
    default_split_dir = DEFAULT_SPLIT_DIR_DEPTH
    split_dir = (Path(args.split_dir).expanduser().resolve() if args.split_dir else default_split_dir) #expanduser : expand to your home direcotory if path starts with ~. resolve() : resolve to absolute path.
    if args.mode == "train":
        output_dir = Path(args.output_dir).expanduser().resolve()
        output_dir.mkdir(parents=True, exist_ok=True)
        #parents=True : makes parent directories if not exist 
        #exist_ok=True : does not raise error if the directory already exists

        train_split = resolve_split_path(split_dir, "train") #train.txt file path 
        val_split = resolve_split_path(split_dir, "val") #val.txt file path
        train_dataset = build_dataset_from_split( #build annotated map and input image data for training. image, heatmap pairs 
            train_split,
            image_height=args.image_height,
            image_width=args.image_width,
            sigma=args.sigma,
            train=True,
            rotation_degrees=args.rotation_degrees,
            brightness=args.brightness,
            contrast=args.contrast,
            saturation=args.saturation,
            hue=args.hue,
        )
        val_dataset = build_dataset_from_split( #build annotated map and input image data for validation (no augmentation) image, heatmap pairs 
            val_split,
            image_height=args.image_height,
            image_width=args.image_width,
            sigma=args.sigma,
            train=False, #no augmentation for validation
            rotation_degrees=args.rotation_degrees,
            brightness=args.brightness,
            contrast=args.contrast,
            saturation=args.saturation,
            hue=args.hue,
        )

        train_loader = DataLoader( #load data for training in batches
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True, #randomize sample order for each epoch
            num_workers=args.num_workers,
            pin_memory=torch.cuda.is_available(),
            collate_fn=collate_batch, #stacks images and heatmaps into tensors for each batch
            drop_last=False,  #keep the last smaller batch instead of discarding it 
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
        criterion = WeightedHeatmapMSELoss(positive_weight=args.positive_weight) #loss function that emphasizes nonzero heatmap regions
        #positive weight : more weight to nonzero heatmap regions. 
        use_amp = device.type == "cuda" #check if using GPU for training, if so, use automatic mixed precision for faster training and lower memory usage.
            #True/False output
        scaler = torch.cuda.amp.GradScaler(enabled=use_amp) if use_amp else None #creates an AMP gradient scalar if using GPU, otherwise None
            #AMP gradient scalar : helps with scaling loss values to prevent underflow when using mixed precision.
            #AMP = automatic mixed precision, which allows using float16 precision where possible for faster computation and lower memory usage, while keeping float32 precision where needed for stability. The GradScaler helps to scale the loss values to prevent underflow when using float16.

        start_epoch = 0
        best_val_loss = float("inf")

        if args.resume: #resume training from a checkpoint 
            resume_path = Path(args.resume).expanduser().resolve()
            checkpoint = torch.load(resume_path, map_location=device)
            model.load_state_dict(checkpoint["model"])
            optimizer.load_state_dict(checkpoint["optimizer"])
            if use_amp and scaler is not None and "scaler" in checkpoint and checkpoint["scaler"] is not None:
                scaler.load_state_dict(checkpoint["scaler"])
            start_epoch = int(checkpoint.get("epoch", 0)) + 1
            best_val_loss = float(checkpoint.get("best_val_loss", best_val_loss))

        for epoch in range(start_epoch, args.epochs):
            train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device, use_amp, scaler) #train for one epoch and get the average training loss for that epoch 
            val_loss = validate(model, val_loader, criterion, device) #value loss for that epoch

            checkpoint = {
                "epoch": epoch,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scaler": scaler.state_dict() if scaler is not None else None,
                "best_val_loss": best_val_loss,
                "args": vars(args), #makes it into a dictionary
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
        checkpoint_path = resolve_checkpoint_path(args.checkpoint)
        checkpoint = torch.load(checkpoint_path, map_location=device)
        state_dict = checkpoint.get("model", checkpoint)
        model.load_state_dict(state_dict)

        criterion = WeightedHeatmapMSELoss(positive_weight=args.positive_weight)
        test_split = resolve_split_path(split_dir, "test")
        test_dataset = build_dataset_from_split(
            test_split,
            image_height=args.image_height,
            image_width=args.image_width,
            sigma=args.sigma,
            train=False,
            rotation_degrees=args.rotation_degrees,
            brightness=args.brightness,
            contrast=args.contrast,
            saturation=args.saturation,
            hue=args.hue,
        )
        test_loss = evaluate_split(
            model=model,
            dataset=test_dataset,
            criterion=criterion,
            device=device,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
        )
        print(f"test_loss={test_loss:.5f}")
    else:
        checkpoint_path = resolve_checkpoint_path(args.checkpoint)
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
            threshold=args.inference_threshold,
            topk=args.inference_topk,
            nms_radius=args.inference_nms_radius,
        )


if __name__ == "__main__":
    main()
