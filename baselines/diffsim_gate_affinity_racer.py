from __future__ import annotations

"""DiffSim racer that uses a corner+PAF U-Net to pick the next gate target.

This is a variant of `diffsim_gate_detection_racer.py`, but uses the paper-style
gate detector outputs:

- 4 corner confidence maps (TL, TR, BL, BR)
- 4 Part Affinity Fields (PAFs) for edges:
  (TL->TR), (TR->BR), (BR->BL), (BL->TL) => 8 channels (vx, vy per edge)

Gate targets are computed from the detected gate center (from corner points) and
back-projected into 3D using the depth estimate at that pixel.

Post-processing (corner candidates -> edge scoring -> matching -> gate assembly)
reuses the same helper functions from `gate_detection/train_corner_affinity_detection.py`.
"""

from argparse import ArgumentParser, BooleanOptionalAction
from pathlib import Path
import math
import sys
import time

import airsimdroneracinglab as airsim
import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
GATE_DETECTION_DIR = REPO_ROOT / "gate_detection"

try:
    import cv2
except ImportError:  # pragma: no cover
    cv2 = None

DEFAULT_AFFINITY_CHECKPOINT = GATE_DETECTION_DIR / "corner_affinity_checkpoints" / "best.pt"
DEFAULT_CORNER_CONF_THRESHOLD = 0.25
DEFAULT_CORNER_TOPK = 50
DEFAULT_CORNER_NMS_RADIUS = 5
DEFAULT_EDGE_MIN_SCORE = 0.05
DEFAULT_INTEGRAL_SAMPLES = 10

DEFAULT_DEPTH_ONNX_PATH = (
    REPO_ROOT
    / "depth_estimation"
    / "results"
    / "run_20260422_135255"
    / "export"
    / "dn_model_latest.onnx"
)
DEFAULT_DEPTH_INPUT_WIDTH = 252
DEFAULT_DEPTH_INPUT_HEIGHT = 140
DEFAULT_DEPTH_DEVICE = "auto"
DEFAULT_GATE_MAX_DEPTH_M = 50.0 #max depth for gate selection

if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
if str(GATE_DETECTION_DIR) not in sys.path:
    sys.path.insert(0, str(GATE_DETECTION_DIR))

try:
    import torch
except ImportError as exc:  # pragma: no cover
    torch = None
    TORCH_IMPORT_ERROR = exc
else:
    TORCH_IMPORT_ERROR = None

from diffsim_depth_estimate_racer import DepthAnythingOnnxEstimator
from diffsim_racer import (
    DiffSimRacer,
    DEFAULT_MODEL_PATH,
    airsim_to_flightmare_rotation,
    airsim_to_flightmare_vector,
    compute_pinhole_intrinsics,
    flightmare_to_airsim_vector,
    normalize,
    quaternion_to_rotation_matrix,
)

from train_corner_affinity_detection import (  # noqa: E402
    CORNER_NAMES,
    EDGE_TYPES,
    assemble_gates_from_edges,
    extract_corner_candidates,
    score_and_match_edges,
)


class CornerAffinityEstimator:
    def __init__(self, checkpoint_path=DEFAULT_AFFINITY_CHECKPOINT, device="cpu"):
        if torch is None:
            raise ImportError(
                "torch is required to run the corner+PAF detector. Install PyTorch first."
            ) from TORCH_IMPORT_ERROR

        from corner_unet import CornerUNet

        checkpoint_path = Path(checkpoint_path).expanduser().resolve()
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Corner affinity checkpoint does not exist: {checkpoint_path}")

        self.checkpoint_path = checkpoint_path
        self.device = torch.device(device)
        self.model = CornerUNet(out_channels=12).to(self.device)

        checkpoint = torch.load(str(self.checkpoint_path), map_location=self.device)
        if isinstance(checkpoint, dict) and "model" in checkpoint:
            state_dict = checkpoint["model"]
        elif isinstance(checkpoint, dict) and "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        else:
            state_dict = checkpoint
        self.model.load_state_dict(state_dict, strict=False)
        self.model.eval()

    def _preprocess_rgb(self, rgb_image: np.ndarray) -> tuple[torch.Tensor, tuple[int, int]]:
        if rgb_image is None:
            raise ValueError("rgb_image cannot be None")
        if rgb_image.ndim != 3 or rgb_image.shape[2] != 3:
            raise ValueError(f"Expected RGB image with shape (H, W, 3), got {rgb_image.shape}")
        height, width = int(rgb_image.shape[0]), int(rgb_image.shape[1])

        image = rgb_image.astype(np.float32) / 255.0
        tensor = torch.from_numpy(np.transpose(image, (2, 0, 1))[None, ...]).contiguous().to(self.device)

        # CornerUNet downsamples 4x (stride 16). Pad to avoid shape mismatches.
        stride = 16
        pad_h = (stride - (height % stride)) % stride
        pad_w = (stride - (width % stride)) % stride
        if pad_h or pad_w:
            tensor = torch.nn.functional.pad(tensor, (0, pad_w, 0, pad_h), mode="constant", value=0.0)

        return tensor, (height, width)

    @torch.no_grad()
    def predict_maps(self, rgb_image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        input_tensor, unpad_hw = self._preprocess_rgb(rgb_image)
        logits = self.model(input_tensor)
        corner_maps = torch.sigmoid(logits[:, :4])[0].detach().float().cpu().numpy().astype(np.float32)
        paf_maps = torch.tanh(logits[:, 4:])[0].detach().float().cpu().numpy().astype(np.float32)
        out_h, out_w = unpad_hw
        return corner_maps[:, :out_h, :out_w], paf_maps[:, :out_h, :out_w]


def _points_to_center_and_size(points: dict[str, tuple[float, float]]) -> tuple[np.ndarray, np.ndarray]:
    tl = np.array(points["TL"], dtype=np.float32)
    tr = np.array(points["TR"], dtype=np.float32)
    br = np.array(points["BR"], dtype=np.float32)
    bl = np.array(points["BL"], dtype=np.float32)
    center = (tl + tr + br + bl) / 4.0
    width_px = float(max(np.linalg.norm(tr - tl), np.linalg.norm(br - bl)))
    height_px = float(max(np.linalg.norm(bl - tl), np.linalg.norm(br - tr)))
    return center.astype(np.float32), np.array([width_px, height_px], dtype=np.float32)


class DiffsimGateAffinityRacer(DiffSimRacer):
    def __init__(
        self,
        corner_model_path=DEFAULT_AFFINITY_CHECKPOINT,
        corner_conf_threshold=DEFAULT_CORNER_CONF_THRESHOLD,
        corner_topk=DEFAULT_CORNER_TOPK,
        corner_nms_radius=DEFAULT_CORNER_NMS_RADIUS,
        edge_min_score=DEFAULT_EDGE_MIN_SCORE,
        integral_samples=DEFAULT_INTEGRAL_SAMPLES,
        swap_rb: bool = False,
        depth_onnx_path=DEFAULT_DEPTH_ONNX_PATH,
        depth_input_width=DEFAULT_DEPTH_INPUT_WIDTH,
        depth_input_height=DEFAULT_DEPTH_INPUT_HEIGHT,
        depth_device=DEFAULT_DEPTH_DEVICE,
        gate_max_depth_m: float = DEFAULT_GATE_MAX_DEPTH_M,
        profile_gate: bool = False,
        **kwargs,
    ):
        self.corner_estimator = CornerAffinityEstimator(
            checkpoint_path=corner_model_path,
            device=kwargs.get("device", "cpu"),
        )
        self.corner_conf_threshold = float(corner_conf_threshold)
        self.corner_topk = int(corner_topk)
        self.corner_nms_radius = int(corner_nms_radius)
        self.edge_min_score = float(edge_min_score)
        self.integral_samples = int(integral_samples)
        self.swap_rb = bool(swap_rb)
        self._gate_postproc_debug_counter = 0
        self._gate_select_debug_counter = 0
        self.last_selected_gate_center_px = None
        self.last_backup_gate_center_px = None
        self.last_selected_gate_points_px = None
        self.last_backup_gate_points_px = None
        self.last_selected_gate_depth_m = None
        self.last_selected_gate_depth_source = None
        self.last_backup_gate_depth_m = None
        self.last_backup_gate_depth_source = None

        self.last_rgb_response = None
        self.last_corner_target_airsim = None
        self.last_corner_backup_target_airsim = None
        self.last_corner_candidate_targets_airsim = []
        self.last_corner_gate_candidates = []
        self.last_corner_candidate_timestamp = 0.0
        self.last_viz_center_px = None
        self.gate_max_depth_m = float(gate_max_depth_m)
        self.profile_gate = bool(profile_gate)
        self._gate_profile_counter = 0

        self.depth_estimator = DepthAnythingOnnxEstimator(
            onnx_path=depth_onnx_path,
            input_width=depth_input_width,
            input_height=depth_input_height,
            device=depth_device,
        )

        kwargs.setdefault("target_source", "corner_detection")
        kwargs.setdefault("sync_viz_to_control", True)
        super().__init__(**kwargs)

    def build_viz_overlay(self, aux: dict, frame_id: int) -> dict | None:
        rect = aux.get("segmentation_rect")
        if not isinstance(rect, dict):
            rect = aux.get("segmentation_primary_rect")
        if not isinstance(rect, dict):
            rect = None

        center_px = aux.get("segmentation_center_px")
        if center_px is None and isinstance(rect, dict):
            center_px = rect.get("center")

        points_px = None if rect is None else rect.get("points")
        depth_m = aux.get("segmentation_depth_m")
        if center_px is None and not points_px:
            return None
        return {
            "frame_id": int(frame_id),
            "center_px": None if center_px is None else np.asarray(center_px, dtype=np.float32).reshape(-1),
            "points_px": points_px,
            "depth_m": depth_m,
        }

    def _extract_gate_candidates(
        self,
        corner_maps: np.ndarray,
        paf_maps: np.ndarray,
        image_width: int,
        image_height: int,
        max_gates: int = 5,
    ) -> list[dict]:
        if corner_maps.ndim != 3 or corner_maps.shape[0] != 4:
            return []
        if paf_maps.ndim != 3 or paf_maps.shape[0] != 8:
            return []

        candidates = extract_corner_candidates(
            corner_maps,
            threshold=self.corner_conf_threshold,
            topk=self.corner_topk,
            nms_radius=self.corner_nms_radius,
        )
        if any(len(candidates[name]) == 0 for name in CORNER_NAMES):
            return []

        edge_matches = score_and_match_edges(
            candidates,
            paf_maps,
            edge_min_score=self.edge_min_score,
            integral_samples=self.integral_samples,
        )
        raw_gates = assemble_gates_from_edges(edge_matches)
        if not raw_gates:
            return []

        if self.debug_print:
            self._gate_postproc_debug_counter += 1
            if (self._gate_postproc_debug_counter - 1) % self.debug_print_every == 0:
                cand_counts = {k: len(v) for k, v in candidates.items()}
                match_counts = {f"{a}_{b}": len(edge_matches.get((a, b), [])) for a, b in EDGE_TYPES}
                print("[gate_postproc]", "candidates=", cand_counts, "edge_matches=", match_counts, "raw_gates=", len(raw_gates))

        gate_candidates: list[dict] = []
        for gate in raw_gates:
            points = gate.get("points", {})
            if not all(k in points for k in CORNER_NAMES):
                continue

            center, size = _points_to_center_and_size(points)

            scores = gate.get("scores", {})
            heatmap_score = float(np.mean([float(scores.get(k, 0.0)) for k in CORNER_NAMES]))
            gate_score = float(gate.get("gate_score", heatmap_score))

            gate_candidates.append(
                {
                    "points": {k: np.array(points[k], dtype=np.float32) for k in CORNER_NAMES},
                    "scores": {k: float(scores.get(k, 0.0)) for k in CORNER_NAMES},
                    "edge_scores": gate.get("edge_scores", {}),
                    "center": center,
                    "size": size,
                    "confidence": heatmap_score,
                    "gate_score": gate_score,
                }
            )

        gate_candidates.sort(key=lambda g: float(g.get("gate_score", 0.0)), reverse=True)

        return gate_candidates[: int(max_gates)]

    def get_sensor_images(self):
        responses = self.airsim_client_images.simGetImages(
            [
                airsim.ImageRequest(
                    "fpv_cam",
                    airsim.ImageType.Scene,
                    pixels_as_float=False,
                    compress=False,
                ),
            ],
            vehicle_name=self.drone_name,
        )
        if len(responses) < 1:
            return None, None, None, None

        rgb_response = responses[0]
        if rgb_response.width <= 0 or rgb_response.height <= 0:
            return None, None, None, None

        rgb = np.frombuffer(rgb_response.image_data_uint8, dtype=np.uint8)
        rgb = rgb.reshape(rgb_response.height, rgb_response.width, 3)
        if self.swap_rb:
            rgb = rgb[..., ::-1]

        depth = self.depth_estimator.predict_depth(rgb)
        self.last_rgb_response = rgb_response
        return depth, None, rgb, None

    def decorate_rgb_for_viz(self, rgb_image: np.ndarray) -> np.ndarray:
        """Overlay the currently selected gate (from the control loop) on the RGB image."""
        if cv2 is None:
            return rgb_image
        if rgb_image is None or rgb_image.ndim != 3 or rgb_image.shape[2] != 3:
            return rgb_image

        selected_center = self.last_selected_gate_center_px
        selected_points = self.last_selected_gate_points_px
        selected_depth = self.last_selected_gate_depth_m
        overlay = rgb_image.copy()
        h, w = overlay.shape[:2]

        # If the control loop hasn't produced a depth-valid "selected" gate yet,
        # fall back to visualizing the best 2D gate candidate (if available).
        is_fallback = False
        if selected_center is None:
            candidates = getattr(self, "last_corner_gate_candidates", None)
            if isinstance(candidates, list) and candidates:
                cand = candidates[0] or {}
                selected_center = cand.get("center")
                selected_points = cand.get("points")
                selected_depth = None
                is_fallback = True
            else:
                return rgb_image

        def _clamp_point(pt):
            x = int(round(float(pt[0])))
            y = int(round(float(pt[1])))
            x = max(0, min(w - 1, x))
            y = max(0, min(h - 1, y))
            return x, y

        # `overlay` is displayed with OpenCV (`cv2.imshow`), so colors are BGR.
        color_selected = (0, 215, 255) if is_fallback else (0, 255, 0)  # gold / green
        if isinstance(selected_points, dict) and selected_points:
            poly_order = ("TL", "TR", "BR", "BL")
            poly_pts = []
            for name in poly_order:
                pt = selected_points.get(name)
                if pt is None:
                    poly_pts = []
                    break
                poly_pts.append(_clamp_point(pt))
            if poly_pts:
                cv2.polylines(
                    overlay,
                    [np.asarray(poly_pts, dtype=np.int32).reshape((-1, 1, 2))],
                    isClosed=True,
                    color=color_selected,
                    thickness=3,
                    lineType=cv2.LINE_AA,
                )
            for name, pt in selected_points.items():
                if pt is None:
                    continue
                x, y = _clamp_point(pt)
                cv2.circle(overlay, (x, y), 5, color_selected, -1, lineType=cv2.LINE_AA)
                cv2.putText(
                    overlay,
                    str(name),
                    (x + 4, y - 4),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45,
                    (255, 255, 255),
                    1,
                    cv2.LINE_AA,
                )

        # Draw selected center.
        cx, cy = _clamp_point(selected_center)
        cv2.drawMarker(
            overlay,
            (cx, cy),
            color_selected,
            markerType=cv2.MARKER_CROSS,
            markerSize=24,
            thickness=3,
            line_type=cv2.LINE_AA,
        )
        depth_disp = None
        if isinstance(selected_depth, (float, int)) and np.isfinite(float(selected_depth)):
            depth_disp = f"{float(selected_depth):.2f}m"
        base_label = "detected_gate" if is_fallback else "selected_gate"
        label = base_label if depth_disp is None else f"{base_label} depth={depth_disp}"
        cv2.putText(
            overlay,
            label,
            (max(8, cx + 10), max(18, cy - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color_selected,
            3,
            cv2.LINE_AA,
        )
        if self.debug_print:
            cv2.putText(
                overlay,
                "overlay_on",
                (8, 22),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
        return overlay

    def _sample_depth_at_pixel(self, depth: np.ndarray | None, pixel: np.ndarray, source_size: tuple[int, int]) -> float:
        if depth is None or pixel is None:
            return np.inf
        if depth.ndim != 2:
            return np.inf

        src_h, src_w = source_size
        depth_h, depth_w = depth.shape[:2]
        # Map from RGB pixel coordinates into the depth-map pixel coordinates by
        # reversing the crop/pad performed by the depth estimator input fitting
        # (see `_fit_image_to_size` in `diffsim_depth_estimate_racer.py`).
        x_src = float(pixel[0])
        y_src = float(pixel[1])

        crop_left = (int(src_w) - int(depth_w)) // 2 if src_w > depth_w else 0
        crop_top = (int(src_h) - int(depth_h)) // 2 if src_h > depth_h else 0
        cropped_w = int(depth_w) if src_w > depth_w else int(src_w)
        cropped_h = int(depth_h) if src_h > depth_h else int(src_h)

        pad_left = (int(depth_w) - int(cropped_w)) // 2 if cropped_w < depth_w else 0
        pad_top = (int(depth_h) - int(cropped_h)) // 2 if cropped_h < depth_h else 0

        x = (x_src - float(crop_left)) if src_w > depth_w else (x_src + float(pad_left))
        y = (y_src - float(crop_top)) if src_h > depth_h else (y_src + float(pad_top))
        xi = int(round(x))
        yi = int(round(y))
        if xi < 0 or yi < 0 or xi >= depth_w or yi >= depth_h:
            return np.inf

        x0 = max(0, xi - 1)
        x1 = min(depth_w, xi + 2)
        y0 = max(0, yi - 1)
        y1 = min(depth_h, yi + 2)
        patch = depth[y0:y1, x0:x1]
        patch = patch[np.isfinite(patch) & (patch > 0.0)]
        if patch.size == 0:
            return np.inf
        return float(np.median(patch))

    def _gate_target_to_airsim(self, candidate, intr, rgb_response, gate_dimensions=None): 
        if candidate is None:
            return None, np.inf, "unavailable", {"reason": "candidate_none"}

        center_u, center_v = float(candidate["center"][0]), float(candidate["center"][1])
        rect_w_px = float(candidate["size"][0])
        rect_h_px = float(candidate["size"][1])

        source_size = (int(rgb_response.height), int(rgb_response.width))

        # Use the mean depth of the 4 gate corners (more stable than a single center sample),
        # but only if all 4 corner depths are consistent (range <= 2m). If the corner depths
        # are missing/inconsistent, skip depth-map-based depth entirely (no nominal-size fallback).
        depth_dbg: dict = {
            "reason": "unavailable",
            "center_px": (float(center_u), float(center_v)),
            "corner_depths_m": {},
            "max_corner_pair_diff_m": None,
        }
        corner_depths_by_name: dict[str, float] = {}
        points = candidate.get("points", {})
        for name in CORNER_NAMES:
            corner_px = points.get(name)
            if corner_px is None:
                continue
            depth_corner = self._sample_depth_at_pixel(self.last_depth, corner_px, source_size)
            depth_dbg["corner_depths_m"][str(name)] = (
                None if (not np.isfinite(depth_corner)) else float(depth_corner)
            )
            if np.isfinite(depth_corner) and depth_corner > 1e-6:
                corner_depths_by_name[str(name)] = float(depth_corner)

        depth_m = np.inf
        depth_source = "unavailable"

        if len(corner_depths_by_name) == len(CORNER_NAMES):
            corner_depths = np.array(
                [corner_depths_by_name[name] for name in CORNER_NAMES],
                dtype=np.float32,
            )
            corner_depth_consistency_thr_m = 50.0
            pair_diffs = []
            for i in range(int(corner_depths.shape[0])):
                for j in range(i + 1, int(corner_depths.shape[0])):
                    pair_diffs.append(float(abs(float(corner_depths[i]) - float(corner_depths[j]))))
            max_pair_diff = float(max(pair_diffs)) if pair_diffs else np.inf
            depth_dbg["max_corner_pair_diff_m"] = max_pair_diff
            if all(d <= corner_depth_consistency_thr_m for d in pair_diffs):
                depth_m = float(np.mean(corner_depths))
                depth_source = "depth_map_corners_mean"
            else:
                depth_m = np.inf
                depth_source = "depth_map_corners_inconsistent"
        else:
            depth_m = np.inf
            depth_source = "depth_map_corners_incomplete"

        if not np.isfinite(depth_m) or depth_m <= 1e-6:
            depth_dbg["reason"] = depth_source
            return None, np.inf, depth_source, depth_dbg

        x_off = (center_u - intr["cx"]) * depth_m / intr["fx"]
        y_off = (center_v - intr["cy"]) * depth_m / intr["fy"]
        target_rel_camera = np.array([depth_m, x_off, y_off], dtype=np.float32) #target point in camera frame 

        camera_position = np.array(
            [
                rgb_response.camera_position.x_val,
                rgb_response.camera_position.y_val,
                rgb_response.camera_position.z_val,
            ],
            dtype=np.float32,
        )
        camera_orientation = np.array(
            [
                rgb_response.camera_orientation.w_val,
                rgb_response.camera_orientation.x_val,
                rgb_response.camera_orientation.y_val,
                rgb_response.camera_orientation.z_val,
            ],
            dtype=np.float32,
        )
        camera_rot = quaternion_to_rotation_matrix(camera_orientation)
        p_target_airsim = camera_position + camera_rot @ target_rel_camera #target position in world frame  
        depth_dbg["reason"] = "ok"
        return p_target_airsim, depth_m, depth_source, depth_dbg

    def estimate_corner_target_point_airsim(self):
        t0_total = time.perf_counter()
        if self.last_rgb is None or self.last_rgb_response is None:
            if self.last_corner_target_airsim is not None:
                return self.last_corner_target_airsim, {
                    "segmentation_depth_source": "cached_primary",
                    "segmentation_target_cache_used": True,
                    "segmentation_target_cache_rank": 1,
                    "segmentation_center_px": self.last_selected_gate_center_px,
                    "segmentation_depth_m": self.last_selected_gate_depth_m,
                    "segmentation_selected_depth_m": self.last_selected_gate_depth_m,
                }
            if self.last_corner_backup_target_airsim is not None:
                return self.last_corner_backup_target_airsim, {
                    "segmentation_depth_source": "cached_secondary",
                    "segmentation_target_cache_used": True,
                    "segmentation_target_cache_rank": 2,
                    "segmentation_center_px": self.last_backup_gate_center_px,
                    "segmentation_depth_m": self.last_backup_gate_depth_m,
                    "segmentation_selected_depth_m": self.last_backup_gate_depth_m,
                }
            return None, {}

        t0_predict = time.perf_counter()
        corner_maps, paf_maps = self.corner_estimator.predict_maps(self.last_rgb)
        t_ms_predict = (time.perf_counter() - t0_predict) * 1000.0
        t0_post = time.perf_counter()
        gate_candidates = self._extract_gate_candidates(
            corner_maps,
            paf_maps,
            int(self.last_rgb_response.width),
            int(self.last_rgb_response.height),
        )
        t_ms_post = (time.perf_counter() - t0_post) * 1000.0
        self.last_corner_gate_candidates = gate_candidates
        self.last_corner_candidate_targets_airsim = []
        if not gate_candidates:
            # If the detector loses the gate temporarily, keep flying toward the last target
            # instead of switching to the forward-fallback target_v.
            if self.last_corner_target_airsim is not None:
                return self.last_corner_target_airsim, {
                    "segmentation_depth_source": "cached_primary",
                    "segmentation_target_cache_used": True,
                    "segmentation_target_cache_rank": 1,
                    "segmentation_center_px": self.last_selected_gate_center_px,
                    "segmentation_depth_m": self.last_selected_gate_depth_m,
                    "segmentation_selected_depth_m": self.last_selected_gate_depth_m,
                }
            if self.last_corner_backup_target_airsim is not None:
                return self.last_corner_backup_target_airsim, {
                    "segmentation_depth_source": "cached_secondary",
                    "segmentation_target_cache_used": True,
                    "segmentation_target_cache_rank": 2,
                    "segmentation_center_px": self.last_backup_gate_center_px,
                    "segmentation_depth_m": self.last_backup_gate_depth_m,
                    "segmentation_selected_depth_m": self.last_backup_gate_depth_m,
                }
            return None, {}

        intr = self.get_camera_intrinsics(
            int(self.last_rgb_response.width),
            int(self.last_rgb_response.height),
        )
        gate_dimensions = self.get_active_gate_dimensions()

        t0_targets = time.perf_counter()
        candidate_targets_all = []
        rejected_targets = []
        for gate_candidate in gate_candidates:
            target_airsim, target_depth_m, target_depth_source, target_dbg = self._gate_target_to_airsim(
                gate_candidate,
                intr,
                self.last_rgb_response,
                gate_dimensions=gate_dimensions,
            )
            if target_airsim is None:
                rejected_targets.append(
                    {
                        "candidate": gate_candidate,
                        "depth_source": target_depth_source,
                        "dbg": target_dbg,
                    }
                )
                continue
            candidate_targets_all.append(
                {
                    "candidate": gate_candidate,
                    "target_airsim": target_airsim,
                    "depth_m": target_depth_m,
                    "depth_source": target_depth_source,
                }
            )

        t_ms_targets = (time.perf_counter() - t0_targets) * 1000.0
        t_ms_total = (time.perf_counter() - t0_total) * 1000.0
        if self.profile_gate and self.debug_print:
            self._gate_profile_counter += 1
            if (self._gate_profile_counter - 1) % self.debug_print_every == 0:
                fps = (1000.0 / t_ms_total) if t_ms_total > 1e-6 else float("inf")
                print(
                    "[gate_profile]",
                    "predict_ms=",
                    round(t_ms_predict, 3),
                    "post_ms=",
                    round(t_ms_post, 3),
                    "targets_ms=",
                    round(t_ms_targets, 3),
                    "total_ms=",
                    round(t_ms_total, 3),
                    "fps=",
                    round(fps, 2),
                    "candidates=",
                    len(gate_candidates),
                )

        max_depth_m = float(self.gate_max_depth_m)
        candidate_targets = []
        for item in candidate_targets_all:
            depth_m = float(item.get("depth_m", np.inf))
            depth_ok = np.isfinite(depth_m) and depth_m > 1e-6
            if depth_ok and depth_m <= max_depth_m:
                candidate_targets.append(item)
            else:
                rejected_targets.append(
                    {
                        "candidate": item.get("candidate"),
                        "depth_source": item.get("depth_source"),
                        "dbg": {
                            "reason": "depth_too_far",
                            "depth_m": None if not np.isfinite(depth_m) else depth_m,
                            "max_depth_m": max_depth_m,
                        },
                    }
                )

        self.last_corner_candidate_targets_airsim = [item["target_airsim"] for item in candidate_targets]
        if not candidate_targets:
            if self.debug_print:
                rejected_preview = []
                for item in rejected_targets[:3]:
                    cand = item.get("candidate", {}) or {}
                    center = cand.get("center")
                    center_disp = None
                    if center is not None:
                        center_disp = (round(float(center[0]), 1), round(float(center[1]), 1))
                    dbg = item.get("dbg", {}) or {}
                    corner_depths = dbg.get("corner_depths_m", {}) or {}
                    rejected_preview.append(
                        {
                            "reason": str(dbg.get("reason")),
                            "max_pair_diff": dbg.get("max_corner_pair_diff_m"),
                            "max_depth_m": dbg.get("max_depth_m"),
                            "depth_m": dbg.get("depth_m"),
                            "center": center_disp,
                            "corners": {
                                k: (
                                    None
                                    if v is None or (isinstance(v, float) and (not np.isfinite(v)))
                                    else round(float(v), 3)
                                )
                                for k, v in corner_depths.items()
                            },
                        }
                    )
                if rejected_preview and ((self._gate_postproc_debug_counter - 1) % self.debug_print_every == 0):
                    print("[gate_depth_reject]", "rejected=", len(rejected_targets), "preview=", rejected_preview)
            if self.last_corner_target_airsim is not None:
                return self.last_corner_target_airsim, {
                    "segmentation_depth_source": "cached_primary",
                    "segmentation_target_cache_used": True,
                    "segmentation_target_cache_rank": 1,
                    "segmentation_center_px": self.last_selected_gate_center_px,
                    "segmentation_depth_m": self.last_selected_gate_depth_m,
                    "segmentation_selected_depth_m": self.last_selected_gate_depth_m,
                }
            if self.last_corner_backup_target_airsim is not None:
                return self.last_corner_backup_target_airsim, {
                    "segmentation_depth_source": "cached_secondary",
                    "segmentation_target_cache_used": True,
                    "segmentation_target_cache_rank": 2,
                    "segmentation_center_px": self.last_backup_gate_center_px,
                    "segmentation_depth_m": self.last_backup_gate_depth_m,
                    "segmentation_selected_depth_m": self.last_backup_gate_depth_m,
                }
            return None, {}

        def _closest_gate_key(item: dict) -> tuple[int, float]:
            depth_m = float(item.get("depth_m", np.inf))
            depth_ok = np.isfinite(depth_m) and depth_m > 1e-6
            tier = 0 if depth_ok else 1
            return (tier, depth_m if depth_ok else np.inf)

        # Select the closest gate purely by estimated depth (smaller = closer).
        sorted_targets = sorted(candidate_targets, key=_closest_gate_key)
        selected = sorted_targets[0]
        selected_rank = 1

        backup = sorted_targets[1] if len(sorted_targets) > 1 else selected
        backup_rank = 2 if len(sorted_targets) > 1 else 1

        candidate = selected["candidate"]
        p_target_airsim = selected["target_airsim"]
        depth_m = float(selected["depth_m"])
        depth_source = str(selected["depth_source"])

        backup_candidate = backup["candidate"]
        backup_target_airsim = backup["target_airsim"]
        backup_depth_m = float(backup["depth_m"])
        backup_depth_source = str(backup["depth_source"])

        if self.debug_print:
            self._gate_select_debug_counter += 1
            if (self._gate_select_debug_counter - 1) % self.debug_print_every == 0:
                summary = []
                for idx, item in enumerate(sorted_targets[:5], start=1):
                    cand = item.get("candidate", {})
                    center = cand.get("center")
                    center_disp = None
                    if center is not None:
                        center_disp = (round(float(center[0]), 1), round(float(center[1]), 1))
                    depth_val = float(item.get("depth_m", np.inf))
                    depth_disp = None if (not np.isfinite(depth_val) or depth_val <= 1e-6) else round(depth_val, 3)
                    summary.append(
                        {
                            "rank": idx,
                            "depth": depth_disp,
                            "src": str(item.get("depth_source", "unavailable")),
                            "center": center_disp,
                        }
                    )
                print(
                    "[gate_select]",
                    "selected_rank=",
                    selected_rank,
                    "selected_depth=",
                    round(depth_m, 3) if np.isfinite(depth_m) else depth_m,
                    "selected_src=",
                    depth_source,
                    "candidates=",
                    summary,
                )

        self.last_corner_target_airsim = p_target_airsim
        self.last_corner_backup_target_airsim = backup_target_airsim
        self.last_corner_candidate_timestamp = time.time()
        self.last_selected_gate_center_px = np.asarray(candidate["center"], dtype=np.float32).copy()
        self.last_backup_gate_center_px = np.asarray(backup_candidate["center"], dtype=np.float32).copy()
        self.last_selected_gate_points_px = {
            str(k): np.asarray(v, dtype=np.float32).copy() for k, v in (candidate.get("points", {}) or {}).items()
        }
        self.last_backup_gate_points_px = {
            str(k): np.asarray(v, dtype=np.float32).copy() for k, v in (backup_candidate.get("points", {}) or {}).items()
        }
        self.last_selected_gate_depth_m = depth_m
        self.last_selected_gate_depth_source = depth_source
        self.last_backup_gate_depth_m = backup_depth_m
        self.last_backup_gate_depth_source = backup_depth_source

        aux = {
            "segmentation_mask": None,
            "segmentation_rect": candidate,
            "segmentation_primary_rect": candidate,
            "segmentation_backup_rect": backup_candidate,
            "segmentation_center_px": candidate["center"],
            "segmentation_rect_size_px": candidate["size"],
            "segmentation_depth_m": depth_m,
            "segmentation_depth_source": depth_source,
            "segmentation_primary_depth_m": depth_m,
            "segmentation_primary_rank": selected_rank,
            "segmentation_selected_depth_m": depth_m,
            "segmentation_selected_rank": selected_rank,
            "segmentation_backup_depth_m": backup_depth_m,
            "segmentation_backup_depth_source": backup_depth_source,
            "segmentation_backup_rank": backup_rank,
            "segmentation_blob_count": len(candidate_targets),
            "segmentation_blob_selection": "corner_affinity",
            "segmentation_blob_depth_m": depth_m,
            "segmentation_promoted": False,
            "segmentation_promote_depth_threshold": self.corner_conf_threshold,
            "segmentation_blob_backup_depth_m": backup_depth_m,
            "segmentation_target_airsim": p_target_airsim,
            "segmentation_backup_target_airsim": backup_target_airsim,
            "camera_intrinsics": intr,
            "gate_corner_points_px": candidate["points"],
            "gate_corner_scores": candidate["scores"],
            "gate_center_px": candidate["center"],
            "gate_confidence": candidate["confidence"],
            "corner_gate_candidates": gate_candidates,
            "corner_gate_count": len(gate_candidates),
            "corner_gate_targets_airsim": self.last_corner_candidate_targets_airsim,
            "corner_gate_target_records": candidate_targets,
        }
        return p_target_airsim, aux

    def build_state_tensor(self, state_dict):
        # Same control/target logic pattern as `diffsim_gate_detection_racer.py`.
        if torch is None:
            raise ImportError("torch is required to build model inputs.")

        position = state_dict["position"]
        position_fm = airsim_to_flightmare_vector(position)
        orientation = state_dict["orientation"]
        linear_velocity_airsim = state_dict["linear_velocity"]
        env_rot_airsim = quaternion_to_rotation_matrix(orientation)
        env_rot = airsim_to_flightmare_rotation(env_rot_airsim)
        linear_velocity = airsim_to_flightmare_vector(linear_velocity_airsim)

        forward = env_rot[:, 0].copy()
        forward[2] = 0.0
        if np.linalg.norm(forward) < 1e-6:
            forward = self.current_forward.copy()
        forward = normalize(forward)
        up = np.array([0.0, 0.0, 1.0], dtype=np.float32)
        left = normalize(np.cross(up, forward))
        yaw_only_rot = np.stack([forward, left, up], axis=-1)

        p_target_airsim = None
        p_target_fm = None
        target_v = None
        corner_aux = {}

        gt_target_airsim = None
        if self.target_source == "ground_truth" and self.gate_poses_ground_truth:
            gt_target_airsim = self.ground_truth_gate_target_point_airsim(position)

        if self.target_source == "corner_detection":
            p_target_airsim, corner_aux = self.estimate_corner_target_point_airsim()
            if p_target_airsim is not None:
                if corner_aux.get("segmentation_target_cache_used"):
                    cache_rank = int(corner_aux.get("segmentation_target_cache_rank", 2))
                    self.last_target_source = f"corner_detection_cache_rank_{cache_rank}"
                else:
                    self.last_target_source = "corner_detection"
        elif self.target_source == "ground_truth":
            p_target_airsim = gt_target_airsim
            if p_target_airsim is not None:
                self.last_target_source = "ground_truth_gate"

        if p_target_airsim is not None:
            p_target_fm = airsim_to_flightmare_vector(p_target_airsim)
            target_v = p_target_fm - position_fm
            target_v_norm = np.linalg.norm(target_v)
            if target_v_norm > 1e-6:
                target_v = target_v / target_v_norm * min(target_v_norm, self.target_speed)
            else:
                target_v = None

        if target_v is None:
            self.last_target_source = "forward_fallback"
            target_v = forward * self.target_speed

        local_target_v = target_v @ yaw_only_rot
        state_parts = [local_target_v, env_rot[:, 2], np.array([self.margin], dtype=np.float32)]
        if not self.no_odom:
            local_velocity = linear_velocity @ yaw_only_rot
            state_parts.insert(0, local_velocity)
        state_np = np.concatenate(state_parts)
        state_tensor = torch.as_tensor(state_np, dtype=torch.float32)[None]
        aux = {
            "position_fm": position_fm,
            "position_airsim": position,
            "env_rot_fm": env_rot,
            "env_rot_airsim": env_rot_airsim,
            "local_velocity_fm": linear_velocity @ yaw_only_rot,
            "p_target_fm": p_target_fm,
            "p_target_airsim": p_target_airsim,
            "target_vec_airsim_world": None if p_target_airsim is None else (p_target_airsim - position),
            "target_v_fm": target_v,
        }
        aux.update(corner_aux)
        return state_tensor, yaw_only_rot, aux

    def augment_rgb_viz(self, rgb_viz):
        overlay = getattr(self, "_viz_overlay_active", None)
        if overlay is None or cv2 is None:
            return rgb_viz

        center_px = overlay.get("center_px")
        points_px = overlay.get("points_px")
        depth_m = overlay.get("depth_m")
        if center_px is None and not points_px:
            return rgb_viz

        try:
            canvas = rgb_viz.copy()
            h, w = canvas.shape[:2]

            def _clamp(pt):
                x = int(round(float(pt[0])))
                y = int(round(float(pt[1])))
                x = max(0, min(w - 1, x))
                y = max(0, min(h - 1, y))
                return x, y

            color = (0, 255, 0)  # BGR
            if isinstance(points_px, dict) and points_px:
                poly_order = ("TL", "TR", "BR", "BL")
                poly_pts = []
                for name in poly_order:
                    pt = points_px.get(name)
                    if pt is None:
                        poly_pts = []
                        break
                    poly_pts.append(_clamp(pt))
                if poly_pts:
                    cv2.polylines(
                        canvas,
                        [np.asarray(poly_pts, dtype=np.int32).reshape((-1, 1, 2))],
                        isClosed=True,
                        color=color,
                        thickness=1,
                        lineType=cv2.LINE_AA,
                    )
                for name, pt in points_px.items():
                    if pt is None:
                        continue
                    x, y = _clamp(pt)
                    cv2.circle(canvas, (x, y), 1, color, -1, lineType=cv2.LINE_AA)
                    cv2.putText(
                        canvas,
                        str(name),
                        (x + 4, y - 4),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.45,
                        (255, 255, 255),
                        1,
                        cv2.LINE_AA,
                    )

            if center_px is not None:
                cx, cy = _clamp(center_px)
                cv2.drawMarker(
                    canvas,
                    (cx, cy),
                    color,
                    markerType=cv2.MARKER_CROSS,
                    markerSize=24,
                    thickness=1,
                    line_type=cv2.LINE_AA,
                )
                label = "gate"
                if isinstance(depth_m, (float, int)) and np.isfinite(float(depth_m)):
                    label = f"gate depth={float(depth_m):.2f}m"
                cv2.putText(
                    canvas,
                    label,
                    (max(8, cx + 10), max(18, cy - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    color,
                    2,
                    cv2.LINE_AA,
                )

            return canvas
        except Exception:
            return rgb_viz

    def augment_depth_raw_viz(self, depth_viz):
        overlay = getattr(self, "_viz_overlay_active", None)
        if overlay is None or cv2 is None:
            return depth_viz

        center_px = overlay.get("center_px")
        if center_px is None:
            return depth_viz

        try:
            h, w = depth_viz.shape[:2]
            cx = int(round(float(center_px[0])))
            cy = int(round(float(center_px[1])))
            cx = max(0, min(w - 1, cx))
            cy = max(0, min(h - 1, cy))
            depth_viz = depth_viz.copy()
            cv2.circle(depth_viz, (cx, cy), 6, 255, 2, lineType=cv2.LINE_AA)
            cv2.drawMarker(
                depth_viz,
                (cx, cy),
                255,
                markerType=cv2.MARKER_CROSS,
                markerSize=30,
                thickness=2,
                line_type=cv2.LINE_AA,
            )
            cv2.putText(
                depth_viz,
                f"center=({cx},{cy})",
                (8, 22),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                255,
                2,
                cv2.LINE_AA,
            )
        except Exception:
            if getattr(self, "debug_print", False):
                print("[viz]", "augment_depth_raw_viz_failed")
        return depth_viz

    def infer_acceleration(self, depth, state_dict):
        accel_world, target_v, aux = super().infer_acceleration(depth, state_dict)
        center_px = aux.get("segmentation_center_px")
        if center_px is None:
            center_px = aux.get("gate_center_px")
        self.last_viz_center_px = center_px

        return accel_world, target_v, aux


def build_args():
    parser = ArgumentParser(description="Run the DiffSim racer using corner+PAF (affinity) gate detection.")
    parser.add_argument(
        "--level_name",
        type=str,
        choices=[
            "Soccer_Field_Easy",
            "Soccer_Field_Medium",
            "ZhangJiaJie_Medium",
            "Building99_Hard",
            "Qualifier_Tier_1",
            "Qualifier_Tier_2",
            "Qualifier_Tier_3",
            "Final_Tier_1",
            "Final_Tier_2",
            "Final_Tier_3",
        ],
        default="Soccer_Field_Easy",
    )
    parser.add_argument("--race_tier", type=int, choices=[1, 2, 3], default=1)
    parser.add_argument("--drone_name", type=str, default="drone_1")
    parser.add_argument("--control_mode", type=str, choices=["velocity", "attitude"], default="attitude")
    parser.add_argument("--target_speed", type=float, default=10.0)
    parser.add_argument("--hover_throttle", type=float, default=0.9)
    parser.add_argument("--control_period", type=float, default=0.1) #airsim holds control for___ seconds, control thread sleeps for ___ seconds(when not syncing to depth)
    parser.add_argument("--image_period", type=float, default=0.0) #airsim sleeps for ___seconds in image thread 
    parser.add_argument(
        "--sync_control_to_depth",
        action=BooleanOptionalAction,
        default=True,
        help="When enabled, the control loop waits for a new depth frame before running inference.",
    )
    parser.add_argument("--sync_depth_timeout_sec", type=float, default=0.5)
    parser.add_argument(
        "--sync_viz_to_control",
        action=BooleanOptionalAction,
        default=False,
        help="When enabled, the image callback waits briefly for the matching control overlay before drawing viz.",
    )
    parser.add_argument(
        "--sync_viz_timeout_sec",
        type=float,
        default=0.3,
        help="Max seconds to wait for a matching overlay when sync_viz_to_control is enabled.",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default=DEFAULT_MODEL_PATH,
        help="Path to the learned control policy checkpoint.",
    )
    parser.add_argument(
        "--corner_model_path",
        type=str,
        default=str(DEFAULT_AFFINITY_CHECKPOINT),
        help="Path to the trained corner+PAF checkpoint.",
    )
    parser.add_argument("--corner_conf_threshold", type=float, default=DEFAULT_CORNER_CONF_THRESHOLD)
    parser.add_argument("--corner_topk", type=int, default=DEFAULT_CORNER_TOPK)
    parser.add_argument("--corner_nms_radius", type=int, default=DEFAULT_CORNER_NMS_RADIUS)
    parser.add_argument("--edge_min_score", type=float, default=DEFAULT_EDGE_MIN_SCORE)
    parser.add_argument("--integral_samples", type=int, default=DEFAULT_INTEGRAL_SAMPLES)
    parser.add_argument(
        "--gate_max_depth_m",
        type=float,
        default=DEFAULT_GATE_MAX_DEPTH_M,
        help="Reject gate candidates farther than this estimated depth (meters).",
    )
    parser.add_argument(
        "--profile_gate",
        action="store_true",
        default=False,
        help="Print gate detection timing every --debug_print_every steps (requires --debug_print).",
    )
    parser.add_argument("--swap_rb", type=bool, default=True, help="Swap red/blue channels on AirSim Scene frames before inference (use if colors look wrong).",)
    parser.add_argument(
        "--depth_onnx_path",
        "--depth_checkpoint",
        dest="depth_onnx_path",
        type=str,
        default=str(DEFAULT_DEPTH_ONNX_PATH),
        help="Path to the RGB-to-depth ONNX model.",
    )
    parser.add_argument("--depth_input_width", type=int, default=DEFAULT_DEPTH_INPUT_WIDTH)
    parser.add_argument("--depth_input_height", type=int, default=DEFAULT_DEPTH_INPUT_HEIGHT)
    parser.add_argument("--depth_device", type=str, default=DEFAULT_DEPTH_DEVICE)
    parser.add_argument("--dim_obs", type=int, default=10)
    parser.add_argument("--dim_action", type=int, default=6)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--no_odom", action="store_true", default=False)
    parser.add_argument("--margin", type=float, default=0.2)
    parser.add_argument("--takeoff_height", type=float, default=1.0)
    parser.add_argument("--post_takeoff_delay", type=float, default=1.0)
    parser.add_argument("--max_velocity", type=float, default=8.0)
    parser.add_argument("--max_vertical_velocity", type=float, default=3.0)
    parser.add_argument("--velocity_gain_xy", type=float, default=2.0)
    parser.add_argument("--velocity_gain_z", type=float, default=2.0)
    parser.add_argument(
        "--target_source",
        type=str,
        choices=["ground_truth", "corner_detection"],
        default="corner_detection",
    )
    parser.add_argument("--debug_print", action="store_true", default=False)
    parser.add_argument("--debug_print_every", type=int, default=1)
    parser.add_argument("--viz_rgb", dest="viz_rgb", action="store_true", default=False)
    parser.add_argument("--viz_depth", dest="viz_depth", action="store_true", default=False)
    parser.add_argument("--viz_depth_raw", dest="viz_depth_raw", action="store_true", default=False)
    return parser.parse_args()


def main():
    args = build_args()

    racer = DiffsimGateAffinityRacer(
        corner_model_path=args.corner_model_path,
        corner_conf_threshold=args.corner_conf_threshold,
        corner_topk=args.corner_topk,
        corner_nms_radius=args.corner_nms_radius,
        edge_min_score=args.edge_min_score,
        integral_samples=args.integral_samples,
        gate_max_depth_m=args.gate_max_depth_m,
        profile_gate=args.profile_gate,
        swap_rb=args.swap_rb,
        depth_onnx_path=args.depth_onnx_path,
        depth_input_width=args.depth_input_width,
        depth_input_height=args.depth_input_height,
        depth_device=args.depth_device,
        drone_name=args.drone_name,
        viz_rgb=args.viz_rgb,
        viz_depth=args.viz_depth,
        viz_depth_raw=args.viz_depth_raw,
        control_mode=args.control_mode,
        control_period=args.control_period,
        image_period=args.image_period,
        sync_control_to_depth=args.sync_control_to_depth,
        sync_depth_timeout_sec=args.sync_depth_timeout_sec,
        sync_viz_to_control=args.sync_viz_to_control,
        sync_viz_timeout_sec=args.sync_viz_timeout_sec,
        hover_throttle=args.hover_throttle,
        target_speed=args.target_speed,
        model_path=args.model_path,
        dim_obs=args.dim_obs,
        dim_action=args.dim_action,
        device=args.device,
        no_odom=args.no_odom,
        margin=args.margin,
        post_takeoff_delay=args.post_takeoff_delay,
        max_velocity=args.max_velocity,
        max_vertical_velocity=args.max_vertical_velocity,
        velocity_gain_xy=args.velocity_gain_xy,
        velocity_gain_z=args.velocity_gain_z,
        target_source=args.target_source,
        debug_print=args.debug_print,
        debug_print_every=args.debug_print_every,
    )

    racer.load_level(args.level_name)
    racer.start_race(args.race_tier)
    racer.initialize_drone()
    racer.takeoff(takeoff_height=args.takeoff_height)
    if args.target_source == "ground_truth":
        racer.get_ground_truth_gate_poses()
    racer.start_model_control()

    try:
        while True:
            time.sleep(1.0)
    except KeyboardInterrupt:
        pass
    finally:
        racer.stop_threads()
        racer.reset_race()


if __name__ == "__main__":
    main()
