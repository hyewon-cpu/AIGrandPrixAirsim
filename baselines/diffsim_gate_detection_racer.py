from __future__ import annotations

"""DiffSim racer that uses a corner U-Net to pick the next gate target.

This is modeled after `diffsim_depth_estimate_racer.py`, but it replaces the
segmentation-based target selection with a gate-corner detector:

- the RGB frame is fed into the corner U-Net
- the four heatmaps (TL, TR, BL, BR) are converted into corner points
- those points are averaged into a gate center
- depth at that center is used to back-project a 3D target point
- the learned control policy then uses that target point to build target velocity
"""

from argparse import ArgumentParser
from pathlib import Path
import math
import sys
import time

import airsimdroneracinglab as airsim
import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
GATE_DETECTION_DIR = REPO_ROOT / "gate_detection"
DEFAULT_CORNER_CHECKPOINT = REPO_ROOT / "gate_detection" / "corner_checkpoints" / "best.pt"
DEFAULT_CORNER_CONF_THRESHOLD = 0.5
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

if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
if str(GATE_DETECTION_DIR) not in sys.path:
    sys.path.insert(0, str(GATE_DETECTION_DIR))

try:
    import cv2
except ImportError:  # pragma: no cover
    cv2 = None

try:
    import torch
    import torch.nn as nn
except ImportError as exc:  # pragma: no cover
    torch = None
    nn = None
    TORCH_IMPORT_ERROR = exc
else:
    TORCH_IMPORT_ERROR = None

from diffsim_depth_estimate_racer import (
    DepthAnythingOnnxEstimator,
)
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


class CornerHeatmapEstimator:
    def __init__(
        self,
        checkpoint_path=DEFAULT_CORNER_CHECKPOINT,
        device="cpu",
    ):
        if torch is None:
            raise ImportError(
                "torch is required to run the corner detector. "
                "Install PyTorch in this environment first."
            ) from TORCH_IMPORT_ERROR
        from corner_unet import CornerUNet

        checkpoint_path = Path(checkpoint_path).expanduser().resolve()
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Corner checkpoint does not exist: {checkpoint_path}")

        self.checkpoint_path = checkpoint_path
        self.device = torch.device(device)
        self.model = CornerUNet().to(self.device)

        checkpoint = torch.load(self.checkpoint_path, map_location=self.device)
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
    def predict_heatmaps(self, rgb_image: np.ndarray) -> np.ndarray:
        input_tensor, unpad_hw = self._preprocess_rgb(rgb_image)
        logits = self.model(input_tensor)
        heatmaps = torch.sigmoid(logits).detach().cpu().numpy()[0]
        heatmaps = heatmaps.astype(np.float32)
        out_h, out_w = unpad_hw
        return heatmaps[:, :out_h, :out_w]


class DiffsimGateDetectRacer(DiffSimRacer):
    def __init__(
        self,
        corner_model_path=DEFAULT_CORNER_CHECKPOINT,
        corner_conf_threshold=DEFAULT_CORNER_CONF_THRESHOLD,
        viz_corner_heatmaps: bool = False,
        viz_corner_stride: int = 1,
        swap_rb: bool = True,
        depth_onnx_path=DEFAULT_DEPTH_ONNX_PATH,
        depth_input_width=DEFAULT_DEPTH_INPUT_WIDTH,
        depth_input_height=DEFAULT_DEPTH_INPUT_HEIGHT,
        depth_device=DEFAULT_DEPTH_DEVICE,
        **kwargs,
    ):
        self.corner_estimator = CornerHeatmapEstimator(
            checkpoint_path=corner_model_path,
            device=kwargs.get("device", "cpu"),
        )
        self.corner_conf_threshold = float(corner_conf_threshold)
        self.viz_corner_heatmaps = bool(viz_corner_heatmaps)
        self.viz_corner_stride = max(1, int(viz_corner_stride))
        self._viz_corner_counter = 0
        self._warned_missing_cv2_heatmaps = False
        self.swap_rb = bool(swap_rb)
        self.last_rgb_response = None
        self.last_corner_target_airsim = None
        self.last_corner_backup_target_airsim = None
        self.last_corner_candidate_targets_airsim = []
        self.last_corner_gate_candidates = []
        self.last_corner_candidate_timestamp = 0.0
        self.depth_estimator = DepthAnythingOnnxEstimator(
            onnx_path=depth_onnx_path,
            input_width=depth_input_width,
            input_height=depth_input_height,
            device=depth_device,
        )
        # Allow callers/CLI to override target_source (e.g. ground_truth).
        kwargs.setdefault("target_source", "corner_detection")
        super().__init__(**kwargs)

    def _viz_corner_overlay(self, rgb_image: np.ndarray, gate_candidates: list[dict]) -> None:
        if not self.viz_corner_heatmaps:
            return
        if cv2 is None:
            if not self._warned_missing_cv2_heatmaps:
                print("[corner_overlay] OpenCV (cv2) not available; cannot visualize overlays.")
                self._warned_missing_cv2_heatmaps = True
            return
        if rgb_image is None:
            return
        if rgb_image.ndim != 3 or rgb_image.shape[2] != 3:
            return

        self._viz_corner_counter += 1
        if (self._viz_corner_counter - 1) % self.viz_corner_stride != 0:
            return

        overlay = rgb_image.copy()
        height, width = overlay.shape[:2]
        palette = [
            (0, 0, 255),
            (0, 165, 255),
            (0, 255, 255),
            (0, 255, 0),
            (255, 0, 0),
        ]

        for gate_idx, gate_candidate in enumerate(gate_candidates[:5]):
            color = palette[gate_idx % len(palette)]
            points = gate_candidate.get("points", {})
            center = gate_candidate.get("center")

            if center is not None:
                cx, cy = int(round(float(center[0]))), int(round(float(center[1])))
                cv2.circle(overlay, (cx, cy), 5, color, -1)
                cv2.putText(
                    overlay,
                    f"G{gate_idx + 1}:{gate_candidate.get('gate_score', 0.0):.2f}",
                    (cx + 6, cy - 6),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45,
                    color,
                    1,
                    cv2.LINE_AA,
                )

            for name, point in points.items():
                px, py = int(round(float(point[0]))), int(round(float(point[1])))
                cv2.circle(overlay, (px, py), 4, color, -1)
                cv2.putText(
                    overlay,
                    name,
                    (px + 4, py - 4),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.4,
                    (255, 255, 255),
                    1,
                    cv2.LINE_AA,
                )

        cv2.putText(
            overlay,
            f"gate_candidates={len(gate_candidates)}",
            (8, max(22, height - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        # OpenCV expects BGR; our pipeline uses RGB for model inputs.
        cv2.imshow("corner_overlay", cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
        cv2.waitKey(1)

    def _find_heatmap_peaks(
        self,
        heatmap: np.ndarray,
        max_peaks: int = 4,
        min_distance: int = 6,
        threshold: float | None = None,
    ) -> list[dict]:
        if heatmap.ndim != 2:
            return []

        work = heatmap.astype(np.float32, copy=True)
        threshold = self.corner_conf_threshold if threshold is None else float(threshold)
        peaks: list[dict] = []
        suppression = max(1, int(min_distance))

        for _ in range(max_peaks):
            flat_index = int(np.argmax(work))
            score = float(work.flat[flat_index])
            if not np.isfinite(score) or score < threshold:
                break
            y_idx, x_idx = np.unravel_index(flat_index, work.shape)
            peaks.append(
                {
                    "x_idx": int(x_idx),
                    "y_idx": int(y_idx),
                    "score": score,
                }
            )

            x0 = max(0, x_idx - suppression)
            x1 = min(work.shape[1], x_idx + suppression + 1)
            y0 = max(0, y_idx - suppression)
            y1 = min(work.shape[0], y_idx + suppression + 1)
            work[y0:y1, x0:x1] = -np.inf

        return peaks

    def _extract_gate_candidates(
        self,
        heatmaps: np.ndarray,
        image_width: int,
        image_height: int,
        max_peaks_per_corner: int = 4,
        max_gates: int = 5,
    ) -> list[dict]:
        if heatmaps.ndim != 3 or heatmaps.shape[0] != 4:
            return []

        hmap_h, hmap_w = heatmaps.shape[1], heatmaps.shape[2]
        corner_names = ("TL", "TR", "BL", "BR")
        corner_candidates: dict[str, list[dict]] = {}
        for idx, name in enumerate(corner_names):
            peaks = self._find_heatmap_peaks(
                heatmaps[idx],
                max_peaks=max_peaks_per_corner,
                min_distance=max(4, min(hmap_h, hmap_w) // 20),
            )
            candidates = []
            for peak in peaks:
                x = (float(peak["x_idx"]) + 0.5) * float(image_width) / float(hmap_w)
                y = (float(peak["y_idx"]) + 0.5) * float(image_height) / float(hmap_h)
                candidates.append(
                    {
                        "label": name,
                        "point": np.array([x, y], dtype=np.float32),
                        "score": float(peak["score"]),
                        "pixel": np.array([x, y], dtype=np.float32),
                    }
                )
            corner_candidates[name] = candidates

        if any(not corner_candidates[name] for name in corner_names):
            return []

        depth_map = self.last_depth if isinstance(self.last_depth, np.ndarray) else None
        source_size = (int(image_height), int(image_width))

        gate_candidates: list[dict] = []
        for tl in corner_candidates["TL"]:
            for tr in corner_candidates["TR"]:
                for bl in corner_candidates["BL"]:
                    for br in corner_candidates["BR"]:
                        points = {
                            "TL": tl["point"],
                            "TR": tr["point"],
                            "BL": bl["point"],
                            "BR": br["point"],
                        }
                        scores = {
                            "TL": float(tl["score"]),
                            "TR": float(tr["score"]),
                            "BL": float(bl["score"]),
                            "BR": float(br["score"]),
                        }

                        xs = np.array([points[name][0] for name in corner_names], dtype=np.float32)
                        ys = np.array([points[name][1] for name in corner_names], dtype=np.float32)
                        if not (xs[0] < xs[1] and xs[2] < xs[3] and ys[0] < ys[2] and ys[1] < ys[3]):
                            continue

                        tl_pt = points["TL"]
                        tr_pt = points["TR"]
                        bl_pt = points["BL"]
                        br_pt = points["BR"]
                        center = (tl_pt + tr_pt + bl_pt + br_pt) / 4.0
                        width_px = 0.5 * (np.linalg.norm(tr_pt - tl_pt) + np.linalg.norm(br_pt - bl_pt))
                        height_px = 0.5 * (np.linalg.norm(bl_pt - tl_pt) + np.linalg.norm(br_pt - tr_pt))
                        if width_px < 1.0 or height_px < 1.0:
                            continue

                        depth_samples = {}
                        finite_depths = []
                        if depth_map is not None:
                            for name in corner_names:
                                depth_value = self._sample_depth_at_pixel(
                                    depth_map,
                                    points[name],
                                    source_size,
                                )
                                depth_samples[name] = depth_value
                                if np.isfinite(depth_value):
                                    finite_depths.append(depth_value)

                        if depth_map is not None and len(finite_depths) < 4:
                            continue

                        depth_mean = float(np.mean(finite_depths)) if finite_depths else np.inf
                        depth_std = float(np.std(finite_depths)) if finite_depths else np.inf
                        if np.isfinite(depth_std) and depth_mean > 1e-6:
                            depth_consistency = 1.0 / (1.0 + depth_std / depth_mean)
                        else:
                            depth_consistency = 0.0

                        rect_aspect = width_px / max(height_px, 1e-6)
                        rect_shape_penalty = abs(math.log(max(rect_aspect, 1e-6)))
                        geom_score = 1.0 / (1.0 + rect_shape_penalty)
                        heatmap_score = float(np.mean(list(scores.values())))
                        gate_score = heatmap_score * (0.6 + 0.4 * depth_consistency) * geom_score
                        if np.isfinite(depth_mean):
                            gate_score /= (1.0 + 0.02 * depth_mean)

                        gate_candidates.append(
                            {
                                "points": points,
                                "scores": scores,
                                "center": center.astype(np.float32),
                                "size": np.array([width_px, height_px], dtype=np.float32),
                                "confidence": heatmap_score,
                                "depth_samples": depth_samples,
                                "depth_mean": depth_mean,
                                "depth_std": depth_std,
                                "gate_score": float(gate_score),
                            }
                        )

        gate_candidates.sort(key=lambda item: item["gate_score"], reverse=True)

        selected: list[dict] = []
        min_center_sep = 0.35 * float(min(image_width, image_height))
        for candidate in gate_candidates:
            if len(selected) >= max_gates:
                break
            center = candidate["center"]
            if any(np.linalg.norm(center - other["center"]) < min_center_sep for other in selected):
                continue
            selected.append(candidate)

        return selected

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

        rgb = np.frombuffer(rgb_response.image_data_uint8, dtype=np.uint8).copy()
        rgb = rgb.reshape(rgb_response.height, rgb_response.width, 3)
        if self.swap_rb:
            rgb = rgb[..., ::-1]
        depth = self.depth_estimator.predict_depth(rgb)
        self.last_rgb_response = rgb_response
        return depth, None, rgb, None

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

    def _gate_target_to_airsim(
        self,
        candidate,
        intr,
        rgb_response,
        gate_dimensions=None,
    ):
        if candidate is None:
            return None, np.inf, "unavailable"

        center_u, center_v = float(candidate["center"][0]), float(candidate["center"][1])
        rect_w_px = float(candidate["size"][0])
        rect_h_px = float(candidate["size"][1])
        depth_m = self._sample_depth_at_pixel(
            self.last_depth,
            candidate["center"],
            (int(rgb_response.height), int(rgb_response.width)),
        )
        depth_source = "depth_map"
        if not np.isfinite(depth_m) or depth_m <= 1e-6:
            if gate_dimensions is None:
                gate_dimensions = self.get_active_gate_dimensions()
            gate_width_m, gate_height_m = gate_dimensions
            depth_candidates = []
            if gate_width_m > 1e-6 and rect_w_px > 1e-6:
                depth_candidates.append(intr["fx"] * gate_width_m / rect_w_px)
            if gate_height_m > 1e-6 and rect_h_px > 1e-6:
                depth_candidates.append(intr["fy"] * gate_height_m / rect_h_px)
            if not depth_candidates:
                return None, np.inf, "unavailable"
            depth_m = float(np.mean(depth_candidates))
            depth_source = "nominal_gate_size"

        x_off = (center_u - intr["cx"]) * depth_m / intr["fx"]
        y_off = (center_v - intr["cy"]) * depth_m / intr["fy"]
        target_rel_camera = np.array([depth_m, x_off, y_off], dtype=np.float32)

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
        p_target_airsim = camera_position + camera_rot @ target_rel_camera
        return p_target_airsim, depth_m, depth_source

    def estimate_corner_target_point_airsim(self):
        if self.last_rgb is None or self.last_rgb_response is None:
            if self.last_corner_target_airsim is not None:
                return self.last_corner_target_airsim, {
                    "segmentation_depth_source": "cached_primary",
                    "segmentation_target_cache_used": True,
                    "segmentation_target_cache_rank": 1,
                }
            if self.last_corner_backup_target_airsim is not None:
                return self.last_corner_backup_target_airsim, {
                    "segmentation_depth_source": "cached_secondary",
                    "segmentation_target_cache_used": True,
                    "segmentation_target_cache_rank": 2,
                }
            return None, {}

        heatmaps = self.corner_estimator.predict_heatmaps(self.last_rgb)
        gate_candidates = self._extract_gate_candidates(
            heatmaps,
            int(self.last_rgb_response.width),
            int(self.last_rgb_response.height),
        )
        self.last_corner_gate_candidates = gate_candidates
        self.last_corner_candidate_targets_airsim = []
        self._viz_corner_overlay(self.last_rgb, gate_candidates)
        if not gate_candidates:
            return None, {}

        intr = self.get_camera_intrinsics(
            int(self.last_rgb_response.width),
            int(self.last_rgb_response.height),
        )
        gate_dimensions = self.get_active_gate_dimensions()
        candidate_targets = []
        p_target_airsim = None
        depth_m = np.inf
        depth_source = "unavailable"
        candidate = None
        for gate_candidate in gate_candidates:
            target_airsim, target_depth_m, target_depth_source = self._gate_target_to_airsim(
                gate_candidate,
                intr,
                self.last_rgb_response,
                gate_dimensions=gate_dimensions,
            )
            if target_airsim is None:
                continue
            candidate_targets.append(
                {
                    "candidate": gate_candidate,
                    "target_airsim": target_airsim,
                    "depth_m": target_depth_m,
                    "depth_source": target_depth_source,
                }
            )
            if p_target_airsim is None:
                candidate = gate_candidate
                p_target_airsim = target_airsim
                depth_m = target_depth_m
                depth_source = target_depth_source

        self.last_corner_candidate_targets_airsim = [item["target_airsim"] for item in candidate_targets]
        if p_target_airsim is None or candidate is None:
            return None, {}

        backup_target_airsim = p_target_airsim
        self.last_corner_target_airsim = p_target_airsim
        self.last_corner_backup_target_airsim = backup_target_airsim
        self.last_corner_candidate_timestamp = time.time()

        aux = {
            "segmentation_mask": None,
            "segmentation_rect": candidate,
            "segmentation_primary_rect": candidate,
            "segmentation_backup_rect": candidate,
            "segmentation_center_px": candidate["center"],
            "segmentation_rect_size_px": candidate["size"],
            "segmentation_depth_m": depth_m,
            "segmentation_depth_source": depth_source,
            "segmentation_primary_depth_m": depth_m,
            "segmentation_primary_rank": 1,
            "segmentation_selected_depth_m": depth_m,
            "segmentation_selected_rank": 1,
            "segmentation_backup_depth_m": depth_m,
            "segmentation_backup_depth_source": depth_source,
            "segmentation_backup_rank": 1,
            "segmentation_blob_count": 1,
            "segmentation_blob_selection": "corner_heatmaps",
            "segmentation_blob_depth_m": depth_m,
            "segmentation_promoted": False,
            "segmentation_promote_depth_threshold": self.corner_conf_threshold,
            "segmentation_blob_backup_depth_m": depth_m,
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


def build_args():
    parser = ArgumentParser(description="Run the DiffSim racer using gate-corner detection.")
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
    parser.add_argument("--target_speed", type=float, default=7.0)
    parser.add_argument("--hover_throttle", type=float, default=0.9)
    parser.add_argument("--control_period", type=float, default=0.05)
    parser.add_argument("--image_period", type=float, default=0.05)
    parser.add_argument(
        "--model_path",
        type=str,
        default=DEFAULT_MODEL_PATH,
        help="Path to the learned control policy checkpoint.",
    )
    parser.add_argument(
        "--corner_model_path",
        type=str,
        default=str(DEFAULT_CORNER_CHECKPOINT),
        help="Path to the trained corner heatmap checkpoint.",
    )
    parser.add_argument(
        "--corner_conf_threshold",
        type=float,
        default=DEFAULT_CORNER_CONF_THRESHOLD,
        help="Minimum heatmap confidence for each detected corner.",
    )
    parser.add_argument(
        "--no_swap_rb",
        dest="swap_rb",
        action="store_false",
        default=True,
        help="Do not swap red/blue channels on AirSim Scene frames before inference.",
    )
    parser.add_argument(
        "--depth_onnx_path",
        "--depth_checkpoint",
        dest="depth_onnx_path",
        type=str,
        default=str(DEFAULT_DEPTH_ONNX_PATH),
        help="Path to the RGB-to-depth ONNX model.",
    )
    parser.add_argument(
        "--depth_input_width",
        type=int,
        default=DEFAULT_DEPTH_INPUT_WIDTH,
        help="Input width used for ONNX depth inference.",
    )
    parser.add_argument(
        "--depth_input_height",
        type=int,
        default=DEFAULT_DEPTH_INPUT_HEIGHT,
        help="Input height used for ONNX depth inference.",
    )
    parser.add_argument(
        "--depth_device",
        type=str,
        default=DEFAULT_DEPTH_DEVICE,
        help="Preferred depth backend device: auto, cpu, or cuda.",
    )
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
    parser.add_argument("--debug_print_every", type=int, default=10)
    parser.add_argument("--viz_rgb", dest="viz_rgb", action="store_true", default=False)
    parser.add_argument("--viz_depth", dest="viz_depth", action="store_true", default=False)
    parser.add_argument("--viz_depth_raw", dest="viz_depth_raw", action="store_true", default=False)
    parser.add_argument("--viz_corner_heatmaps", action="store_true", default=False)
    parser.add_argument(
        "--viz_corner_stride",
        type=int,
        default=1,
        help="Show corner overlay every N frames (reduces UI overhead).",
    )
    return parser.parse_args()


def main():
    args = build_args()

    racer = DiffsimGateDetectRacer(
        corner_model_path=args.corner_model_path,
        corner_conf_threshold=args.corner_conf_threshold,
        swap_rb=args.swap_rb,
        viz_corner_heatmaps=args.viz_corner_heatmaps,
        viz_corner_stride=args.viz_corner_stride,
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
