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
        viz_corner_heatmaps: bool = False,
        viz_corner_stride: int = 1,
        swap_rb: bool = False,
        depth_onnx_path=DEFAULT_DEPTH_ONNX_PATH,
        depth_input_width=DEFAULT_DEPTH_INPUT_WIDTH,
        depth_input_height=DEFAULT_DEPTH_INPUT_HEIGHT,
        depth_device=DEFAULT_DEPTH_DEVICE,
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

        self.viz_corner_heatmaps = bool(viz_corner_heatmaps)
        self.viz_corner_stride = max(1, int(viz_corner_stride))
        self._viz_corner_counter = 0
        self._warned_missing_cv2_heatmaps = False
        self._warned_corner_overlay_error = False
        self._gate_postproc_debug_counter = 0

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

        kwargs.setdefault("target_source", "corner_detection")
        super().__init__(**kwargs)

    def image_callback(self):
        # Keep the base depth/rgb update logic intact.
        super().image_callback()

        # Visualize detector output even when flying on ground-truth targets.
        # Also makes visualization independent of the control thread.
        if not self.viz_corner_heatmaps or cv2 is None:
            return
        if self.last_rgb is None or self.last_rgb_response is None:
            return

        self._viz_corner_counter += 1
        if (self._viz_corner_counter - 1) % self.viz_corner_stride != 0:
            return

        try:
            corner_maps, paf_maps = self.corner_estimator.predict_maps(self.last_rgb)
            gate_candidates = self._extract_gate_candidates(
                corner_maps,
                paf_maps,
                int(self.last_rgb_response.width),
                int(self.last_rgb_response.height),
                max_gates=5,
            )
            self.last_corner_gate_candidates = gate_candidates
            self._viz_corner_overlay(self.last_rgb, gate_candidates)
        except Exception as exc:
            if self.debug_print and not self._warned_corner_overlay_error:
                print(f"[corner_overlay] overlay inference failed: {exc}")
                self._warned_corner_overlay_error = True
            return

    def _viz_corner_overlay(self, rgb_image: np.ndarray, gate_candidates: list[dict]) -> None:
        # Keep visualization consistent with `gate_detection/test_corner_affinitiy_detector.py`.
        if not self.viz_corner_heatmaps:
            return
        if cv2 is None:
            if not self._warned_missing_cv2_heatmaps:
                print("[corner_overlay] OpenCV (cv2) not available; cannot visualize overlays.")
                self._warned_missing_cv2_heatmaps = True
            return
        if rgb_image is None or rgb_image.ndim != 3 or rgb_image.shape[2] != 3:
            return

        display_bgr = rgb_image.copy()

        palette = [
            (0, 0, 255),
            (0, 165, 255),
            (0, 255, 255),
            (0, 255, 0),
            (255, 0, 0),
        ]
        for gate_idx, gate in enumerate(gate_candidates[:5]):
            color = palette[gate_idx % len(palette)]
            pts = gate.get("points", {})
            for name, point in pts.items():
                x, y = float(point[0]), float(point[1])
                cv2.circle(display_bgr, (int(round(x)), int(round(y))), 4, color, -1)
                cv2.putText(
                    display_bgr,
                    name,
                    (int(round(x)) + 4, int(round(y)) - 4),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.4,
                    (255, 255, 255),
                    1,
                    cv2.LINE_AA,
                )
            for a, b in EDGE_TYPES:
                if a in pts and b in pts:
                    x0, y0 = float(pts[a][0]), float(pts[a][1])
                    x1, y1 = float(pts[b][0]), float(pts[b][1])
                    cv2.line(
                        display_bgr,
                        (int(round(x0)), int(round(y0))),
                        (int(round(x1)), int(round(y1))),
                        color,
                        1,
                    )

        cv2.putText(
            display_bgr,
            f"gates={len(gate_candidates)} thr={self.corner_conf_threshold:.2f} topk={self.corner_topk} nms={self.corner_nms_radius}",
            (8, 22),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        cv2.imshow("corner_affinity_detector_overlay", display_bgr)
        cv2.waitKey(1)

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
        x = float(pixel[0]) * float(depth_w) / float(src_w)
        y = float(pixel[1]) * float(depth_h) / float(src_h)
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

        corner_maps, paf_maps = self.corner_estimator.predict_maps(self.last_rgb)
        gate_candidates = self._extract_gate_candidates(
            corner_maps,
            paf_maps,
            int(self.last_rgb_response.width),
            int(self.last_rgb_response.height),
        )
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
                }
            if self.last_corner_backup_target_airsim is not None:
                return self.last_corner_backup_target_airsim, {
                    "segmentation_depth_source": "cached_secondary",
                    "segmentation_target_cache_used": True,
                    "segmentation_target_cache_rank": 2,
                }
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
            "segmentation_blob_selection": "corner_affinity",
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
        default=str(DEFAULT_AFFINITY_CHECKPOINT),
        help="Path to the trained corner+PAF checkpoint.",
    )
    parser.add_argument("--corner_conf_threshold", type=float, default=DEFAULT_CORNER_CONF_THRESHOLD)
    parser.add_argument("--corner_topk", type=int, default=DEFAULT_CORNER_TOPK)
    parser.add_argument("--corner_nms_radius", type=int, default=DEFAULT_CORNER_NMS_RADIUS)
    parser.add_argument("--edge_min_score", type=float, default=DEFAULT_EDGE_MIN_SCORE)
    parser.add_argument("--integral_samples", type=int, default=DEFAULT_INTEGRAL_SAMPLES)
    parser.add_argument("--swap_rb", action="store_true", default=True, help="Swap red/blue channels on AirSim Scene frames before inference (use if colors look wrong).",)
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
    parser.add_argument("--debug_print_every", type=int, default=10)
    parser.add_argument("--viz_rgb", dest="viz_rgb", action="store_true", default=False)
    parser.add_argument("--viz_depth", dest="viz_depth", action="store_true", default=False)
    parser.add_argument("--viz_depth_raw", dest="viz_depth_raw", action="store_true", default=False)
    parser.add_argument("--viz_corner_heatmaps", action="store_true", default=False)
    parser.add_argument("--viz_corner_stride", type=int, default=1)
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
        viz_corner_heatmaps=args.viz_corner_heatmaps,
        viz_corner_stride=args.viz_corner_stride,
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
