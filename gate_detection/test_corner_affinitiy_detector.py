#!/usr/bin/env python3

"""Visualize corner+PAF (affinity) gate detector outputs live in AirSim.

This is the "paper-style" gate detector:
- 4 corner heatmaps (TL/TR/BL/BR)
- 4 PAFs (edges) => 8 channels (vx, vy per edge)

It runs inference with the same post-processing implemented in
`gate_detection/train_corner_affinity_detection.py`:
corner peak extraction -> PAF line-integral scoring -> edge matching -> gate assembly.
"""

from __future__ import annotations

from argparse import ArgumentParser
from pathlib import Path
import sys
import time

import numpy as np

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

import airsimdroneracinglab as airsim


REPO_ROOT = Path(__file__).resolve().parents[1]
BASELINES_DIR = REPO_ROOT / "baselines"
GATE_DETECTION_DIR = REPO_ROOT / "gate_detection"
DEFAULT_CHECKPOINT = GATE_DETECTION_DIR / "corner_affinity_checkpoints" / "best.pt"

if str(BASELINES_DIR) not in sys.path:
    sys.path.insert(0, str(BASELINES_DIR))
if str(GATE_DETECTION_DIR) not in sys.path:
    sys.path.insert(0, str(GATE_DETECTION_DIR))

from baseline_racer import BaselineRacer  # noqa: E402
from corner_unet import CornerUNet  # noqa: E402
from train_corner_affinity_detection import (  # noqa: E402
    CORNER_NAMES,
    EDGE_TYPES,
    assemble_gates_from_edges,
    extract_corner_candidates,
    score_and_match_edges,
)


class CornerAffinityEstimator:
    def __init__(self, checkpoint_path: str | Path, device: str):
        if torch is None:  # pragma: no cover
            raise ImportError("torch is required for inference.") from TORCH_IMPORT_ERROR

        checkpoint_path = Path(checkpoint_path).expanduser().resolve()
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint does not exist: {checkpoint_path}")

        self.device = torch.device(device)
        self.model = CornerUNet(out_channels=12).to(self.device)

        checkpoint = torch.load(str(checkpoint_path), map_location=self.device)
        if isinstance(checkpoint, dict) and "model" in checkpoint:
            state_dict = checkpoint["model"]
        elif isinstance(checkpoint, dict) and "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        else:
            state_dict = checkpoint
        self.model.load_state_dict(state_dict, strict=False)
        self.model.eval()

    def _preprocess(self, image_rgb: np.ndarray) -> tuple[torch.Tensor, tuple[int, int]]:
        if image_rgb.ndim != 3 or image_rgb.shape[2] != 3:
            raise ValueError(f"Expected RGB image HxWx3, got {image_rgb.shape}")
        height, width = int(image_rgb.shape[0]), int(image_rgb.shape[1])
        image = image_rgb.astype(np.float32) / 255.0
        tensor = torch.from_numpy(np.transpose(image, (2, 0, 1))[None, ...]).contiguous().to(self.device)

        # CornerUNet downsamples 4x -> stride 16. Pad to avoid odd-size mismatch.
        stride = 16
        pad_h = (stride - (height % stride)) % stride
        pad_w = (stride - (width % stride)) % stride
        if pad_h or pad_w:
            tensor = torch.nn.functional.pad(tensor, (0, pad_w, 0, pad_h), mode="constant", value=0.0)

        return tensor, (height, width)

    @torch.no_grad()
    def predict(self, image_rgb: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        tensor, unpad_hw = self._preprocess(image_rgb)
        logits = self.model(tensor)
        corner_maps = torch.sigmoid(logits[:, :4])[0].detach().float().cpu().numpy().astype(np.float32)
        paf_maps = torch.tanh(logits[:, 4:])[0].detach().float().cpu().numpy().astype(np.float32)
        out_h, out_w = unpad_hw
        return corner_maps[:, :out_h, :out_w], paf_maps[:, :out_h, :out_w]


class CornerAffinityVizRacer(BaselineRacer):
    def __init__(
        self,
        *,
        checkpoint_path: str | Path,
        device: str,
        corner_threshold: float,
        corner_topk: int,
        corner_nms_radius: int,
        edge_min_score: float,
        integral_samples: int,
        swap_rb: bool,
        viz_stride: int,
        **kwargs,
    ):
        super().__init__(viz_image_cv2=False, **kwargs)
        self.estimator = CornerAffinityEstimator(checkpoint_path=checkpoint_path, device=device)
        self.corner_threshold = float(corner_threshold)
        self.corner_topk = int(corner_topk)
        self.corner_nms_radius = int(corner_nms_radius)
        self.edge_min_score = float(edge_min_score)
        self.integral_samples = int(integral_samples)
        self.swap_rb = bool(swap_rb)
        self.viz_stride = max(1, int(viz_stride))
        self._viz_counter = 0

    def image_callback(self):
        if cv2 is None:  # pragma: no cover
            return

        request = [airsim.ImageRequest("fpv_cam", airsim.ImageType.Scene, False, False)]
        responses = self.airsim_client_images.simGetImages(request, vehicle_name=self.drone_name)
        if not responses:
            return
        response = responses[0]
        if response.width <= 0 or response.height <= 0:
            return

        raw = np.frombuffer(response.image_data_uint8, dtype=np.uint8)
        if raw.size != int(response.width) * int(response.height) * 3:
            return
        raw = raw.reshape(int(response.height), int(response.width), 3)

        image_rgb = raw[..., ::-1] if self.swap_rb else raw
        display_bgr = image_rgb.copy()

        self._viz_counter += 1
        if (self._viz_counter - 1) % self.viz_stride != 0:
            return

        corner_maps, paf_maps = self.estimator.predict(image_rgb)
        candidates = extract_corner_candidates(
            corner_maps,
            threshold=self.corner_threshold,
            topk=self.corner_topk,
            nms_radius=self.corner_nms_radius,
        )
        edge_matches = score_and_match_edges(
            candidates,
            paf_maps,
            edge_min_score=self.edge_min_score,
            integral_samples=self.integral_samples,
        )
        gates = assemble_gates_from_edges(edge_matches)

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
                    x0, y0 = pts[a]
                    x1, y1 = pts[b]
                    cv2.line(
                        display_bgr,
                        (int(round(x0)), int(round(y0))),
                        (int(round(x1)), int(round(y1))),
                        color,
                        1,
                    )

        cv2.putText(
            display_bgr,
            f"gates={len(gates)} thr={self.corner_threshold:.2f} topk={self.corner_topk} nms={self.corner_nms_radius}",
            (8, 22),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        cv2.imshow("corner_affinity_detector_overlay", display_bgr)
        key = int(cv2.waitKey(1) & 0xFF)
        if key in (ord("q"), 27):  # q or ESC
            self.is_image_thread_active = False


def build_args():
    parser = ArgumentParser(description="Run corner+PAF detector inference in AirSim and visualize results.")
    parser.add_argument("--level_name", type=str, default="Soccer_Field_Easy")
    parser.add_argument("--race_tier", type=int, choices=[1, 2, 3], default=1)
    parser.add_argument("--drone_name", type=str, default="drone_1")
    parser.add_argument("--checkpoint", type=str, default=str(DEFAULT_CHECKPOINT))
    parser.add_argument("--device", type=str, default="cuda" if (torch and torch.cuda.is_available()) else "cpu")
    parser.add_argument("--corner_threshold", type=float, default=0.25)
    parser.add_argument("--corner_topk", type=int, default=50)
    parser.add_argument("--corner_nms_radius", type=int, default=5)
    parser.add_argument("--edge_min_score", type=float, default=0.05)
    parser.add_argument("--integral_samples", type=int, default=10)
    parser.add_argument("--swap_rb", action="store_true", default=False)
    parser.add_argument("--viz_stride", type=int, default=1)
    parser.add_argument("--no_takeoff", action="store_true", default=False)
    parser.add_argument("--takeoff_height", type=float, default=1.0)
    parser.add_argument(
        "--fly_baseline",
        action="store_true",
        default=False,
        help="Fly the standard BaselineRacer spline through gates while visualizing detector output.",
    )
    parser.add_argument(
        "--planning_baseline_type",
        type=str,
        choices=["all_gates_at_once", "all_gates_one_by_one"],
        default="all_gates_at_once",
        help="Which gate sequence to fly when --fly_baseline is set.",
    )
    parser.add_argument(
        "--planning_and_control_api",
        type=str,
        choices=["moveOnSpline", "moveOnSplineVelConstraints"],
        default="moveOnSpline",
        help="Which AirSim API to use when --fly_baseline is set.",
    )
    parser.add_argument("--duration_sec", type=float, default=0.0)
    return parser.parse_args()


def main():
    args = build_args()
    if cv2 is None:
        raise ImportError("cv2 is required for visualization.")
    if torch is None:
        raise ImportError("torch is required for inference.") from TORCH_IMPORT_ERROR

    # Match BaselineRacer behavior for qualifier tiers (tier is tied to the level).
    if args.level_name == "Qualifier_Tier_1":
        args.race_tier = 1
    elif args.level_name == "Qualifier_Tier_2":
        args.race_tier = 2
    elif args.level_name == "Qualifier_Tier_3":
        args.race_tier = 3

    racer = CornerAffinityVizRacer(
        drone_name=args.drone_name,
        viz_traj=False,
        checkpoint_path=args.checkpoint,
        device=args.device,
        corner_threshold=args.corner_threshold,
        corner_topk=args.corner_topk,
        corner_nms_radius=args.corner_nms_radius,
        edge_min_score=args.edge_min_score,
        integral_samples=args.integral_samples,
        swap_rb=args.swap_rb,
        viz_stride=args.viz_stride,
    )
    racer.load_level(args.level_name)
    racer.start_race(args.race_tier)
    racer.initialize_drone()
    if not args.no_takeoff:
        racer.takeoff_with_moveOnSpline(takeoff_height=float(args.takeoff_height))

    racer.start_image_callback_thread()
    flight_future = None
    if args.fly_baseline:
        racer.get_ground_truth_gate_poses()
        if args.planning_baseline_type == "all_gates_at_once":
            if args.planning_and_control_api == "moveOnSpline":
                flight_future = racer.fly_through_all_gates_at_once_with_moveOnSpline()
            else:
                flight_future = racer.fly_through_all_gates_at_once_with_moveOnSplineVelConstraints()
        else:
            if args.planning_and_control_api == "moveOnSpline":
                flight_future = racer.fly_through_all_gates_one_by_one_with_moveOnSpline()
            else:
                flight_future = racer.fly_through_all_gates_one_by_one_with_moveOnSplineVelConstraints()
    try:
        if args.duration_sec and args.duration_sec > 0:
            deadline = time.time() + float(args.duration_sec)
            while time.time() < deadline and racer.is_image_thread_active:
                time.sleep(0.1)
        else:
            if flight_future is not None:
                flight_future.join()
            else:
                while racer.is_image_thread_active:
                    time.sleep(0.1)
    finally:
        
        racer.stop_image_callback_thread()
        cv2.destroyAllWindows()
        racer.reset_race()


if __name__ == "__main__":
    main()
