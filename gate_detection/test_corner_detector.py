#!/usr/bin/env python3

"""Visualize gate-corner detector outputs live in AirSim.

This script is meant for quick sanity-checking a trained corner-heatmap model
in the simulator. It uses `baselines/baseline_racer.py` to connect to AirSim
and poll RGB frames, then runs gate-corner inference and overlays the detected
corner points.
"""

from __future__ import annotations

from argparse import ArgumentParser
from pathlib import Path
import sys
import threading
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
DEFAULT_CHECKPOINT = GATE_DETECTION_DIR / "corner_checkpoints" / "best.pt"

if str(BASELINES_DIR) not in sys.path:
    sys.path.insert(0, str(BASELINES_DIR))
if str(GATE_DETECTION_DIR) not in sys.path:
    sys.path.insert(0, str(GATE_DETECTION_DIR))

from baseline_racer import BaselineRacer  # noqa: E402
from corner_unet import CornerUNet  # noqa: E402


CORNER_NAMES = ("TL", "TR", "BL", "BR")


def _load_corner_state_dict(checkpoint_path: Path, device: torch.device) -> dict:
    checkpoint = torch.load(str(checkpoint_path), map_location=device)
    if isinstance(checkpoint, dict) and "model" in checkpoint:
        return checkpoint["model"]
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        return checkpoint["state_dict"]
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        return checkpoint["model_state_dict"]
    return checkpoint


class CornerHeatmapEstimator:
    def __init__(self, checkpoint_path: str | Path, device: str):
        if torch is None:  # pragma: no cover
            raise ImportError(
                "torch is required to run the corner detector. Install PyTorch first."
            ) from TORCH_IMPORT_ERROR

        checkpoint_path = Path(checkpoint_path).expanduser().resolve()
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Corner checkpoint does not exist: {checkpoint_path}")

        self.device = torch.device(device)
        self.model = CornerUNet().to(self.device)
        state_dict = _load_corner_state_dict(checkpoint_path, self.device)
        self.model.load_state_dict(state_dict, strict=False)
        self.model.eval()

    def _preprocess(self, image_rgb: np.ndarray) -> tuple[torch.Tensor, tuple[int, int]]:
        if image_rgb.ndim != 3 or image_rgb.shape[2] != 3:
            raise ValueError(f"Expected RGB image HxWx3, got {image_rgb.shape}")
        height, width = int(image_rgb.shape[0]), int(image_rgb.shape[1])
        image = image_rgb.astype(np.float32) / 255.0
        tensor = torch.from_numpy(np.transpose(image, (2, 0, 1))[None, ...]).contiguous().to(self.device)

        # CornerUNet downsamples 4x -> stride 16. Pad to avoid size mismatch on odd sizes.
        stride = 16
        pad_h = (stride - (height % stride)) % stride
        pad_w = (stride - (width % stride)) % stride
        if pad_h or pad_w:
            tensor = torch.nn.functional.pad(tensor, (0, pad_w, 0, pad_h), mode="constant", value=0.0)

        return tensor, (height, width)

    @torch.no_grad()
    def predict_heatmaps(self, image_rgb: np.ndarray) -> np.ndarray:
        tensor, unpad_hw = self._preprocess(image_rgb)
        logits = self.model(tensor)
        heatmaps = torch.sigmoid(logits)[0].detach().float().cpu().numpy()
        out_h, out_w = unpad_hw
        return heatmaps[:, :out_h, :out_w].astype(np.float32, copy=False)


def extract_corner_predictions(
    heatmaps: np.ndarray,
    *,
    threshold: float,
    topk: int,
    nms_radius: int,
) -> list[dict]:
    if torch is None:  # pragma: no cover
        raise ImportError("torch is required for NMS-based peak extraction.") from TORCH_IMPORT_ERROR
    if heatmaps.ndim != 3 or heatmaps.shape[0] != 4:
        return []

    threshold = float(threshold)
    topk = int(topk)
    nms_radius = max(0, int(nms_radius))
    if topk <= 0:
        return []

    preds: list[dict] = []
    heatmaps_t = torch.from_numpy(heatmaps).float()
    for idx, corner_name in enumerate(CORNER_NAMES):
        hmap = heatmaps_t[idx]
        if nms_radius > 0:
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

    preds.sort(key=lambda item: float(item.get("score", 0.0)), reverse=True)
    return preds


class CornerDetectorVizRacer(BaselineRacer):
    def __init__(
        self,
        *,
        checkpoint_path: str | Path,
        device: str,
        conf_threshold: float,
        topk: int,
        nms_radius: int,
        swap_rb: bool,
        viz_stride: int,
        **kwargs,
    ):
        super().__init__(viz_image_cv2=False, **kwargs)
        self.estimator = CornerHeatmapEstimator(checkpoint_path=checkpoint_path, device=device)
        self.conf_threshold = float(conf_threshold)
        self.topk = int(topk)
        self.nms_radius = int(nms_radius)
        self.swap_rb = bool(swap_rb)
        self.viz_stride = max(1, int(viz_stride))
        self._viz_counter = 0
        self._last_state_text = ""

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

        # AirSim frames are usually RGB, but allow swapping in case the environment uses BGR.
        image_rgb = raw[..., ::-1] if self.swap_rb else raw
        display_bgr = image_rgb[..., ::-1].copy()

        self._viz_counter += 1
        if (self._viz_counter - 1) % self.viz_stride != 0:
            return

        heatmaps = self.estimator.predict_heatmaps(image_rgb)
        preds = extract_corner_predictions(
            heatmaps,
            threshold=self.conf_threshold,
            topk=self.topk,
            nms_radius=self.nms_radius,
        )

        for pred in preds:
            x = int(round(float(pred["x"])))
            y = int(round(float(pred["y"])))
            score = float(pred["score"])
            label = str(pred["label"])
            cv2.circle(display_bgr, (x, y), 4, (0, 0, 255), -1)
            cv2.putText(
                display_bgr,
                f"{label}:{score:.2f}",
                (x + 5, y - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )

        try:
            state = self.airsim_client.getMultirotorState(vehicle_name=self.drone_name)
            z = float(state.kinematics_estimated.position.z_val)
            landed = int(getattr(state, "landed_state", -1))
            self._last_state_text = f"z={z:.2f} landed={landed}"
        except Exception:
            # State queries can fail transiently depending on simulator timing.
            pass

        cv2.putText(
            display_bgr,
            f"peaks={len(preds)} thr={self.conf_threshold:.2f} topk={self.topk} nms={self.nms_radius}",
            (8, 22),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        if self._last_state_text:
            cv2.putText(
                display_bgr,
                self._last_state_text,
                (8, 44),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
        cv2.imshow("corner_detector_overlay", display_bgr)
        key = int(cv2.waitKey(1) & 0xFF)
        if key in (ord("q"), 27):  # q or ESC
            self.is_image_thread_active = False


def fly_through_gates(
    racer: BaselineRacer,
    *,
    pattern: str,
    control_api: str,
) -> None:
    racer.get_ground_truth_gate_poses()
    if not racer.gate_poses_ground_truth:
        raise RuntimeError("No gates found (gate_poses_ground_truth is empty).")

    if pattern == "one_by_one":
        for gate_pose in racer.gate_poses_ground_truth:
            if control_api == "moveOnSplineVelConstraints":
                task = racer.airsim_client.moveOnSplineVelConstraintsAsync(
                    [gate_pose.position],
                    [racer.get_gate_facing_vector_from_quaternion(gate_pose.orientation, scale=2.5)],
                    vel_max=15.0,
                    acc_max=3.0,
                    add_position_constraint=True,
                    add_velocity_constraint=True,
                    add_acceleration_constraint=False,
                    viz_traj=racer.viz_traj,
                    viz_traj_color_rgba=racer.viz_traj_color_rgba,
                    vehicle_name=racer.drone_name,
                )
            else:
                task = racer.airsim_client.moveOnSplineAsync(
                    [gate_pose.position],
                    vel_max=10.0,
                    acc_max=5.0,
                    add_position_constraint=True,
                    add_velocity_constraint=False,
                    add_acceleration_constraint=False,
                    viz_traj=racer.viz_traj,
                    viz_traj_color_rgba=racer.viz_traj_color_rgba,
                    vehicle_name=racer.drone_name,
                )
            task.join()
        return

    if control_api == "moveOnSplineVelConstraints":
        racer.airsim_client.moveOnSplineVelConstraintsAsync(
            [gate_pose.position for gate_pose in racer.gate_poses_ground_truth],
            [
                racer.get_gate_facing_vector_from_quaternion(gate_pose.orientation, scale=2.5)
                for gate_pose in racer.gate_poses_ground_truth
            ],
            vel_max=15.0,
            acc_max=7.5,
            add_position_constraint=True,
            add_velocity_constraint=True,
            add_acceleration_constraint=False,
            viz_traj=racer.viz_traj,
            viz_traj_color_rgba=racer.viz_traj_color_rgba,
            vehicle_name=racer.drone_name,
        ).join()
        return

    racer.airsim_client.moveOnSplineAsync(
        [gate_pose.position for gate_pose in racer.gate_poses_ground_truth],
        vel_max=30.0,
        acc_max=15.0,
        add_position_constraint=True,
        add_velocity_constraint=False,
        add_acceleration_constraint=False,
        viz_traj=racer.viz_traj,
        viz_traj_color_rgba=racer.viz_traj_color_rgba,
        vehicle_name=racer.drone_name,
    ).join()


def safe_takeoff(racer: BaselineRacer, *, takeoff_height_m: float, method: str) -> None:
    """Take off to a target height (best-effort across AirSim variants)."""
    target_height_m = float(takeoff_height_m)
    if target_height_m <= 0:
        return

    try:
        start_pose = racer.airsim_client.simGetVehiclePose(vehicle_name=racer.drone_name)
        start_z = float(start_pose.position.z_val)
    except Exception:
        start_z = float("nan")

    target_z = start_z - target_height_m if np.isfinite(start_z) else -target_height_m
    print(f"[takeoff] method={method} start_z={start_z:.2f} target_z={target_z:.2f}")

    def _fallback():
        try:
            racer.airsim_client.takeoffAsync(vehicle_name=racer.drone_name).join()
            print("[takeoff] takeoffAsync complete")
        except TypeError:
            racer.airsim_client.takeoffAsync().join()
            print("[takeoff] takeoffAsync complete (no vehicle_name)")

        # Ensure we reach (approximately) the desired altitude.
        try:
            racer.airsim_client.moveToZAsync(target_z, 1.0, vehicle_name=racer.drone_name).join()
            print("[takeoff] moveToZAsync complete")
            return
        except Exception:
            pass

        try:
            racer.airsim_client.moveByVelocityZAsync(
                0.0, 0.0, target_z, 2.0, vehicle_name=racer.drone_name
            ).join()
            print("[takeoff] moveByVelocityZAsync complete")
        except Exception as exc:
            print(f"[takeoff] fallback move commands failed: {exc}")

    if method == "async":
        _fallback()
        return

    if method == "spline":
        try:
            racer.takeoff_with_moveOnSpline(takeoff_height=target_height_m)
        except Exception as exc:
            print(f"[takeoff] moveOnSpline takeoff failed: {exc}")
            _fallback()
        return

    raise ValueError(f"Unknown takeoff method: {method}")


def build_args():
    parser = ArgumentParser(description="Run corner detector inference in AirSim and visualize results.")
    parser.add_argument("--level_name", type=str, default="Soccer_Field_Easy")
    parser.add_argument("--race_tier", type=int, choices=[1, 2, 3], default=1)
    parser.add_argument("--drone_name", type=str, default="drone_1")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=str(DEFAULT_CHECKPOINT),
        help="Path to the trained corner heatmap checkpoint.",
    )
    parser.add_argument("--device", type=str, default="cuda" if (torch and torch.cuda.is_available()) else "cpu")
    parser.add_argument("--threshold", type=float, default=0.8, help="Min heatmap confidence for peaks.")
    parser.add_argument("--topk", type=int, default=50, help="Max peaks per corner channel.")
    parser.add_argument("--nms_radius", type=int, default=5, help="NMS radius in heatmap pixels.")
    parser.add_argument(
        "--swap_rb",
        action="store_true",
        default=False,
        help="Swap red/blue channels before inference (use if colors look wrong).",
    )
    parser.add_argument(
        "--viz_stride",
        type=int,
        default=1,
        help="Only run visualization every N frames (reduces overhead).",
    )
    parser.add_argument(
        "--no_takeoff",
        action="store_true",
        default=False,
        help="Skip takeoff and just visualize from the current drone position.",
    )
    parser.add_argument(
        "--takeoff_method",
        type=str,
        choices=["spline", "async"],
        default="spline",
        help="Takeoff strategy: spline (default) or async+moveToZ fallback.",
    )
    parser.add_argument(
        "--takeoff_height",
        type=float,
        default=1.0,
        help="Takeoff height in meters (only used when not --no_takeoff).",
    )
    parser.add_argument(
        "--fly_through_gates",
        action="store_true",
        default=False,
        help="Also command the BaselineRacer to fly through the ground-truth gates while visualizing.",
    )
    parser.add_argument(
        "--flight_pattern",
        type=str,
        choices=["all_at_once", "one_by_one"],
        default="all_at_once",
        help="Gate flight pattern used when --fly_through_gates is set.",
    )
    parser.add_argument(
        "--control_api",
        type=str,
        choices=["moveOnSpline", "moveOnSplineVelConstraints"],
        default="moveOnSpline",
        help="AirSim planning/control API used when --fly_through_gates is set.",
    )
    parser.add_argument(
        "--duration_sec",
        type=float,
        default=0.0,
        help="If >0, stop after this many seconds (otherwise run until 'q'/'esc').",
    )
    return parser.parse_args()


def main():
    args = build_args()
    if cv2 is None:
        raise ImportError("cv2 is required for visualization (pip install opencv-python).")
    if torch is None:
        raise ImportError("torch is required for corner inference.") from TORCH_IMPORT_ERROR

    racer = CornerDetectorVizRacer(
        drone_name=args.drone_name,
        viz_traj=False,
        checkpoint_path=args.checkpoint,
        device=args.device,
        conf_threshold=args.threshold,
        topk=args.topk,
        nms_radius=args.nms_radius,
        swap_rb=args.swap_rb,
        viz_stride=args.viz_stride,
    )
    racer.load_level(args.level_name)
    racer.start_race(args.race_tier)
    racer.initialize_drone()
    if not args.no_takeoff:
        safe_takeoff(
            racer,
            takeoff_height_m=float(args.takeoff_height),
            method=str(args.takeoff_method),
        )

    flight_thread = None
    if args.fly_through_gates:
        def _fly():
            try:
                print(f"[flight] starting pattern={args.flight_pattern} api={args.control_api}")
                fly_through_gates(
                    racer,
                    pattern=str(args.flight_pattern),
                    control_api=str(args.control_api),
                )
                print("[flight] completed")
            except Exception as exc:  # pragma: no cover
                print(f"[flight] stopped due to error: {exc}")

        flight_thread = threading.Thread(target=_fly, daemon=True)
        flight_thread.start()
    else:
        print("[flight] not started (pass --fly_through_gates to command motion)")

    racer.start_image_callback_thread()
    try:
        if args.duration_sec and args.duration_sec > 0:
            deadline = time.time() + float(args.duration_sec)
            while time.time() < deadline and racer.is_image_thread_active:
                time.sleep(0.1)
        else:
            while racer.is_image_thread_active:
                time.sleep(0.1)
    finally:
        racer.stop_image_callback_thread()
        cv2.destroyAllWindows()
        racer.reset_race()


if __name__ == "__main__":
    main()
