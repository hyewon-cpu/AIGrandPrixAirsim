#!/usr/bin/env python3

"""Collect RGB frames with BaselineRacer and auto-annotate visible gate corners.

This script flies the drone using the repo's BaselineRacer flow, captures raw RGB
frames from the FPV camera, projects the 4 gate corners of every visible gate
into the image plane, and writes paired training samples:

- `rgb/image_000001.png` raw input image
- `annotations/image_000001.json` LabelMe corner annotations for the same frame
- `pairs.txt` one line per sample with image and annotation paths

The output is designed to work with `gate_detection/train_corner_detector.py`.
"""

from __future__ import annotations

from argparse import ArgumentParser
from datetime import datetime
from pathlib import Path
import json
import math
import os
import re
import sys
import threading
import time

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
BASELINES_DIR = REPO_ROOT / "baselines"
AIRSIM_PY_DIR = REPO_ROOT / "airsimdroneracinglab-1.0.2"
if str(BASELINES_DIR) not in sys.path:
    sys.path.insert(0, str(BASELINES_DIR))
if str(AIRSIM_PY_DIR) not in sys.path:
    sys.path.insert(0, str(AIRSIM_PY_DIR))

import airsimdroneracinglab as airsim
from baseline_racer import BaselineRacer


def sanitize_name(value: str) -> str:
    cleaned = []
    for char in str(value):
        if char.isalnum() or char in ("-", "_"):
            cleaned.append(char)
        else:
            cleaned.append("_")
    return "".join(cleaned).strip("_") or "run"


def quaternion_to_rotation_matrix(q: np.ndarray) -> np.ndarray:
    """Convert [w, x, y, z] quaternion into a 3x3 rotation matrix."""
    w, x, y, z = map(float, q)
    n = w * w + x * x + y * y + z * z
    if n < np.finfo(float).eps:
        return np.eye(3, dtype=np.float32)
    s = 2.0 / n
    xx, yy, zz = x * x * s, y * y * s, z * z * s
    xy, xz, yz = x * y * s, x * z * s, y * z * s
    wx, wy, wz = w * x * s, w * y * s, w * z * s
    return np.array(
        [
            [1.0 - (yy + zz), xy - wz, xz + wy],
            [xy + wz, 1.0 - (xx + zz), yz - wx],
            [xz - wy, yz + wx, 1.0 - (xx + yy)],
        ],
        dtype=np.float32,
    )


def compute_pinhole_intrinsics(width: int, height: int, fov_degrees: float) -> dict[str, float]:
    fov_radians = math.radians(float(fov_degrees))
    fx = width / (2.0 * math.tan(fov_radians / 2.0))
    fy = fx
    cx = width / 2.0
    cy = height / 2.0
    return {"fx": fx, "fy": fy, "cx": cx, "cy": cy, "width": width, "height": height}


def project_world_point(
    point_world: np.ndarray,
    camera_position: np.ndarray,
    camera_rotation: np.ndarray,
    intr: dict[str, float],
) -> tuple[float, float] | None:
    """Project a world-space point to pixel coordinates."""
    point_cam = camera_rotation.T @ (point_world - camera_position)
    x, y, z = map(float, point_cam)
    if x <= 1e-6:
        return None
    u = intr["cx"] + intr["fx"] * (y / x)
    v = intr["cy"] + intr["fy"] * (z / x)
    return float(u), float(v)


def gate_corners_world(
    gate_position: np.ndarray,
    gate_rotation: np.ndarray,
    gate_width: float,
    gate_height: float,
) -> dict[str, np.ndarray]:
    """Return the 4 gate corners in world coordinates."""
    half_w = gate_width * 0.5
    half_h = gate_height * 0.5
    local_corners = {
        "TL": np.array([-half_w, 0.0, -half_h], dtype=np.float32),
        "TR": np.array([half_w, 0.0, -half_h], dtype=np.float32),
        "BL": np.array([-half_w, 0.0, half_h], dtype=np.float32),
        "BR": np.array([half_w, 0.0, half_h], dtype=np.float32),
    }
    return {
        name: gate_position + gate_rotation @ corner
        for name, corner in local_corners.items()
    }


def build_labelme_record(image_path: str, width: int, height: int, shapes: list[dict]) -> dict:
    return {
        "version": "5.4.1",
        "flags": {},
        "shapes": shapes,
        "imagePath": image_path,
        "imageData": None,
        "imageHeight": int(height),
        "imageWidth": int(width),
    }


MIN_VISIBLE_DEPTH_SAMPLES = 3
DEPTH_OCCLUSION_MARGIN_M = 0.35
GATE_NAME_RE = re.compile(r"^Gate(\d+)(?:_.+)?$")


def infer_race_tier(level_name: str, fallback: int = 1) -> int:
    match = re.search(r"_Tier_(\d+)$", str(level_name))
    if not match:
        return int(fallback)
    tier = int(match.group(1))
    return tier if tier in (1, 2, 3) else int(fallback)





def discover_gate_names(client) -> list[str]:
    gate_names = [name for name in client.simListSceneObjects("Gate.*") if GATE_NAME_RE.match(name)]

    def sort_key(name: str) -> tuple[int, str]:
        match = GATE_NAME_RE.match(name)
        gate_idx = int(match.group(1)) if match else 10**9
        return gate_idx, name

    return sorted(gate_names, key=sort_key)


def build_args():
    parser = ArgumentParser(description="Fly with BaselineRacer and annotate gate corners.")
    parser.add_argument("--output_dir", type=str, default=str(REPO_ROOT / "gate_detection" / "datasets"), help="Root directory for the generated training set.")
    parser.add_argument("--run_name", type=str, default="", help="Optional subfolder name for the annotation run.")
    parser.add_argument("--level_name", type=str, choices=[
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
        default="Final_Tier_1", help="AirSim level to load before capture.")
    parser.add_argument("--race_tier", type=int, default=1, help="Race tier to start.")
    parser.add_argument("--drone_name", type=str, default="drone_1")
    parser.add_argument("--camera_name", type=str, default="fpv_cam")
    parser.add_argument("--num_samples", type=int, default=7000)
    parser.add_argument("--image_period", type=float, default=0.05, help="Target capture period in seconds.")
    parser.add_argument("--sleep_sec", type=float, default=2.0)
    parser.add_argument(
        "--corner_mode",
        type=str,
        choices=["inner", "outer"],
        default="inner",
        help="Choose gate geometry. Auto picks inner for qualification scenes and outer otherwise.",
    )
    parser.add_argument("--save_overlay", action="store_true", default=False)
    parser.add_argument("--show_preview", action="store_true", default=False)
    parser.add_argument(
        "--flight_mode",
        type=str,
        choices=["all_gates_at_once", "all_gates_one_by_one"],
        default="all_gates_at_once",
        help="Which BaselineRacer flight routine to use.",
    )
    parser.add_argument(
        "--max_flight_loops",
        type=int,
        default=100,
        help="How many times to repeat the race if num_samples is not reached.",
    )
    return parser.parse_args()


class GateCornerCollector(BaselineRacer):
    def __init__(
        self,
        output_dir: Path,
        run_name: str,
        num_samples: int,
        camera_name: str,
        corner_mode: str,
        image_period: float,
        save_overlay: bool,
        show_preview: bool,
        drone_name: str = "drone_1",
    ):
        super().__init__(drone_name=drone_name, viz_traj=False, viz_traj_color_rgba=[1.0, 0.0, 0.0, 1.0], viz_image_cv2=False)
        self.output_dir = Path(output_dir).expanduser().resolve()
        self.run_name = sanitize_name(run_name) if run_name else datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        self.num_samples = int(num_samples)
        self.camera_name = camera_name
        self.corner_mode = corner_mode
        self.image_period = float(image_period)
        self.save_overlay = bool(save_overlay)
        self.show_preview = bool(show_preview)

        self.run_dir = self.output_dir / self.run_name
        self.rgb_dir = self.run_dir / "rgb"
        self.ann_dir = self.run_dir / "annotations"
        self.overlay_dir = self.run_dir / "overlays"
        self.pairs_path = self.run_dir / "pairs.txt"

        self.sample_index = 0
        self.samples_saved = 0
        self.capture_active = False
        self.capture_lock = threading.Lock()
        self.pairs_fh = None
        self.gate_records: list[dict] = []
        self.gate_names_ground_truth: list[str] = []

        self.camera_fov_degrees = 90.0
        self.image_callback_thread = None
        self._recreate_image_callback_thread()

    def _recreate_image_callback_thread(self):
        """Threads cannot be started twice, so create a fresh one on demand."""
        self.image_callback_thread = threading.Thread(
            target=self.repeat_timer_image_callback,
            args=(self.image_callback, self.image_period),
            daemon=True,
        )

    def prepare_run_directory(self):
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.run_dir.mkdir(parents=True, exist_ok=False)
        self.rgb_dir.mkdir()
        self.ann_dir.mkdir()
        if self.save_overlay:
            self.overlay_dir.mkdir()
        self.pairs_fh = self.pairs_path.open("w", encoding="utf-8")
        print(f"Saving collected data to: {self.run_dir}")

    def start_collection(self):
        if self.image_callback_thread is None or not self.image_callback_thread.is_alive():
            self._recreate_image_callback_thread()
        self.capture_active = True
        self.start_image_callback_thread()

    def stop_collection(self):
        self.capture_active = False
        self.stop_image_callback_thread()
        if self.pairs_fh is not None:
            self.pairs_fh.flush()
            self.pairs_fh.close()
            self.pairs_fh = None

    def _gate_dimensions(self, gate_name: str) -> tuple[float, float]:
        gate_scale = self.airsim_client.simGetObjectScale(gate_name)
        if self.corner_mode == "inner":
            gate_dims = self.airsim_client.simGetNominalGateInnerDimensions()
        else:
            gate_dims = self.airsim_client.simGetNominalGateOuterDimensions()
        gate_width = float(gate_dims.x_val) * float(gate_scale.x_val)
        gate_height = float(gate_dims.z_val) * float(gate_scale.z_val)
        return gate_width, gate_height

    def get_ground_truth_gate_poses(self):
        super().get_ground_truth_gate_poses()
        self.gate_names_ground_truth = discover_gate_names(self.airsim_client)
        if (
            self.gate_poses_ground_truth
            and len(self.gate_names_ground_truth) != len(self.gate_poses_ground_truth)
        ):
            print(
                "WARNING: detected "
                f"{len(self.gate_names_ground_truth)} gate objects but "
                f"{len(self.gate_poses_ground_truth)} ground-truth poses."
            )

    def cache_gate_records(self):
        self.gate_records = []
        gate_names = self.gate_names_ground_truth or discover_gate_names(self.airsim_client)
        for gate_name in gate_names:
            gate_pose = self.airsim_client.simGetObjectPose(gate_name)
            gate_width, gate_height = self._gate_dimensions(gate_name)
            if gate_width <= 1e-6 or gate_height <= 1e-6:
                continue

            self.gate_records.append(
                {
                    "name": gate_name,
                    "position": np.array(
                        [
                            gate_pose.position.x_val,
                            gate_pose.position.y_val,
                            gate_pose.position.z_val,
                        ],
                        dtype=np.float32,
                    ),
                    "rotation": quaternion_to_rotation_matrix(
                        np.array(
                            [
                                gate_pose.orientation.w_val,
                                gate_pose.orientation.x_val,
                                gate_pose.orientation.y_val,
                                gate_pose.orientation.z_val,
                            ],
                            dtype=np.float32,
                        )
                    ),
                    "width": gate_width,
                    "height": gate_height,
                }
            )

    def _sample_depth(self, depth_image: np.ndarray | None, u: float, v: float) -> float | None:
        if depth_image is None:
            return None
        if not np.isfinite(u) or not np.isfinite(v):
            return None
        x = int(round(float(u)))
        y = int(round(float(v)))
        if y < 0 or x < 0 or y >= depth_image.shape[0] or x >= depth_image.shape[1]:
            return None
        depth_value = float(depth_image[y, x])
        if not np.isfinite(depth_value) or depth_value <= 0.0:
            return None
        return depth_value

    def _gate_is_visible(
        self,
        gate: dict,
        response_width: int,
        response_height: int,
        camera_position: np.ndarray,
        camera_rotation: np.ndarray,
        intr: dict[str, float],
        depth_image: np.ndarray | None,
    ) -> bool:
        corners_world = gate_corners_world(
            gate_position=gate["position"],
            gate_rotation=gate["rotation"],
            gate_width=gate["width"],
            gate_height=gate["height"],
        )

        samples: list[bool] = []
        sample_points = [("center", gate["position"])] + list(corners_world.items())
        for _name, point_world in sample_points:
            point_cam = camera_rotation.T @ (point_world - camera_position)
            expected_depth = float(point_cam[0])
            if expected_depth <= 1e-6:
                return False

            uv = project_world_point(point_world, camera_position, camera_rotation, intr)
            if uv is None:
                return False
            u, v = uv
            if u < 0 or v < 0 or u >= response_width or v >= response_height:
                return False

            depth_value = self._sample_depth(depth_image, u, v)
            if depth_value is None:
                samples.append(False)
                continue

            margin_m = max(DEPTH_OCCLUSION_MARGIN_M, 0.05 * expected_depth)
            samples.append(depth_value + margin_m >= expected_depth)

        return sum(samples) >= MIN_VISIBLE_DEPTH_SAMPLES

    def _build_shapes(
        self,
        response_width: int,
        response_height: int,
        camera_position: np.ndarray,
        camera_rotation: np.ndarray,
        depth_image: np.ndarray | None,
    ) -> list[dict]:
        intr = compute_pinhole_intrinsics(response_width, response_height, self.camera_fov_degrees)
        shapes: list[dict] = []

        for gate in self.gate_records:
            if not self._gate_is_visible(
                gate=gate,
                response_width=response_width,
                response_height=response_height,
                camera_position=camera_position,
                camera_rotation=camera_rotation,
                intr=intr,
                depth_image=depth_image,
            ):
                continue

            corners_world = gate_corners_world(
                gate_position=gate["position"],
                gate_rotation=gate["rotation"],
                gate_width=gate["width"],
                gate_height=gate["height"],
            )

            projected = {}
            for corner_name, corner_world in corners_world.items():
                uv = project_world_point(corner_world, camera_position, camera_rotation, intr)
                if uv is None:
                    projected = {}
                    break
                u, v = uv
                if not (np.isfinite(u) and np.isfinite(v)):
                    projected = {}
                    break
                if u < 0 or v < 0 or u >= response_width or v >= response_height:
                    projected = {}
                    break
                projected[corner_name] = (u, v)

            if len(projected) != 4:
                continue

            for corner_name, (u, v) in projected.items():
                shapes.append(
                    {
                        "label": f'{gate["name"]}_{corner_name}',
                        "points": [[float(u), float(v)]],
                        "group_id": gate["name"],
                        "shape_type": "point",
                        "flags": {},
                    }
                )

        return shapes

    def _save_sample(self, image_bgr: np.ndarray, response_width: int, response_height: int, shapes: list[dict]) -> None:
        image_name = f"image_{self.sample_index + 1:06d}.png"
        image_path = self.rgb_dir / image_name
        json_path = self.ann_dir / f"image_{self.sample_index + 1:06d}.json"

        # Save the training input as RGB, not OpenCV's default BGR view.
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        cv2.imwrite(str(image_path), image_rgb)
        image_path_rel = os.path.relpath(image_path, start=self.ann_dir)
        record = build_labelme_record(
            image_path=image_path_rel,
            width=response_width,
            height=response_height,
            shapes=shapes,
        )
        json_path.write_text(json.dumps(record, indent=2), encoding="utf-8")

        if self.pairs_fh is not None:
            self.pairs_fh.write(f"{image_path} {json_path}\n")
            self.pairs_fh.flush()

        if self.save_overlay and self.sample_index < 10:
            overlay = image_bgr.copy()
            for shape in shapes:
                u, v = shape["points"][0]
                cv2.circle(overlay, (int(round(u)), int(round(v))), 4, (0, 0, 255), -1)
                cv2.putText(
                    overlay,
                    shape["label"],
                    (int(round(u)) + 4, int(round(v)) - 4),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.4,
                    (0, 255, 0),
                    1,
                    cv2.LINE_AA,
                )
            cv2.imwrite(str(self.overlay_dir / image_name), overlay)

        self.sample_index += 1
        self.samples_saved = self.sample_index
        if self.samples_saved % 100 == 0:
            print(f"Saved {image_name} with {len(shapes) // 4} gates")

    def image_callback(self):
        if not self.capture_active:
            return

        with self.capture_lock:
            if self.sample_index >= self.num_samples:
                self.capture_active = False
                return

            responses = self.airsim_client_images.simGetImages(
                [
                    airsim.ImageRequest(
                        self.camera_name,
                        airsim.ImageType.DepthPerspective,
                        pixels_as_float=True,
                        compress=False,
                    ),
                    airsim.ImageRequest(
                        self.camera_name,
                        airsim.ImageType.Scene,
                        pixels_as_float=False,
                        compress=False,
                    ),
                ],
                vehicle_name=self.drone_name,
            )
            if len(responses) < 2:
                return

            depth_response = responses[0]
            response = responses[1]
            if response.width <= 0 or response.height <= 0:
                return

            depth_image = None
            if depth_response.width > 0 and depth_response.height > 0:
                depth_image = np.array(depth_response.image_data_float, dtype=np.float32)
                depth_image = depth_image.reshape(depth_response.height, depth_response.width)

            rgb = np.frombuffer(response.image_data_uint8, dtype=np.uint8).copy()
            rgb = rgb.reshape(response.height, response.width, 3)
            image_bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

            camera_position = np.array(
                [
                    response.camera_position.x_val,
                    response.camera_position.y_val,
                    response.camera_position.z_val,
                ],
                dtype=np.float32,
            )
            camera_rotation = quaternion_to_rotation_matrix(
                np.array(
                    [
                        response.camera_orientation.w_val,
                        response.camera_orientation.x_val,
                        response.camera_orientation.y_val,
                        response.camera_orientation.z_val,
                    ],
                    dtype=np.float32,
                )
            )

            shapes = self._build_shapes(
                response_width=response.width,
                response_height=response.height,
                camera_position=camera_position,
                camera_rotation=camera_rotation,
                depth_image=depth_image,
            )

            self._save_sample(image_bgr, response.width, response.height, shapes)

            if self.show_preview:
                preview = image_bgr.copy()
                for shape in shapes:
                    u, v = shape["points"][0]
                    cv2.circle(preview, (int(round(u)), int(round(v))), 4, (0, 0, 255), -1)
                    cv2.putText(
                        preview,
                        shape["label"],
                        (int(round(u)) + 4, int(round(v)) - 4),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.4,
                        (0, 255, 0),
                        1,
                        cv2.LINE_AA,
                    )
                cv2.imshow("gate_corners", preview)
                cv2.waitKey(1)

            if self.sample_index >= self.num_samples:
                self.capture_active = False

    def collect_with_baseline_racer(self, flight_mode: str, max_flight_loops: int = 3, sleep_sec: float = 2.0, race_tier: int =  1, level_name: str | None = None):
        if level_name:
            self.level_name = level_name
        if self.level_name is None:
            raise RuntimeError("level_name must be set before collection starts.")
        race_tier = infer_race_tier(self.level_name, fallback=race_tier)
        print(f"Using level={self.level_name}, race_tier={race_tier}, corner_mode={self.corner_mode}")
        self.prepare_run_directory()

        if hasattr(self.airsim_client, "simFlushPersistentMarkers"):
            self.airsim_client.simFlushPersistentMarkers()

        self.load_level(self.level_name, sleep_sec=sleep_sec)
        self.start_race(race_tier)
        self.initialize_drone()
        self.takeoff_with_moveOnSpline()
        self.get_ground_truth_gate_poses()
        self.cache_gate_records()

        camera_info = self.airsim_client_images.simGetCameraInfo(self.camera_name, vehicle_name=self.drone_name)
        self.camera_fov_degrees = float(camera_info.fov) if math.isfinite(float(camera_info.fov)) else 90.0

        self.start_collection()

        try:
            loops = 0
            while self.samples_saved < self.num_samples and loops < int(max_flight_loops):
                if flight_mode == "all_gates_at_once":
                    self.fly_through_all_gates_at_once_with_moveOnSpline().join()
                else:
                    self.fly_through_all_gates_one_by_one_with_moveOnSplineVelConstraints().join()

                if self.samples_saved >= self.num_samples:
                    break

                loops += 1
                self.reset_race()
                if hasattr(self.airsim_client, "simFlushPersistentMarkers"):
                    self.airsim_client.simFlushPersistentMarkers()
                self.start_race(race_tier)
                self.takeoff_with_moveOnSpline()
                if not self.is_image_thread_active or not self.image_callback_thread.is_alive():
                    self._recreate_image_callback_thread()
                    self.start_image_callback_thread()

                if self.samples_saved >= self.num_samples:
                    break

            if self.samples_saved < self.num_samples:
                print(
                    f"Warning: only collected {self.samples_saved}/{self.num_samples} samples "
                    f"after {loops} flight loop(s)."
                )
        finally:
            self.stop_collection()
            self.reset_race()
            if hasattr(self.airsim_client, "simFlushPersistentMarkers"):
                self.airsim_client.simFlushPersistentMarkers()
            cv2.destroyAllWindows()


def main():
    args = build_args()
    output_root = Path(args.output_dir).expanduser().resolve()
    collector = GateCornerCollector(
        output_dir=output_root,
        run_name=args.run_name,
        num_samples=args.num_samples,
        camera_name=args.camera_name,
        corner_mode=args.corner_mode,
        image_period=args.image_period,
        save_overlay=args.save_overlay,
        show_preview=args.show_preview,
        drone_name=args.drone_name,
    )
    collector.collect_with_baseline_racer(
        flight_mode=args.flight_mode,
        max_flight_loops=args.max_flight_loops,
        sleep_sec=args.sleep_sec,
        race_tier=int(args.race_tier),
        level_name=args.level_name,
    )


if __name__ == "__main__":
    main()
