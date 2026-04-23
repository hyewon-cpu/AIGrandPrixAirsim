#!/usr/bin/env python3

"""Collect flipped depth frames and annotate the two closest visible gates.

This variant is derived from `annotate_gate_corners.py`, but it changes two
things:

- only the two closest visible gates are annotated
- the capture input is depth, not RGB, and depth maps are saved as vertically
  flipped PFM files

Corner labels are mirrored top/bottom to match the AirSim camera frame
convention used here.
"""

from __future__ import annotations

from argparse import ArgumentParser
from pathlib import Path
import os
import json

import cv2
import numpy as np
import airsimdroneracinglab as airsim

from annotate_gate_corners import (
    GateCornerCollector,
    build_labelme_record,
    compute_pinhole_intrinsics,
    quaternion_to_rotation_matrix,
    project_world_point,
    gate_corners_world,
    infer_race_tier,
    discover_gate_names,
    sanitize_name,
    MIN_VISIBLE_DEPTH_SAMPLES,
    DEPTH_OCCLUSION_MARGIN_M,
    GATE_NAME_RE,
)


def flip_corner_label(label: str) -> str:
    mapping = {
        "TL": "BL",
        "BL": "TL",
        "TR": "BR",
        "BR": "TR",
    }
    return mapping.get(label, label)


def write_pfm(path: Path, image: np.ndarray) -> None:
    """Write a single-channel PFM file."""
    if image.ndim != 2:
        raise ValueError(f"PFM expects a 2D array, got {image.shape}")

    image = np.asarray(image, dtype=np.float32)
    height, width = image.shape
    with path.open("wb") as fh:
        fh.write(b"Pf\n")
        fh.write(f"{width} {height}\n".encode("ascii"))
        fh.write(b"-1.0\n")
        fh.write(image.astype(np.float32).tobytes())


class GateCornerDepthCollector(GateCornerCollector):
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
        super().__init__(
            output_dir=output_dir,
            run_name=run_name,
            num_samples=num_samples,
            camera_name=camera_name,
            corner_mode=corner_mode,
            image_period=image_period,
            save_overlay=save_overlay,
            show_preview=show_preview,
            drone_name=drone_name,
        )
        self.depth_dir = self.run_dir / "depth"
        self.rgb_dir = self.depth_dir
        self.depth_display_name = "gate_depth"

    def prepare_run_directory(self):
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.run_dir.mkdir(parents=True, exist_ok=False)
        self.depth_dir.mkdir()
        self.ann_dir.mkdir()
        if self.save_overlay:
            self.overlay_dir.mkdir()
        self.pairs_fh = self.pairs_path.open("w", encoding="utf-8")
        print(f"Saving collected data to: {self.run_dir}")

    def _build_shapes(
        self,
        response_width: int,
        response_height: int,
        camera_position: np.ndarray,
        camera_rotation: np.ndarray,
        depth_image: np.ndarray | None,
    ) -> list[dict]:
        intr = compute_pinhole_intrinsics(response_width, response_height, self.camera_fov_degrees)
        visible_gates: list[tuple[float, dict]] = []

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
            distance = float(np.linalg.norm(gate["position"] - camera_position))
            visible_gates.append((distance, gate))

        visible_gates.sort(key=lambda item: item[0])
        selected_gates = [gate for _distance, gate in visible_gates[:2]]

        shapes: list[dict] = []
        for gate in selected_gates:
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
                flipped_name = flip_corner_label(corner_name)
                shapes.append(
                    {
                        "label": f'{gate["name"]}_{flipped_name}',
                        "points": [[float(u), float(v)]],
                        "group_id": gate["name"],
                        "shape_type": "point",
                        "flags": {},
                    }
                )

        return shapes

    def _save_sample(
        self,
        depth_image: np.ndarray,
        response_width: int,
        response_height: int,
        shapes: list[dict],
    ) -> None:
        image_name = f"image_{self.sample_index + 1:06d}.pfm"
        image_path = self.depth_dir / image_name
        json_path = self.ann_dir / f"image_{self.sample_index + 1:06d}.json"

        depth_flipped = np.flipud(depth_image)
        write_pfm(image_path, depth_flipped)
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

        if self.save_overlay and self.sample_index < 100:
            overlay = cv2.applyColorMap(np.clip(depth_image, 0.0, 200.0).astype(np.uint8), cv2.COLORMAP_JET)
            for shape in shapes:
                u, v = shape["points"][0]
                cv2.circle(overlay, (int(round(u)), int(round(v))), 4, (0, 0, 255), -1)
                cv2.putText(
                    overlay,
                    shape["label"],
                    (int(round(u)) + 4, int(round(v)) - 4),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.4,
                    (255, 255, 255),
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
                ],
                vehicle_name=self.drone_name,
            )
            if len(responses) < 1:
                return

            depth_response = responses[0]
            if depth_response.width <= 0 or depth_response.height <= 0:
                return

            depth_image = np.array(depth_response.image_data_float, dtype=np.float32)
            depth_image = depth_image.reshape(depth_response.height, depth_response.width)

            camera_position = np.array(
                [
                    depth_response.camera_position.x_val,
                    depth_response.camera_position.y_val,
                    depth_response.camera_position.z_val,
                ],
                dtype=np.float32,
            )
            camera_rotation = quaternion_to_rotation_matrix(
                np.array(
                    [
                        depth_response.camera_orientation.w_val,
                        depth_response.camera_orientation.x_val,
                        depth_response.camera_orientation.y_val,
                        depth_response.camera_orientation.z_val,
                    ],
                    dtype=np.float32,
                )
            )

            shapes = self._build_shapes(
                response_width=depth_response.width,
                response_height=depth_response.height,
                camera_position=camera_position,
                camera_rotation=camera_rotation,
                depth_image=depth_image,
            )

            self._save_sample(depth_image, depth_response.width, depth_response.height, shapes)

            if self.show_preview:
                preview = cv2.applyColorMap((np.clip(depth_image, 0.0, 200.0).astype(np.uint8)), cv2.COLORMAP_JET)
                for shape in shapes:
                    u, v = shape["points"][0]
                    cv2.circle(preview, (int(round(u)), int(round(v))), 4, (0, 0, 255), -1)
                    cv2.putText(
                        preview,
                        shape["label"],
                        (int(round(u)) + 4, int(round(v)) - 4),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.4,
                        (255, 255, 255),
                        1,
                        cv2.LINE_AA,
                    )
                cv2.imshow("gate_depth", preview)
                cv2.waitKey(1)

            if self.sample_index >= self.num_samples:
                self.capture_active = False


def build_args():
    parser = ArgumentParser(description="Fly with BaselineRacer and annotate the two closest gate corners from depth.")
    parser.add_argument("--output_dir", type=str, default=str(Path(__file__).resolve().parents[1] / "gate_detection" / "datasets_depth"), help="Root directory for the generated training set.")
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


def main():
    args = build_args()
    output_root = Path(args.output_dir).expanduser().resolve()
    collector = GateCornerDepthCollector(
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
