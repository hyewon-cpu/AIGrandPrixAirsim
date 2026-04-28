from __future__ import annotations

from argparse import ArgumentParser
from datetime import datetime
import time
import sys
import threading
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
AIRSIM_PY_DIR = REPO_ROOT / "airsimdroneracinglab-1.0.2"
BASELINES_DIR = REPO_ROOT / "baselines"
LOCAL_DEPTH_ROOT = (REPO_ROOT / "depth_estimation").resolve()
WORKSPACE_DEPTH_ROOT = Path("/workspace/depth_estimation")
if str(AIRSIM_PY_DIR) not in sys.path:
    sys.path.insert(0, str(AIRSIM_PY_DIR))
if str(BASELINES_DIR) not in sys.path:
    sys.path.insert(0, str(BASELINES_DIR))

try:
    import airsimdroneracinglab as airsim
    import cv2
    import numpy as np
    from baseline_racer import BaselineRacer
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "collect_tier1_rgb_depth.py requires numpy, OpenCV, and the local "
        "airsimdroneracinglab package. Activate the project environment that "
        "has those dependencies installed before running this script."
    ) from exc


def sanitize_name(value):
    cleaned = []
    for char in str(value):
        if char.isalnum() or char in ("-", "_"):
            cleaned.append(char)
        else:
            cleaned.append("_")
    return "".join(cleaned).strip("_") or "run"


def to_workspace_depth_path(path: Path) -> str:
    resolved = Path(path).resolve()
    try:
        relative_path = resolved.relative_to(LOCAL_DEPTH_ROOT)
    except ValueError:
        return str(resolved)
    return str(WORKSPACE_DEPTH_ROOT / relative_path)


class BaselineTier1Collector(BaselineRacer):
    def __init__(
        self,
        output_root,
        drone_name="drone_1",
        viz_traj=False,
        image_period=0.03,
        camera_name="fpv_cam",
        max_samples: int | None = None,
    ):
        self.output_root = Path(output_root).expanduser().resolve()
        self.image_period = float(image_period)
        self.camera_name = camera_name
        self.sample_index = 0
        self.samples_saved = 0
        self.max_samples = None if max_samples in (None, 0) else int(max_samples)
        self.last_image_callback_time = None
        self.run_dir = None
        self.rgb_dir = None
        self.depth_dir = None
        self.pairs_path = None
        self.pairs_fh = None
        self.capture_active = False
        self.capture_lock = threading.Lock()

        super().__init__(
            drone_name=drone_name,
            viz_traj=viz_traj,
            viz_image_cv2=False,
        )

        # Replace the default image polling interval with the requested one.
        self._recreate_callback_threads()

    def _recreate_callback_threads(self):
        """Threads cannot be started twice; recreate before each collection run."""
        self.image_callback_thread = threading.Thread(
            target=self.repeat_timer_image_callback,
            args=(self.image_callback, self.image_period),
            daemon=True,
        )

    def repeat_timer_image_callback(self, task, period):
        """Run image callback on a fixed-rate schedule based on wall-clock time."""
        period = float(period)
        next_tick = time.perf_counter()
        while self.is_image_thread_active:
            task()
            next_tick += period
            sleep_time = next_tick - time.perf_counter()
            if sleep_time > 0:
                time.sleep(sleep_time)
            else:
                # If callback is slower than requested period, skip catch-up sleeps.
                next_tick = time.perf_counter()

    def prepare_run_directory(
        self,
        level_name,
        planning_baseline_type,
        planning_and_control_api,
        run_name=None,
    ):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        if run_name:
            run_name = sanitize_name(run_name)
        else:
            run_name = timestamp

        self.run_dir = self.output_root / run_name
        self.run_dir.mkdir(parents=True, exist_ok=False)
        self.rgb_dir = self.run_dir / "rgb"
        self.depth_dir = self.run_dir / "depth"
        self.rgb_dir.mkdir()
        self.depth_dir.mkdir()
        self.pairs_path = self.run_dir / "pairs.txt"

        print(f"Saving collected data to: {self.run_dir}")
        return self.run_dir

    def start_collection(self):
        if self.run_dir is None or self.pairs_path is None:
            raise RuntimeError("Call prepare_run_directory() before start_collection().")

        self._sync_sample_index_with_existing_files()
        self._recreate_callback_threads()
        self.capture_active = True
        if self.pairs_fh is None:
            self.pairs_fh = open(self.pairs_path, "a", encoding="utf-8")
        self.start_image_callback_thread()

    def _sync_sample_index_with_existing_files(self):
        """Continue numbering from existing files to avoid overwriting."""
        if self.rgb_dir is None or self.depth_dir is None:
            return
        max_index = -1
        for image_path in self.rgb_dir.glob("image_*.png"):
            stem = image_path.stem
            try:
                num = int(stem.split("_")[-1])
            except ValueError:
                continue
            # File names are 1-based (image_001), sample_index is 0-based.
            max_index = max(max_index, num - 1)
        for depth_path in self.depth_dir.glob("image_*.pfm"):
            stem = depth_path.stem
            try:
                num = int(stem.split("_")[-1])
            except ValueError:
                continue
            max_index = max(max_index, num - 1)
        for depth_path in self.depth_dir.glob("image_*.png"):
            stem = depth_path.stem
            try:
                num = int(stem.split("_")[-1])
            except ValueError:
                continue
            max_index = max(max_index, num - 1)
        if max_index >= self.sample_index:
            self.sample_index = max_index + 1

    def stop_collection(self):
        self.capture_active = False
        self.stop_image_callback_thread()
        if self.pairs_fh is not None:
            self.pairs_fh.flush()
            self.pairs_fh.close()
            self.pairs_fh = None

    def _write_sample(self, sample, pairs_file):
        sample_index = int(sample["sample_index"])
        rgb_path = self.rgb_dir / f"image_{sample_index + 1:03d}.png"
        depth_path = self.depth_dir / f"image_{sample_index + 1:03d}.pfm"

        cv2.imwrite(str(rgb_path), sample["rgb"])

        depth_image = sample["depth"].astype(np.float32, copy=False)
        # AirSim depth can appear vertically inverted when stored directly.
        # Flip before writing so saved depth aligns with RGB orientation.
        depth_image = np.flipud(depth_image)
        airsim.write_pfm(str(depth_path), depth_image)

        pairs_file.write(
            f"{to_workspace_depth_path(rgb_path)} {to_workspace_depth_path(depth_path)}\n"
        )
        pairs_file.flush()

    def image_callback(self):
        if not self.capture_active:
            return

        with self.capture_lock:
            if self.max_samples is not None and self.samples_saved >= self.max_samples:
                self.capture_active = False
                return
            if self.pairs_fh is None:
                return

            try:
                now = time.perf_counter()
                dt = None if self.last_image_callback_time is None else (now - self.last_image_callback_time)
                self.last_image_callback_time = now

                responses = self.airsim_client_images.simGetImages(
                    [
                        airsim.ImageRequest(
                            self.camera_name,
                            airsim.ImageType.Scene,
                            pixels_as_float=False,
                            compress=False,
                        ),
                        airsim.ImageRequest(
                            self.camera_name,
                            airsim.ImageType.DepthPerspective,
                            pixels_as_float=True,
                            compress=False,
                        ),
                    ],
                    vehicle_name=self.drone_name,
                )
                if len(responses) < 2:
                    return

                rgb_response = responses[0]
                depth_response = responses[1]
                if (
                    rgb_response.width <= 0
                    or rgb_response.height <= 0
                    or depth_response.width <= 0
                    or depth_response.height <= 0
                ):
                    return

                rgb = np.frombuffer(rgb_response.image_data_uint8, dtype=np.uint8).copy()
                rgb = rgb.reshape(rgb_response.height, rgb_response.width, 3)

                depth = np.array(depth_response.image_data_float, dtype=np.float32)
                depth = depth.reshape(depth_response.height, depth_response.width)

                sample = {
                    "sample_index": self.sample_index,
                    "rgb": rgb,
                    "depth": depth,
                }
                self._write_sample(sample, self.pairs_fh)
                self.samples_saved += 1
                self.sample_index += 1

                if self.samples_saved % 100 == 0:
                    if dt is None or dt <= 0.0:
                        print(f"[collector] samples={self.samples_saved}")
                    else:
                        print(
                            f"[collector] samples={self.samples_saved} "
                            f"callback_dt={dt:.3f}s (~{1.0/dt:.1f} Hz)"
                        )
            except Exception as exc:  # pragma: no cover
                print("image_callback failed:", exc)


def build_args():
    parser = ArgumentParser(description="Collect tier-1 AirSim RGB and depth pairs.")
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
        default="Qualifier_Tier_1",
    )
    parser.add_argument("--race_tier", type=int, choices=[1, 2, 3], default=1)
    parser.add_argument("--drone_name", type=str, default="drone_1")
    parser.add_argument(
        "--output_dir",
        type=str,
        default=str(REPO_ROOT / "depth_estimation" / "datasets"),
    )
    parser.add_argument("--run_name", type=str, default="")
    parser.add_argument("--num_games",type=int, default=100,
        help="Number of races to run sequentially for data collection.",
    )
    parser.add_argument("--max_samples", type=int,default=4000,
        help="Stop after saving this many samples total across all games. 0 disables the limit.",
    )
    parser.add_argument("--camera_name", type=str, default="fpv_cam")
    parser.add_argument("--image_period", type=float, default=0.03)
    parser.add_argument("--takeoff_height", type=float, default=1.0)
    parser.add_argument(
        "--planning_baseline_type",
        type=str,
        choices=["all_gates_at_once", "all_gates_one_by_one"],
        default="all_gates_at_once",
    )
    parser.add_argument(
        "--planning_and_control_api",
        type=str,
        choices=["moveOnSpline", "moveOnSplineVelConstraints"],
        default="moveOnSpline",
    )
    parser.add_argument(
        "--enable_viz_traj",
        dest="viz_traj",
        action="store_true",
        default=False,
    )
    return parser.parse_args()


def main():
    args = build_args()
    if args.num_games < 1:
        raise ValueError(f"--num_games must be >= 1, got {args.num_games}")
    if args.max_samples < 0:
        raise ValueError(f"--max_samples must be >= 0, got {args.max_samples}")

    # Match the baseline script's tier handling for qualifier levels.
    if args.level_name == "Qualifier_Tier_1":
        args.race_tier = 1
    if args.level_name == "Qualifier_Tier_2":
        args.race_tier = 2
    if args.level_name == "Qualifier_Tier_3":
        args.race_tier = 3

    collector = BaselineTier1Collector(
        output_root=args.output_dir,
        drone_name=args.drone_name,
        viz_traj=args.viz_traj,
        image_period=args.image_period,
        camera_name=args.camera_name,
        max_samples=args.max_samples,
    )

    try:
        collector.prepare_run_directory(
            level_name=args.level_name,
            planning_baseline_type=args.planning_baseline_type,
            planning_and_control_api=args.planning_and_control_api,
            run_name=args.run_name or None,
        )

        for game_index in range(args.num_games):
            if args.max_samples and collector.samples_saved >= args.max_samples:
                print(
                    f"Reached --max_samples={args.max_samples}; stopping data collection."
                )
                break
            print(f"Starting game {game_index + 1}/{args.num_games}")
            game_start_sample_index = collector.sample_index
            game_start_saved = collector.samples_saved

            # Match `annotate_gate_corners.py`: load the level once, then just reset/start.
            if game_index == 0:
                collector.load_level(args.level_name)
            collector.start_race(args.race_tier)
            collector.initialize_drone()
            collector.takeoff_with_moveOnSpline(takeoff_height=args.takeoff_height)
            collector.get_ground_truth_gate_poses()
            collector.start_collection()

            try:
                if args.planning_baseline_type == "all_gates_at_once":
                    if args.planning_and_control_api == "moveOnSpline":
                        flight = collector.fly_through_all_gates_at_once_with_moveOnSpline()
                    else:
                        flight = collector.fly_through_all_gates_at_once_with_moveOnSplineVelConstraints()
                else:
                    if args.planning_and_control_api == "moveOnSpline":
                        flight = collector.fly_through_all_gates_one_by_one_with_moveOnSpline()
                    else:
                        flight = collector.fly_through_all_gates_one_by_one_with_moveOnSplineVelConstraints()

                flight.join()
            finally:
                try:
                    collector.stop_collection()
                except Exception as exc:  # pragma: no cover
                    print(f"Failed to stop collection cleanly: {exc}")
                try:
                    collector.reset_race()
                except Exception as exc:  # pragma: no cover
                    print(f"Failed to reset race cleanly: {exc}")
                if args.max_samples and collector.samples_saved >= args.max_samples:
                    print(
                        f"Reached --max_samples={args.max_samples}; stopping data collection."
                    )
                    break

                if collector.run_dir is not None:
                    game_saved = collector.samples_saved - game_start_saved
                    game_samples = collector.sample_index - game_start_sample_index
                    print(
                        f"Game {game_index + 1}/{args.num_games}: "
                        f"new_samples={game_samples}, saved={game_saved}. "
                        f"total_saved={collector.samples_saved} to {collector.run_dir}"
                    )
    except KeyboardInterrupt:
        print("Interrupted by user.")


if __name__ == "__main__":
    main()
