from __future__ import annotations

from argparse import ArgumentParser
from datetime import datetime
import queue
import sys
import threading
import time
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
    from diffsim_racer import DEFAULT_MODEL_PATH, DiffSimRacer
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "data_collection_diffsim.py requires numpy, OpenCV, the local "
        "airsimdroneracinglab package, and baselines/diffsim_racer.py dependencies."
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


def normalize_last_gate_passed(raw_gate_index, total_gates=0):
    gate_index = int(raw_gate_index)
    if gate_index >= (1 << 31):
        gate_index -= 1 << 32
    elif gate_index >= (1 << 15):
        gate_index -= 1 << 16

    if gate_index < -1:
        return -1
    if total_gates > 0 and gate_index >= total_gates:
        return -1
    return gate_index


class DiffSimCollector(DiffSimRacer):
    def __init__(
        self,
        output_root,
        camera_name="fpv_cam",
        queue_size=256,
        **kwargs,
    ):
        self.output_root = Path(output_root).expanduser().resolve()
        self.camera_name = camera_name
        self.sample_queue = queue.Queue(maxsize=int(queue_size))
        self.writer_stop = threading.Event()
        self.writer_thread = None
        self.collecting = False
        self.sample_index = 0
        self.samples_saved = 0
        self.samples_dropped = 0
        self.last_image_callback_time = None
        self.run_dir = None
        self.rgb_dir = None
        self.depth_dir = None
        self.pairs_path = None

        super().__init__(**kwargs)

        self.airsim_client_race_monitor = airsim.MultirotorClient()
        self.airsim_client_race_monitor.confirmConnection()
        self._reset_diffsim_run_state()
        self._recreate_callback_threads()

    def _reset_diffsim_run_state(self):
        self.last_depth = None
        self.last_depth_timestamp = 0.0
        self.last_depth_frame_id = 0
        self.last_control_depth_frame_id = -1
        self.last_rgb = None
        self.last_segmentation = None
        self.last_segmentation_mask = None
        self.last_segmentation_response = None
        self.last_segmentation_timestamp = 0.0
        self.last_segmentation_target_airsim = None
        self.last_segmentation_backup_target_airsim = None
        self.last_segmentation_candidate_targets_airsim = []
        self.last_segmentation_candidate_timestamp = 0.0
        self.camera_intrinsics = None
        self.last_target_source = "forward_fallback"
        self.current_state = None
        self.current_forward = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        self.active_gate_index = 0
        self.debug_counter = 0
        self.last_image_callback_time = None
        if self.model is not None:
            self.model.hidden = None

    def _recreate_callback_threads(self):
        self.image_callback_thread = threading.Thread(
            target=self.repeat_timer_image_callback,
            args=(self.image_callback, self.image_period),
            daemon=True,
        )
        self.control_loop_thread = threading.Thread(
            target=self.repeat_timer_control_callback,
            args=(self.control_callback, self.control_period),
            daemon=True,
        )

    def load_level(self, level_name, sleep_sec=2.0):
        super().load_level(level_name, sleep_sec=sleep_sec)
        self.airsim_client_race_monitor.confirmConnection()
        self._reset_diffsim_run_state()

    def prepare_run_directory(
        self,
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

    def _sync_sample_index_with_existing_files(self):
        if self.rgb_dir is None or self.depth_dir is None:
            return
        max_index = -1
        for image_path in self.rgb_dir.glob("image_*.png"):
            stem = image_path.stem
            try:
                num = int(stem.split("_")[-1])
            except ValueError:
                continue
            max_index = max(max_index, num - 1)
        for depth_path in self.depth_dir.glob("image_*.pfm"):
            stem = depth_path.stem
            try:
                num = int(stem.split("_")[-1])
            except ValueError:
                continue
            max_index = max(max_index, num - 1)
        if max_index >= self.sample_index:
            self.sample_index = max_index + 1

    def start_collection(self):
        if self.run_dir is None or self.pairs_path is None:
            raise RuntimeError("Call prepare_run_directory() before start_collection().")

        self._sync_sample_index_with_existing_files()
        self._recreate_callback_threads()
        self.writer_stop.clear()
        self.writer_thread = threading.Thread(target=self._writer_loop, daemon=True)
        self.writer_thread.start()
        self.collecting = True
        try:
            self.start_model_control()
        except Exception:
            self.collecting = False
            self.writer_stop.set()
            if self.writer_thread is not None:
                self.writer_thread.join(timeout=1.0)
                self.writer_thread = None
            raise

    def stop_collection(self):
        if not self.collecting:
            return

        self.collecting = False
        try:
            self.stop_threads()
        finally:
            self.writer_stop.set()
            if self.writer_thread is not None:
                self.sample_queue.join()
                self.writer_thread.join()
                self.writer_thread = None
            self.last_image_callback_time = None

    def _enqueue_sample(self, sample):
        try:
            self.sample_queue.put_nowait(sample)
        except queue.Full:
            self.samples_dropped += 1
            if self.samples_dropped % 20 == 1:
                print(
                    f"WARNING: sample queue is full; dropped {self.samples_dropped} frames so far."
                )

    def _record_sample(self, rgb, depth, dt=None):
        if not self.collecting or rgb is None or depth is None:
            return

        sample = {
            "sample_index": self.sample_index,
            "rgb": rgb,
            "depth": depth,
        }
        self.sample_index += 1
        self._enqueue_sample(sample)
        if self.sample_index % 100 == 0:
            if dt is None or dt <= 0.0:
                print(f"[collector] samples={self.sample_index}")
            else:
                print(
                    f"[collector] samples={self.sample_index} "
                    f"callback_dt={dt:.3f}s (~{1.0 / dt:.1f} Hz)"
                )

    def _writer_loop(self):
        with open(self.pairs_path, "a", encoding="utf-8") as pairs_file:
            while not self.writer_stop.is_set() or not self.sample_queue.empty():
                try:
                    sample = self.sample_queue.get(timeout=0.1)
                except queue.Empty:
                    continue

                try:
                    self._write_sample(sample, pairs_file)
                    self.samples_saved += 1
                except Exception as exc:  # pragma: no cover
                    print(f"Failed to save sample {sample.get('sample_index')}: {exc}")
                finally:
                    self.sample_queue.task_done()

    def _write_sample(self, sample, pairs_file):
        sample_index = int(sample["sample_index"])
        rgb_path = self.rgb_dir / f"image_{sample_index + 1:03d}.png"
        depth_path = self.depth_dir / f"image_{sample_index + 1:03d}.pfm"

        cv2.imwrite(str(rgb_path), sample["rgb"])

        depth_image = sample["depth"].astype(np.float32, copy=False)
        depth_image = np.flipud(depth_image)
        airsim.write_pfm(str(depth_path), depth_image)

        pairs_file.write(
            f"{to_workspace_depth_path(rgb_path)} {to_workspace_depth_path(depth_path)}\n"
        )
        pairs_file.flush()

    def get_sensor_images(self):
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
                    airsim.ImageType.Segmentation,
                    pixels_as_float=False,
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
        if len(responses) < 3:
            return None, None, None, None

        depth_response = responses[0]
        segmentation_response = responses[1]
        rgb_response = responses[2]

        depth = None
        if depth_response.width > 0 and depth_response.height > 0:
            depth = np.array(depth_response.image_data_float, dtype=np.float32)
            depth = depth.reshape(depth_response.height, depth_response.width)

        segmentation = None
        if segmentation_response.width > 0 and segmentation_response.height > 0:
            segmentation = np.frombuffer(
                segmentation_response.image_data_uint8, dtype=np.uint8
            ).copy()
            segmentation = segmentation.reshape(
                segmentation_response.height, segmentation_response.width, 3
            )

        rgb = None
        if rgb_response.width > 0 and rgb_response.height > 0:
            rgb = np.frombuffer(rgb_response.image_data_uint8, dtype=np.uint8).copy()
            rgb = rgb.reshape(rgb_response.height, rgb_response.width, 3)

        return depth, segmentation, rgb, segmentation_response

    def image_callback(self):
        try:
            now = time.perf_counter()
            dt = None
            if self.last_image_callback_time is not None:
                dt = now - self.last_image_callback_time
            self.last_image_callback_time = now

            depth, segmentation, rgb, segmentation_response = self.get_sensor_images()
            sensor_now = time.time()
            if depth is not None:
                self.last_depth = depth
                self.last_depth_timestamp = sensor_now
                self.last_depth_frame_id += 1
                if self.viz_depth and cv2 is not None:
                    depth_viz = np.clip(depth / 25.0 * 255.0, 0, 255).astype(np.uint8)
                    cv2.imshow("depth", depth_viz)
                    cv2.waitKey(1)
            if rgb is not None:
                self.last_rgb = rgb
                if self.viz_rgb and cv2 is not None:
                    rgb_viz = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
                    cv2.imshow("rgb", rgb_viz)
                    cv2.waitKey(1)
                if depth is not None:
                    self._record_sample(rgb, depth, dt=dt)
            else:
                self.last_rgb = None

            if segmentation is not None:
                self.last_segmentation = segmentation
                self.last_segmentation_response = segmentation_response
                self.last_segmentation_mask = self.build_segmentation_mask(segmentation)
                self.last_segmentation_timestamp = sensor_now
                did_show_segmentation = False
                if self.viz_segmentation_map and cv2 is not None:
                    seg_viz = cv2.cvtColor(segmentation, cv2.COLOR_RGB2BGR)
                    cv2.imshow("segmentation_map", seg_viz)
                    cv2.setMouseCallback(
                        "segmentation_map", self.segmentation_map_mouse_callback
                    )
                    did_show_segmentation = True
                if (
                    self.viz_gate_mask
                    and cv2 is not None
                    and self.last_segmentation_mask is not None
                ):
                    mask_viz = self.last_segmentation_mask.astype(np.uint8) * 255
                    cv2.imshow("gate_mask", mask_viz)
                    did_show_segmentation = True
                if did_show_segmentation:
                    cv2.waitKey(1)
            else:
                self.last_segmentation = None
                self.last_segmentation_response = None
                self.last_segmentation_mask = None
        except Exception as exc:  # pragma: no cover
            print("image_callback failed:", exc)


def wait_for_diffsim_run(collector, max_duration=90.0, poll_period=0.25):
    start_time = time.perf_counter()
    last_gate_passed = -1
    last_invalid_gate_value = None
    total_gates = len(collector.gate_poses_ground_truth)

    if total_gates <= 0 and max_duration <= 0.0:
        raise RuntimeError(
            "DiffSim collection needs either known gate poses or a positive "
            "--diffsim_max_duration so the run can stop."
        )

    while True:
        if collector.airsim_client_race_monitor.simIsRacerDisqualified(
            vehicle_name=collector.drone_name
        ):
            return "disqualified", time.perf_counter() - start_time

        raw_gate_passed = collector.airsim_client_race_monitor.simGetLastGatePassed(
            vehicle_name=collector.drone_name
        )
        current_gate_passed = normalize_last_gate_passed(
            raw_gate_passed,
            total_gates=total_gates,
        )
        raw_gate_passed_int = int(raw_gate_passed)
        if (
            current_gate_passed != raw_gate_passed_int
            and raw_gate_passed_int not in (-1, current_gate_passed)
            and raw_gate_passed_int != last_invalid_gate_value
        ):
            last_invalid_gate_value = raw_gate_passed_int
            print(
                f"[diffsim] ignoring invalid last_gate_passed={raw_gate_passed_int}"
            )

        if current_gate_passed != last_gate_passed and current_gate_passed >= 0:
            last_gate_passed = current_gate_passed
            print(f"[diffsim] passed_gate={current_gate_passed + 1}/{total_gates}")

        if total_gates > 0 and current_gate_passed >= total_gates - 1:
            return "finished", time.perf_counter() - start_time

        elapsed = time.perf_counter() - start_time
        if max_duration > 0.0 and elapsed >= max_duration:
            return "timeout", elapsed

        time.sleep(poll_period)


def build_args():
    parser = ArgumentParser(
        description="Collect AirSim RGB and depth pairs while flying with DiffSimRacer."
    )
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
    parser.add_argument(
        "--output_dir",
        type=str,
        default=str(REPO_ROOT / "depth_estimation" / "datasets"),
    )
    parser.add_argument("--run_name", type=str, default="")
    parser.add_argument(
        "--num_games",
        type=int,
        default=100,
        help="Number of races to run sequentially for data collection.",
    )
    parser.add_argument("--camera_name", type=str, default="fpv_cam")
    parser.add_argument("--image_period", type=float, default=0.03)
    parser.add_argument("--takeoff_height", type=float, default=1.0)
    parser.add_argument(
        "--model_path",
        type=str,
        default=DEFAULT_MODEL_PATH,
    )
    parser.add_argument(
        "--control_mode",
        type=str,
        choices=["velocity", "attitude"],
        default="attitude",
    )
    parser.add_argument("--control_period", type=float, default=0.05)
    parser.add_argument("--target_speed", type=float, default=7.0)
    parser.add_argument("--hover_throttle", type=float, default=0.9)
    parser.add_argument("--dim_obs", type=int, default=10)
    parser.add_argument("--dim_action", type=int, default=6)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--no_odom", action="store_true", default=False)
    parser.add_argument("--margin", type=float, default=0.2)
    parser.add_argument("--post_takeoff_delay", type=float, default=1.0)
    parser.add_argument("--max_velocity", type=float, default=8.0)
    parser.add_argument("--max_vertical_velocity", type=float, default=3.0)
    parser.add_argument("--velocity_gain_xy", type=float, default=2.0)
    parser.add_argument("--velocity_gain_z", type=float, default=2.0)
    parser.add_argument(
        "--target_source",
        type=str,
        choices=["ground_truth", "segmentation"],
        default="segmentation",
    )
    parser.add_argument(
        "--target_rgb",
        type=int,
        nargs=3,
        default=[134, 93, 210],
        metavar=("R", "G", "B"),
    )
    parser.add_argument(
        "--segmentation_promote_depth_threshold",
        type=float,
        default=2.0,
    )
    parser.add_argument("--debug_print", action="store_true", default=False)
    parser.add_argument("--debug_print_every", type=int, default=10)
    parser.add_argument("--viz_rgb", action="store_true", default=False)
    parser.add_argument("--viz_depth", action="store_true", default=False)
    parser.add_argument("--viz_segmentation", action="store_true", default=False)
    parser.add_argument("--viz_segmentation_map", action="store_true", default=False)
    parser.add_argument("--viz_gate_mask", action="store_true", default=False)
    parser.add_argument(
        "--diffsim_max_duration",
        type=float,
        default=90.0,
        help="Maximum seconds to keep a DiffSim-controlled game running; <= 0 disables the timeout.",
    )
    return parser.parse_args()


def main():
    args = build_args()
    if args.num_games < 1:
        raise ValueError(f"--num_games must be >= 1, got {args.num_games}")

    if args.level_name == "Qualifier_Tier_1":
        args.race_tier = 1
    if args.level_name == "Qualifier_Tier_2":
        args.race_tier = 2
    if args.level_name == "Qualifier_Tier_3":
        args.race_tier = 3

    collector = DiffSimCollector(
        output_root=args.output_dir,
        camera_name=args.camera_name,
        drone_name=args.drone_name,
        viz_rgb=args.viz_rgb,
        viz_depth=args.viz_depth,
        viz_segmentation=args.viz_segmentation,
        viz_segmentation_map=args.viz_segmentation_map,
        viz_gate_mask=args.viz_gate_mask,
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
        target_rgb=args.target_rgb,
        segmentation_promote_depth_threshold=args.segmentation_promote_depth_threshold,
        debug_print=args.debug_print,
        debug_print_every=args.debug_print_every,
    )

    try:
        collector.prepare_run_directory(
            run_name=args.run_name or None,
        )

        for game_index in range(args.num_games):
            print(f"Starting game {game_index + 1}/{args.num_games} with DiffSimRacer")
            game_start_sample_index = collector.sample_index
            game_start_saved = collector.samples_saved
            game_start_dropped = collector.samples_dropped
            flight_status = "aborted"
            flight_duration = None

            collector.load_level(args.level_name)
            collector.start_race(args.race_tier)
            collector.initialize_drone()
            collector.get_ground_truth_gate_poses()
            collector.takeoff(takeoff_height=args.takeoff_height)
            collector.start_collection()

            try:
                flight_status, flight_duration = wait_for_diffsim_run(
                    collector,
                    max_duration=args.diffsim_max_duration,
                )
            finally:
                try:
                    collector.stop_collection()
                except Exception as exc:  # pragma: no cover
                    print(f"Failed to stop collection cleanly: {exc}")
                try:
                    collector.reset_race()
                except Exception as exc:  # pragma: no cover
                    print(f"Failed to reset race cleanly: {exc}")

                if collector.run_dir is not None:
                    game_saved = collector.samples_saved - game_start_saved
                    game_dropped = collector.samples_dropped - game_start_dropped
                    game_samples = collector.sample_index - game_start_sample_index
                    duration_text = ""
                    if flight_duration is not None:
                        duration_text = f", flight_time={flight_duration:.1f}s"
                    print(
                        f"Game {game_index + 1}/{args.num_games}: "
                        f"status={flight_status}{duration_text}, "
                        f"new_samples={game_samples}, saved={game_saved}, dropped={game_dropped}. "
                        f"total_saved={collector.samples_saved} to {collector.run_dir}"
                    )
    except KeyboardInterrupt:
        print("Interrupted by user.")


if __name__ == "__main__":
    main()
