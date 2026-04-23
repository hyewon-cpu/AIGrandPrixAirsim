from __future__ import annotations

"""ONNX-backed racer that predicts depth from RGB before control."""

from argparse import ArgumentParser
from pathlib import Path
import sys
import time

import airsimdroneracinglab as airsim
import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
DEFAULT_DEPTH_ONNX_PATH = (
    REPO_ROOT
    / "depth_estimation"
    / "results"
    / "run_20260422_160049"
    / "export"
    / "dn_model_latest.onnx"
)
DEFAULT_DEPTH_CHECKPOINT = DEFAULT_DEPTH_ONNX_PATH
DEFAULT_DEPTH_INPUT_WIDTH = 252
DEFAULT_DEPTH_INPUT_HEIGHT = 140
DEFAULT_DEPTH_INPUT_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
DEFAULT_DEPTH_INPUT_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

try:
    import cv2
except ImportError:  # pragma: no cover
    cv2 = None

try:
    import onnxruntime as ort
except ImportError as exc:  # pragma: no cover
    ort = None
    ONNXRUNTIME_IMPORT_ERROR = exc
else:
    ONNXRUNTIME_IMPORT_ERROR = None

from diffsim_racer import DEFAULT_MODEL_PATH, DiffSimRacer


DEFAULT_DEPTH_DEVICE = "auto"


def _resize_bilinear_numpy(image, target_width, target_height):
    """Resize an HxWxC image with a pure-numpy bilinear fallback."""
    src_height, src_width = image.shape[:2]
    if src_height == target_height and src_width == target_width:
        return image

    y_coords = np.linspace(0.0, max(src_height - 1, 0), target_height, dtype=np.float32)
    x_coords = np.linspace(0.0, max(src_width - 1, 0), target_width, dtype=np.float32)
    x_grid, y_grid = np.meshgrid(x_coords, y_coords)

    x0 = np.floor(x_grid).astype(np.int32)
    y0 = np.floor(y_grid).astype(np.int32)
    x1 = np.clip(x0 + 1, 0, src_width - 1)
    y1 = np.clip(y0 + 1, 0, src_height - 1)

    wa = (x1 - x_grid) * (y1 - y_grid)
    wb = (x_grid - x0) * (y1 - y_grid)
    wc = (x1 - x_grid) * (y_grid - y0)
    wd = (x_grid - x0) * (y_grid - y0)

    top_left = image[y0, x0]
    top_right = image[y0, x1]
    bottom_left = image[y1, x0]
    bottom_right = image[y1, x1]
    return (
        wa[..., None] * top_left
        + wb[..., None] * top_right
        + wc[..., None] * bottom_left
        + wd[..., None] * bottom_right
    )


def _resize_rgb_image(image, target_width, target_height):
    if cv2 is not None:
        interpolation = cv2.INTER_AREA
        if target_width > image.shape[1] or target_height > image.shape[0]:
            interpolation = cv2.INTER_CUBIC
        return cv2.resize(image, (target_width, target_height), interpolation=interpolation)
    return _resize_bilinear_numpy(image, target_width, target_height)


def _resize_depth_image(depth, target_width, target_height):
    if cv2 is not None:
        interpolation = cv2.INTER_AREA
        if target_width > depth.shape[1] or target_height > depth.shape[0]:
            interpolation = cv2.INTER_CUBIC
        return cv2.resize(depth, (target_width, target_height), interpolation=interpolation)
    resized = _resize_bilinear_numpy(depth[..., None], target_width, target_height)
    return resized[..., 0]


def _fit_image_to_size(image: np.ndarray, target_width: int, target_height: int) -> tuple[np.ndarray, str]:
    """Center-crop or zero-pad an image to the requested size."""
    src_height, src_width = image.shape[:2]
    out = image
    actions: list[str] = []

    if src_width > target_width:
        left = (src_width - target_width) // 2
        out = out[:, left : left + target_width]
        actions.append(f"crop_w({src_width}->{target_width})")
    if src_height > target_height:
        top = (src_height - target_height) // 2
        out = out[top : top + target_height, :]
        actions.append(f"crop_h({src_height}->{target_height})")

    pad_height = max(0, target_height - out.shape[0])
    pad_width = max(0, target_width - out.shape[1])
    if pad_height > 0 or pad_width > 0:
        pre_pad_h, pre_pad_w = out.shape[:2]
        pad_top = pad_height // 2
        pad_bottom = pad_height - pad_top
        pad_left = pad_width // 2
        pad_right = pad_width - pad_left
        out = np.pad(
            out,
            ((pad_top, pad_bottom), (pad_left, pad_right), (0, 0)),
            mode="constant",
            constant_values=0,
        )
        pad_parts = []
        if pad_width > 0:
            pad_parts.append(f"pad_w({pre_pad_w}->{target_width})")
        if pad_height > 0:
            pad_parts.append(f"pad_h({pre_pad_h}->{target_height})")
        actions.extend(pad_parts)

    if out.shape[0] != target_height or out.shape[1] != target_width:
        raise ValueError(
            f"Failed to fit image to {(target_height, target_width)}; got {out.shape[:2]}"
        )

    if not actions:
        actions.append("ok")
    return out, "+".join(actions)


class DepthAnythingOnnxEstimator:
    def __init__(
        self,
        onnx_path=DEFAULT_DEPTH_ONNX_PATH,
        input_width=DEFAULT_DEPTH_INPUT_WIDTH,
        input_height=DEFAULT_DEPTH_INPUT_HEIGHT,
        input_mean=DEFAULT_DEPTH_INPUT_MEAN,
        input_std=DEFAULT_DEPTH_INPUT_STD,
        device=DEFAULT_DEPTH_DEVICE,
    ):
        onnx_path = Path(onnx_path).expanduser().resolve()
        if not onnx_path.exists():
            raise FileNotFoundError(f"Depth ONNX model does not exist: {onnx_path}")
        if ort is None and cv2 is None:
            raise ImportError(
                "onnxruntime or opencv-python is required for ONNX depth inference."
            ) from ONNXRUNTIME_IMPORT_ERROR

        self.onnx_path = onnx_path
        self.input_width = int(input_width)
        self.input_height = int(input_height)
        self.input_mean = np.asarray(input_mean, dtype=np.float32).reshape(1, 1, 3)
        self.input_std = np.asarray(input_std, dtype=np.float32).reshape(1, 1, 3)
        self.device = str(device).lower()
        self.runtime = None
        self.session = None
        self.net = None

        if ort is not None:
            available_providers = ort.get_available_providers()
            providers = ["CPUExecutionProvider"]
            if self.device in {"auto", "cuda", "gpu"} and "CUDAExecutionProvider" in available_providers:
                providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
            self.session = ort.InferenceSession(str(self.onnx_path), providers=providers)
            self.input_name = self.session.get_inputs()[0].name
            self.output_name = self.session.get_outputs()[0].name
            self.runtime = "onnxruntime"
            self.providers = providers
        else:
            self.net = cv2.dnn.readNetFromONNX(str(self.onnx_path))
            self.runtime = "opencv_dnn"
            self.providers = ["opencv_dnn_cpu"]

    def _preprocess_bgr(self, rgb_image):
        if rgb_image is None:
            raise ValueError("rgb_image cannot be None")
        if rgb_image.ndim != 3 or rgb_image.shape[2] != 3:
            raise ValueError(
                f"Expected an RGB image with shape (H, W, 3), got {rgb_image.shape}"
        )

        rgb = np.asarray(rgb_image)
        rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
        fitted_rgb, action = _fit_image_to_size(rgb, self.input_width, self.input_height)
        print(
            f"[depth_input] src={rgb.shape[1]}x{rgb.shape[0]} "
            f"target={self.input_width}x{self.input_height} action={action}"
        )
        rgb = fitted_rgb.astype(np.float32, copy=False) / 255.0
        rgb = (rgb - self.input_mean) / self.input_std
        rgb = np.transpose(rgb, (2, 0, 1))[None, ...]
        return np.ascontiguousarray(rgb, dtype=np.float32)

    def predict_depth(self, rgb_image):
        input_tensor = self._preprocess_bgr(rgb_image)
        if self.runtime == "onnxruntime":
            outputs = self.session.run([self.output_name], {self.input_name: input_tensor})
            depth = outputs[0]
        elif self.runtime == "opencv_dnn":
            self.net.setInput(input_tensor)
            depth = self.net.forward()
        else:  # pragma: no cover
            raise RuntimeError(f"Unsupported ONNX runtime backend: {self.runtime}")

        depth = np.asarray(depth, dtype=np.float32)
        depth = np.squeeze(depth)
        if depth.ndim != 2:
            raise ValueError(
                f"Expected depth output to be 2D after squeeze, got shape {depth.shape}"
            )
        return depth


class DiffsimDepthEstimateRacer(DiffSimRacer):
    def __init__(
        self,
        depth_onnx_path=DEFAULT_DEPTH_ONNX_PATH,
        depth_input_width=DEFAULT_DEPTH_INPUT_WIDTH,
        depth_input_height=DEFAULT_DEPTH_INPUT_HEIGHT,
        depth_device=DEFAULT_DEPTH_DEVICE,
        **kwargs,
    ):
        self.depth_estimator = DepthAnythingOnnxEstimator(
            onnx_path=depth_onnx_path,
            input_width=depth_input_width,
            input_height=depth_input_height,
            device=depth_device,
        )
        super().__init__(**kwargs)

    def get_sensor_images(self):
        responses = self.airsim_client_images.simGetImages(
            [
                airsim.ImageRequest(
                    "fpv_cam",
                    airsim.ImageType.Scene,
                    pixels_as_float=False,
                    compress=False,
                ),
                airsim.ImageRequest(
                    "fpv_cam",
                    airsim.ImageType.Segmentation,
                    pixels_as_float=False,
                    compress=False,
                ),
            ],
            vehicle_name=self.drone_name,
        )
        if len(responses) < 2:
            return None, None, None, None

        rgb_response = responses[0]
        segmentation_response = responses[1]
        if (
            rgb_response.width <= 0
            or rgb_response.height <= 0
            or segmentation_response.width <= 0
            or segmentation_response.height <= 0
        ):
            return None, None, None, None

        rgb = np.frombuffer(rgb_response.image_data_uint8, dtype=np.uint8).copy()
        rgb = rgb.reshape(rgb_response.height, rgb_response.width, 3)
        depth = self.depth_estimator.predict_depth(rgb)

        segmentation = np.frombuffer(
            segmentation_response.image_data_uint8, dtype=np.uint8
        ).copy()
        segmentation = segmentation.reshape(
            segmentation_response.height, segmentation_response.width, 3
        )

        return depth, segmentation, rgb, segmentation_response


def build_args():
    parser = ArgumentParser(
        description="Run the depth-based racer using RGB-to-depth ONNX inference."
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
        "--control_mode", type=str, choices=["velocity", "attitude"], default="attitude"
    )
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
    parser.add_argument("--viz_rgb", dest="viz_rgb", action="store_true", default=False)
    parser.add_argument(
        "--viz_depth", dest="viz_depth", action="store_true", default=False
    )
    parser.add_argument(
        "--viz_depth_raw",
        dest="viz_depth_raw",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--viz_segmentation",
        dest="viz_segmentation",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--viz_segmentation_map",
        dest="viz_segmentation_map",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--viz_gate_mask",
        dest="viz_gate_mask",
        action="store_true",
        default=False,
    )
    return parser.parse_args()


def main():
    args = build_args()

    racer = DiffsimDepthEstimateRacer(
        depth_onnx_path=args.depth_onnx_path,
        depth_input_width=args.depth_input_width,
        depth_input_height=args.depth_input_height,
        depth_device=args.depth_device,
        drone_name=args.drone_name,
        viz_rgb=args.viz_rgb,
        viz_depth=args.viz_depth,
        viz_depth_raw=args.viz_depth_raw,
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

    racer.load_level(args.level_name)
    racer.start_race(args.race_tier)
    racer.initialize_drone()
    racer.takeoff(takeoff_height=args.takeoff_height)
    if args.target_source == "ground_truth":
        racer.get_ground_truth_gate_poses()
        racer.fly_through_first_gate_with_spline()
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
