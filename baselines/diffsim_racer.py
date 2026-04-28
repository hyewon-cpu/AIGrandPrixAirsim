from argparse import ArgumentParser
import math
import os
import threading
import time

import airsimdroneracinglab as airsim
import numpy as np

import cv2 
import torch 
from torch import nn
import torch.nn.functional as F


def configure_qt_fontdir():
    if os.environ.get("QT_QPA_FONTDIR"): #if the environment variable QT_QPA_FONTDIR is already set
        #QT_QPA_FONTDIR : environment variable used by Qt's QPA layer to specify directories for font files. 
        return
    candidate_dirs = [ #font directories 
        "/usr/share/fonts/truetype/dejavu",
        "/usr/share/fonts/dejavu",
        "/usr/local/share/fonts",
    ]
    for font_dir in candidate_dirs:
        if os.path.isdir(font_dir): #if direcotiry exists 
            os.environ["QT_QPA_FONTDIR"] = font_dir #set it as the environment variable's directory 
            return


configure_qt_fontdir()


def quaternion_to_rotation_matrix(q):
    w, x, y, z = q
    n = w * w + x * x + y * y + z * z #sqaured norm 
    if n < 1e-12: #if norm is too small, it means the quaternion may not represent a valid rotation. 
        return np.eye(3, dtype=np.float32) #return identity matrix 
    s = 2.0 / n
    wx, wy, wz = s * w * x, s * w * y, s * w * z
    xx, xy, xz = s * x * x, s * x * y, s * x * z
    yy, yz, zz = s * y * y, s * y * z, s * z * z
    return np.array(
        [
            [1.0 - (yy + zz), xy - wz, xz + wy],
            [xy + wz, 1.0 - (xx + zz), yz - wx],
            [xz - wy, yz + wx, 1.0 - (xx + yy)],
        ],
        dtype=np.float32,
    )


def rotation_matrix_to_euler_zyx(rot):
    yaw = math.atan2(rot[1, 0], rot[0, 0])
    pitch = math.asin(-max(-1.0, min(1.0, rot[2, 0])))
    roll = math.atan2(rot[2, 1], rot[2, 2])
    return roll, pitch, yaw


def normalize(vec, eps=1e-6):
    norm = np.linalg.norm(vec)
    if norm < eps:
        return vec.copy()
    return vec / norm


def compute_pinhole_intrinsics(width, height, fov_degrees):
    fov_radians = math.radians(float(fov_degrees))
    fx = 0.5 * float(width) / math.tan(0.5 * fov_radians)
    fy = fx
    cx = 0.5 * float(width)
    cy = 0.5 * float(height)
    return fx, fy, cx, cy


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_MODEL_PATH = os.path.join(
    SCRIPT_DIR, "race_model", "racing1", "checkpoint0004.pth"
)
AIRSIM_TO_FLIGHTMARE = np.diag([1.0, -1.0, -1.0]).astype(np.float32)


def airsim_to_flightmare_vector(vec):
    return AIRSIM_TO_FLIGHTMARE @ np.asarray(vec, dtype=np.float32)


def flightmare_to_airsim_vector(vec):
    return AIRSIM_TO_FLIGHTMARE @ np.asarray(vec, dtype=np.float32)


def airsim_to_flightmare_rotation(rot):
    rot = np.asarray(rot, dtype=np.float32)
    return AIRSIM_TO_FLIGHTMARE @ rot @ AIRSIM_TO_FLIGHTMARE


def flightmare_to_airsim_rotation(rot):
    rot = np.asarray(rot, dtype=np.float32)
    return AIRSIM_TO_FLIGHTMARE @ rot @ AIRSIM_TO_FLIGHTMARE


if nn is not None:
    class Model(nn.Module):
        def __init__(self, dim_obs=10, dim_action=6) -> None:
            super().__init__()
            self.stem = nn.Sequential(
                nn.Conv2d(1, 32, 2, 2, bias=False),
                nn.LeakyReLU(0.05),
                nn.Conv2d(32, 64, 3, bias=False),
                nn.LeakyReLU(0.05),
                nn.Conv2d(64, 128, 3, bias=False),
                nn.LeakyReLU(0.05),
                nn.Flatten(),
                nn.Linear(128 * 2 * 4, 192, bias=False),
            )
            self.dim_obs = dim_obs
            self.observation_fc = nn.Linear(dim_obs, 192)
            self.gru = nn.GRUCell(192, 192)
            self.action_fc = nn.Linear(192, dim_action, bias=False)
            self.activation = nn.LeakyReLU(0.05)

        def forward(self, x: torch.Tensor, v, hx=None):
            img_feat = self.stem(x)
            x = self.activation(img_feat + self.observation_fc(v))
            hx = self.gru(x, hx)
            action = self.action_fc(self.activation(hx))
            return action, hx


class LearnedAccelModel:
    def __init__(self, model_path, dim_obs=10, dim_action=6, device="cpu"):
        if torch is None:
            raise ImportError(
                "torch is required to run the learned controller. "
                "Install PyTorch in this environment first."
            ) from TORCH_IMPORT_ERROR
        self.device = torch.device(device)
        self.model = Model(dim_obs=dim_obs, dim_action=dim_action).to(self.device)
        checkpoint = torch.load(model_path, map_location=self.device)
        if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        else:
            state_dict = checkpoint
        self.model.load_state_dict(state_dict, strict=False)
        self.model.eval()
        self.hidden = None

    @torch.no_grad()
    def predict_action(self, depth_tensor, state_tensor):
        depth_tensor = depth_tensor.to(self.device)
        state_tensor = state_tensor.to(self.device)
        action, self.hidden = self.model(depth_tensor, state_tensor, self.hidden)
        action = action.reshape(-1).detach().cpu().numpy()
        if action.shape[0] < 6:
            raise ValueError(
                "Model output must contain 6 values so it can be reshaped to (3, 2)."
            )
        return action


class DiffSimRacer:
    def __init__(
        self,
        drone_name="drone_1",
        viz_rgb=False,
        viz_depth=False,
        viz_depth_raw=False,
        viz_segmentation=False,
        viz_segmentation_map=False,
        viz_gate_mask=False,
        control_mode="velocity",
        control_period=0.05,
        image_period=0.05,
        sync_control_to_depth: bool = False,
        sync_depth_timeout_sec: float = 0.5,
        sync_viz_to_control: bool = False,
        sync_viz_timeout_sec: float = 0.05,
        hover_throttle=1.0,
        target_speed=7.0,
        model_path=None,
        dim_obs=10,
        dim_action=6,
        device="cpu",
        no_odom=False,
        margin=0.2,
        post_takeoff_delay=1.0,
        max_velocity=8.0,
        max_vertical_velocity=3.0,
        velocity_gain_xy=2.0,
        velocity_gain_z=2.0,
        target_source="segmentation",
        target_rgb=(134, 93, 210),
        segmentation_promote_depth_threshold=2.0,
        debug_print=False,
        debug_print_every=10,
    ):
        self.drone_name = drone_name
        self.viz_rgb = viz_rgb
        self.viz_depth = viz_depth
        self.viz_depth_raw = viz_depth_raw
        self.viz_segmentation = viz_segmentation
        self.viz_segmentation_map = viz_segmentation_map or viz_segmentation
        self.viz_gate_mask = viz_gate_mask or viz_segmentation
        self.control_mode = control_mode
        self.control_period = control_period
        self.image_period = image_period
        self.sync_control_to_depth = bool(sync_control_to_depth)
        self.sync_depth_timeout_sec = float(sync_depth_timeout_sec)
        self.sync_viz_to_control = bool(sync_viz_to_control)
        self.sync_viz_timeout_sec = float(sync_viz_timeout_sec)
        self.hover_throttle = hover_throttle
        self.target_speed = target_speed
        self.gravity = 9.81
        self.no_odom = no_odom
        self.margin = margin
        self.post_takeoff_delay = post_takeoff_delay
        self.max_velocity = max_velocity
        self.max_vertical_velocity = max_vertical_velocity
        self.velocity_gain_xy = velocity_gain_xy
        self.velocity_gain_z = velocity_gain_z
        self.target_source = target_source
        self.target_rgb = np.array(target_rgb, dtype=np.uint8)
        self.segmentation_promote_depth_threshold = float(
            segmentation_promote_depth_threshold
        )
        self.debug_print = debug_print
        self.debug_print_every = max(1, int(debug_print_every))
        self.debug_counter = 0
        self.last_depth = None
        self.last_depth_timestamp = 0.0
        self.last_depth_frame_id = 0
        self.last_control_depth_frame_id = -1
        self._depth_fps_last_time = None
        self._depth_fps_last_frame_id = 0
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
        self.is_image_thread_active = False
        self.is_control_thread_active = False
        self._sensor_cond = threading.Condition() #creates condition object that has methods like : notify_all(), notify(), wait() 
        self._viz_overlay_cond = threading.Condition()
        self._viz_overlay_by_frame_id: dict[int, dict] = {}
        self._viz_overlay_active: dict | None = None
        self.level_name = None
        self.gate_poses_ground_truth = []
        self.gate_names_ground_truth = []
        self.MAX_NUMBER_OF_GETOBJECTPOSE_TRIALS = 10
        self.active_gate_index = 0

        self.airsim_client = airsim.MultirotorClient()
        self.airsim_client.confirmConnection()
        self.airsim_client_images = airsim.MultirotorClient()
        self.airsim_client_images.confirmConnection()
        self.airsim_client_odom = airsim.MultirotorClient()
        self.airsim_client_odom.confirmConnection()

        self.model = None
        if model_path:
            self.model = LearnedAccelModel(
                model_path,
                dim_obs=dim_obs,
                dim_action=dim_action,
                device=device,
            )

        self.image_callback_thread = threading.Thread(
            target=self.repeat_timer_image_callback, args=(self.image_callback, image_period)
        )
        self.control_loop_thread = threading.Thread(
            target=self.repeat_timer_control_callback,
            args=(self.control_callback, control_period),
        )

    def load_level(self, level_name, sleep_sec=2.0):
        self.level_name = level_name
        self.airsim_client.simLoadLevel(level_name)
        self.airsim_client.confirmConnection()
        time.sleep(sleep_sec)

    def start_race(self, tier=1):
        self.airsim_client.simStartRace(tier)

    def reset_race(self):
        self.airsim_client.simResetRace()

    def initialize_drone(self):
        self.airsim_client.enableApiControl(vehicle_name=self.drone_name)
        self.airsim_client.arm(vehicle_name=self.drone_name)

        traj_tracker_gains = airsim.TrajectoryTrackerGains( #how strongly the simulator should correct position/velocity/yaw errors while following a spline
            kp_cross_track=5.0,
            kd_cross_track=0.0,
            kp_vel_cross_track=3.0,
            kd_vel_cross_track=0.0,
            kp_along_track=0.4,
            kd_along_track=0.0,
            kp_vel_along_track=0.04,
            kd_vel_along_track=0.0,
            kp_z_track=2.0,
            kd_z_track=0.0,
            kp_vel_z=0.4,
            kd_vel_z=0.0,
            kp_yaw=3.0,
            kd_yaw=0.1,
        )
        self.airsim_client.setTrajectoryTrackerGains(
            traj_tracker_gains, vehicle_name=self.drone_name
        )
        time.sleep(0.2)

    def takeoff(self, takeoff_height=1.0):
        start_position = self.airsim_client.simGetVehiclePose(
            vehicle_name=self.drone_name
        ).position #current position at that moment 

      
        print("[takeoff] starting")
        takeoff_waypoint = airsim.Vector3r(
            start_position.x_val,
            start_position.y_val,
            start_position.z_val - takeoff_height,
        )
        waypoints = [takeoff_waypoint]
        print("[takeoff] waypoints set")

        self.airsim_client.moveOnSplineAsync(
            waypoints,
            vel_max=5.0,
            acc_max=2.0,
            add_position_constraint=True,
            add_velocity_constraint=False,
            add_acceleration_constraint=False,
            vehicle_name=self.drone_name,
        ).join()
        print("[takeoff] called moveonspline")

        # `moveOnSplineAsync(...).join()` returns when the RPC call completes, not when the
        # drone physically reaches the waypoint. If we start sending other commands right
        # away, AirSim cancels the spline task. Wait briefly for altitude to settle.
        try:
            desired_z = float(start_position.z_val - takeoff_height)
            timeout_sec = 100.0
            tol_m = 0.15
            t_end = time.time() + timeout_sec
            while time.time() < t_end:
                pose = self.airsim_client.simGetVehiclePose(vehicle_name=self.drone_name)
                z_now = float(pose.position.z_val)
                if abs(z_now - desired_z) <= tol_m:
                    break
                time.sleep(0.05)
        except Exception:
            pass

        self.align()


    def align(self):
        # Align yaw toward the active gate (baseline moveOnSpline allocates yaw along
        # the trajectory tangent, so drone_2 typically ends up facing the first gate
        # before/while starting to move). Our learned controller may start from an
        # arbitrary spawn yaw, so do a one-time yaw alignment after takeoff.
        if self.gate_poses_ground_truth:
            print("[align] aligning started")
            try:
                current_position = self.airsim_client.simGetVehiclePose(
                    vehicle_name=self.drone_name
                ).position
                pos = np.array(
                    [current_position.x_val, current_position.y_val, current_position.z_val],
                    dtype=np.float32,
                )
                target = self.ground_truth_gate_target_point_airsim(pos)
                if target is not None:
                    delta_xy = target[:2] - pos[:2]
                    if float(np.linalg.norm(delta_xy)) > 1e-3:
                        yaw_deg = math.degrees(math.atan2(float(delta_xy[1]), float(delta_xy[0])))
                        self.airsim_client.moveToYawAsync(
                            yaw_deg, vehicle_name=self.drone_name
                        ).join()
            except Exception:
                # Best-effort; yaw alignment shouldn't block takeoff.
                print("[align]could not align")
                pass
        else: 
            print("[align] no gate_poses_ground truth")
    
    

    def get_ground_truth_gate_poses(self):
        gate_names_sorted_bad = sorted(self.airsim_client.simListSceneObjects("Gate.*"))
        gate_indices_bad = [
            int(gate_name.split("_")[0][4:]) for gate_name in gate_names_sorted_bad
        ]
        gate_indices_correct = sorted(
            range(len(gate_indices_bad)), key=lambda k: gate_indices_bad[k]
        )
        gate_names_sorted = [
            gate_names_sorted_bad[gate_idx] for gate_idx in gate_indices_correct
        ]
        self.gate_names_ground_truth = gate_names_sorted
        self.gate_poses_ground_truth = []
        for gate_name in gate_names_sorted:
            curr_pose = self.airsim_client.simGetObjectPose(gate_name)
            counter = 0
            while (
                math.isnan(curr_pose.position.x_val)
                or math.isnan(curr_pose.position.y_val)
                or math.isnan(curr_pose.position.z_val)
            ) and (counter < self.MAX_NUMBER_OF_GETOBJECTPOSE_TRIALS):
                counter += 1
                curr_pose = self.airsim_client.simGetObjectPose(gate_name)
            assert not math.isnan(curr_pose.position.x_val)
            assert not math.isnan(curr_pose.position.y_val)
            assert not math.isnan(curr_pose.position.z_val)
            self.gate_poses_ground_truth.append(curr_pose)
        self.active_gate_index = min(self.active_gate_index, max(0, len(self.gate_poses_ground_truth) - 1))

    def fly_through_first_gate_with_spline(self):
        if not self.gate_poses_ground_truth:
            self.get_ground_truth_gate_poses()
        current_position = self.airsim_client.simGetVehiclePose(
            vehicle_name=self.drone_name
        ).position
        first_gate_position = self.gate_poses_ground_truth[0].position
        intermediate_waypoint = airsim.Vector3r(
            current_position.x_val,
            current_position.y_val,
            min(current_position.z_val, first_gate_position.z_val) - 1.0,
        )

        if self.level_name == "Building99_Hard":
            vel_max = 5.0
            acc_max = 2.0
        else:
            vel_max = 10.0
            acc_max = 5.0

        self.airsim_client.moveOnSplineAsync(
            [intermediate_waypoint, first_gate_position],
            vel_max=vel_max,
            acc_max=acc_max,
            add_position_constraint=True,
            add_velocity_constraint=False,
            add_acceleration_constraint=False,
            vehicle_name=self.drone_name,
        ).join()
        if len(self.gate_poses_ground_truth) > 1:
            self.active_gate_index = 1
        else:
            self.active_gate_index = 0

    def start_model_control(self):
        time.sleep(self.post_takeoff_delay)
        self.start_threads()

    def get_sensor_images(self):
        request = [
            airsim.ImageRequest(
                "fpv_cam",
                airsim.ImageType.DepthPerspective,
                pixels_as_float=True,
                compress=False,
            ),
            airsim.ImageRequest(
                "fpv_cam",
                airsim.ImageType.Segmentation,
                pixels_as_float=False,
                compress=False,
            ),
        ]
        if self.viz_rgb:
            request.append(
                airsim.ImageRequest(
                    "fpv_cam",
                    airsim.ImageType.Scene,
                    pixels_as_float=False,
                    compress=False,
                )
            )
        responses = self.airsim_client_images.simGetImages(
            request, vehicle_name=self.drone_name
        )
        if len(responses) < 2:
            return None, None, None, None

        depth_response = responses[0]
        segmentation_response = responses[1]

        depth = None
        if depth_response.width > 0 and depth_response.height > 0:
            depth = np.array(depth_response.image_data_float, dtype=np.float32)
            depth = depth.reshape(depth_response.height, depth_response.width)

        segmentation = None
        if segmentation_response.width > 0 and segmentation_response.height > 0:
            segmentation = np.frombuffer(
                segmentation_response.image_data_uint8, dtype=np.uint8
            )
            segmentation = segmentation.reshape(
                segmentation_response.height, segmentation_response.width, 3
            )

        rgb = None
        if len(responses) > 2:
            rgb_response = responses[2]
            if rgb_response.width > 0 and rgb_response.height > 0:
                rgb = np.frombuffer(
                    rgb_response.image_data_uint8, dtype=np.uint8
                ).copy()
                rgb = rgb.reshape(rgb_response.height, rgb_response.width, 3)

        return depth, segmentation, rgb, segmentation_response

    def get_depth_image(self):
        depth, _, _, _ = self.get_sensor_images()
        return depth

    def get_camera_intrinsics(self, width, height):
        if (
            self.camera_intrinsics is not None
            and self.camera_intrinsics["width"] == width
            and self.camera_intrinsics["height"] == height
        ):
            return self.camera_intrinsics

        camera_info = self.airsim_client_images.simGetCameraInfo(
            "fpv_cam", vehicle_name=self.drone_name
        )
        fov_degrees = float(camera_info.fov)
        if not math.isfinite(fov_degrees) or fov_degrees <= 0.0:
            fov_degrees = 90.0
        fx, fy, cx, cy = compute_pinhole_intrinsics(width, height, fov_degrees)
        self.camera_intrinsics = {
            "width": width,
            "height": height,
            "fov_degrees": fov_degrees,
            "fx": fx,
            "fy": fy,
            "cx": cx,
            "cy": cy,
        }
        return self.camera_intrinsics

    def get_nominal_gate_dimensions(self):
        gate_dims = self.airsim_client.simGetNominalGateOuterDimensions()
        return float(gate_dims.x_val), float(gate_dims.z_val)

    def get_active_gate_dimensions(self):
        width_m, height_m = self.get_nominal_gate_dimensions()

        if (
            self.gate_names_ground_truth
            and 0 <= self.active_gate_index < len(self.gate_names_ground_truth)
        ):
            gate_name = self.gate_names_ground_truth[self.active_gate_index]
            try:
                gate_scale = self.airsim_client.simGetObjectScale(gate_name)
                width_m *= float(gate_scale.x_val)
                height_m *= float(gate_scale.z_val)
            except Exception:
                pass

        return width_m, height_m

    def build_segmentation_mask(self, segmentation, target_rgb=None):
        if segmentation is None:
            return None
        if target_rgb is None:
            target_rgb = self.target_rgb
        target_rgb = np.asarray(target_rgb, dtype=np.uint8)
        return np.all(segmentation == target_rgb, axis=2).astype(np.uint8)

    def extract_segmentation_candidates_from_mask(self, mask, depth_image=None):
        if mask is None:
            return []

        if cv2 is not None:
            mask_u8 = (mask.astype(np.uint8) > 0).astype(np.uint8)
            if not np.any(mask_u8):
                return []

            num_labels, labels, stats, _centroids = cv2.connectedComponentsWithStats(
                mask_u8, connectivity=8
            )
            candidates = []
            for label in range(1, num_labels):
                component_area = int(stats[label, cv2.CC_STAT_AREA])
                if component_area <= 0:
                    continue

                component_mask = labels == label
                component_u8 = component_mask.astype(np.uint8) * 255
                contours_info = cv2.findContours(
                    component_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                )
                contours = (
                    contours_info[0] if len(contours_info) == 2 else contours_info[1]
                )
                if not contours:
                    continue

                contour = max(contours, key=cv2.contourArea)
                if cv2.contourArea(contour) <= 0.0:
                    continue

                center, rect_size, angle = cv2.minAreaRect(contour)
                rect_w = float(rect_size[0])
                rect_h = float(rect_size[1])
                if rect_w <= 1e-6 or rect_h <= 1e-6:
                    x, y, w, h = cv2.boundingRect(contour)
                    center = (float(x + w * 0.5), float(y + h * 0.5))
                    rect_w = float(w)
                    rect_h = float(h)
                    angle = 0.0

                depth_m = np.inf
                depth_mean_m = np.inf
                depth_samples = 0
                if (
                    depth_image is not None
                    and depth_image.shape[:2] == mask.shape[:2]
                ):
                    depth_values = depth_image[component_mask]
                    depth_values = depth_values[
                        np.isfinite(depth_values) & (depth_values > 0.0)
                    ]
                    if depth_values.size > 0:
                        depth_m = float(np.median(depth_values))
                        depth_mean_m = float(np.mean(depth_values))
                        depth_samples = int(depth_values.size)

                candidates.append(
                    {
                        "center": np.array(center, dtype=np.float32),
                        "size": np.array([rect_w, rect_h], dtype=np.float32),
                        "angle": float(angle),
                        "area": float(cv2.contourArea(contour)),
                        "component_area": float(component_area),
                        "depth_m": depth_m,
                        "depth_mean_m": depth_mean_m,
                        "depth_samples": int(depth_samples),
                        "label": int(label),
                        "selection_mode": "depth" if np.isfinite(depth_m) else "area",
                    }
                )

            if not candidates:
                return []

            depth_candidates = [c for c in candidates if np.isfinite(c["depth_m"])]
            if depth_candidates:
                candidates.sort(
                    key=lambda c: (
                        c["depth_m"],
                        -c["component_area"],
                        -c["area"],
                    )
                )
            else:
                candidates.sort(key=lambda c: (-c["component_area"], -c["area"]))
            for candidate in candidates:
                candidate["component_count"] = int(len(candidates))
            return candidates

        ys, xs = np.where(mask > 0)
        if xs.size == 0:
            return []
        depth_m = np.inf
        depth_mean_m = np.inf
        depth_samples = 0
        if depth_image is not None and depth_image.shape[:2] == mask.shape[:2]:
            depth_values = depth_image[mask > 0]
            depth_values = depth_values[np.isfinite(depth_values) & (depth_values > 0.0)]
            if depth_values.size > 0:
                depth_m = float(np.median(depth_values))
                depth_mean_m = float(np.mean(depth_values))
                depth_samples = int(depth_values.size)
        x_min = int(xs.min())
        x_max = int(xs.max())
        y_min = int(ys.min())
        y_max = int(ys.max())
        candidate = {
            "center": np.array(
                [0.5 * (x_min + x_max), 0.5 * (y_min + y_max)], dtype=np.float32
            ),
            "size": np.array(
                [float(x_max - x_min + 1), float(y_max - y_min + 1)],
                dtype=np.float32,
            ),
            "angle": 0.0,
            "area": float(xs.size),
            "component_area": float(xs.size),
            "depth_m": depth_m,
            "depth_mean_m": depth_mean_m,
            "depth_samples": int(depth_samples),
            "component_count": 1,
            "selection_mode": "depth" if np.isfinite(depth_m) else "area",
        }
        return [candidate]

    def extract_rectangle_from_mask(self, mask, depth_image=None):
        candidates = self.extract_segmentation_candidates_from_mask(mask, depth_image)
        if not candidates:
            return None
        return dict(candidates[0])

    def choose_segmentation_target_candidate(self, candidates):
        if not candidates:
            return None, None, None, None, None, False

        primary_candidate = candidates[0]
        primary_depth = float(primary_candidate.get("depth_m", np.inf))
        promoted = (
            np.isfinite(primary_depth)
            and primary_depth <= self.segmentation_promote_depth_threshold
            and len(candidates) > 1
        )
        selected_index = 1 if promoted else 0
        selected_candidate = candidates[selected_index]
        backup_index = min(selected_index + 1, len(candidates) - 1)
        backup_candidate = candidates[backup_index]
        return (
            primary_candidate,
            selected_candidate,
            backup_candidate,
            selected_index,
            backup_index,
            promoted,
        )

    def segmentation_candidate_to_target_point_airsim(
        self, candidate, intr, segmentation_response, gate_dimensions=None
    ):
        if candidate is None:
            return None, np.inf, "unavailable"

        rect_w_px = float(candidate["size"][0])
        rect_h_px = float(candidate["size"][1])
        if rect_w_px <= 1e-6 or rect_h_px <= 1e-6:
            return None, np.inf, "unavailable"

        depth_m = float(candidate.get("depth_m", np.inf))
        depth_source = "depth_map"
        if not np.isfinite(depth_m) or depth_m <= 1e-6:
            if gate_dimensions is None:
                gate_dimensions = self.get_nominal_gate_dimensions()
            gate_width_m, gate_height_m = gate_dimensions
            depth_candidates = []
            if gate_width_m > 1e-6:
                depth_candidates.append(intr["fx"] * gate_width_m / rect_w_px)
            if gate_height_m > 1e-6:
                depth_candidates.append(intr["fy"] * gate_height_m / rect_h_px)
            if not depth_candidates:
                return None, np.inf, "unavailable"
            depth_m = float(np.mean(depth_candidates))
            depth_source = "nominal_gate_size"

        center_u, center_v = float(candidate["center"][0]), float(candidate["center"][1])
        x_off = (center_u - intr["cx"]) * depth_m / intr["fx"]
        y_off = (center_v - intr["cy"]) * depth_m / intr["fy"]
        # AirSim body frame: x forward, y right, z down.
        target_rel_camera = np.array([depth_m, x_off, y_off], dtype=np.float32)

        camera_position = np.array(
            [
                segmentation_response.camera_position.x_val,
                segmentation_response.camera_position.y_val,
                segmentation_response.camera_position.z_val,
            ],
            dtype=np.float32,
        )
        camera_orientation = np.array(
            [
                segmentation_response.camera_orientation.w_val,
                segmentation_response.camera_orientation.x_val,
                segmentation_response.camera_orientation.y_val,
                segmentation_response.camera_orientation.z_val,
            ],
            dtype=np.float32,
        )
        camera_rot = quaternion_to_rotation_matrix(camera_orientation)
        p_target_airsim = camera_position + camera_rot @ target_rel_camera
        return p_target_airsim, depth_m, depth_source

    def get_cached_segmentation_target_point_airsim(self):
        if self.last_segmentation_target_airsim is not None:
            return self.last_segmentation_target_airsim, {
                "segmentation_depth_source": "cached_primary",
                "segmentation_target_cache_used": True,
                "segmentation_target_cache_rank": 1,
            }
        if self.last_segmentation_backup_target_airsim is not None:
            return self.last_segmentation_backup_target_airsim, {
                "segmentation_depth_source": "cached_secondary",
                "segmentation_target_cache_used": True,
                "segmentation_target_cache_rank": 2,
            }
        return None, {}

    def estimate_segmentation_target_point_airsim(self):
        if self.last_segmentation is None or self.last_segmentation_response is None:
            return self.get_cached_segmentation_target_point_airsim()

        mask = self.last_segmentation_mask
        if mask is None:
            mask = self.build_segmentation_mask(self.last_segmentation)
        if mask is None or not np.any(mask):
            return self.get_cached_segmentation_target_point_airsim()

        width = int(self.last_segmentation_response.width)
        height = int(self.last_segmentation_response.height)
        intr = self.get_camera_intrinsics(width, height)

        candidates = self.extract_segmentation_candidates_from_mask(
            mask, depth_image=self.last_depth
        )
        if not candidates:
            return self.get_cached_segmentation_target_point_airsim()

        gate_dimensions = self.get_nominal_gate_dimensions()
        (
            primary_candidate,
            selected_candidate,
            backup_candidate,
            selected_index,
            backup_index,
            promoted,
        ) = self.choose_segmentation_target_candidate(candidates)
        if primary_candidate is None or selected_candidate is None or backup_candidate is None:
            return self.get_cached_segmentation_target_point_airsim()

        selected_target_airsim, selected_depth_m, selected_depth_source = self.segmentation_candidate_to_target_point_airsim(
            selected_candidate,
            intr,
            self.last_segmentation_response,
            gate_dimensions=gate_dimensions,
        )
        if selected_target_airsim is None:
            return self.get_cached_segmentation_target_point_airsim()

        backup_target_airsim, backup_depth_m, backup_depth_source = self.segmentation_candidate_to_target_point_airsim(
            backup_candidate,
            intr,
            self.last_segmentation_response,
            gate_dimensions=gate_dimensions,
        )
        if backup_target_airsim is None:
            backup_target_airsim = selected_target_airsim
            backup_depth_m = selected_depth_m
            backup_depth_source = selected_depth_source

        self.last_segmentation_target_airsim = selected_target_airsim
        self.last_segmentation_backup_target_airsim = backup_target_airsim
        self.last_segmentation_candidate_targets_airsim = [
            selected_target_airsim,
            backup_target_airsim,
        ]
        self.last_segmentation_candidate_timestamp = time.time()

        aux = {
            "segmentation_mask": mask,
            "segmentation_rect": selected_candidate,
            "segmentation_primary_rect": primary_candidate,
            "segmentation_backup_rect": backup_candidate,
            "segmentation_center_px": selected_candidate["center"],
            "segmentation_rect_size_px": selected_candidate["size"],
            "segmentation_depth_m": selected_depth_m,
            "segmentation_depth_source": selected_depth_source,
            "segmentation_primary_depth_m": primary_candidate.get("depth_m"),
            "segmentation_primary_rank": 1,
            "segmentation_selected_depth_m": selected_depth_m,
            "segmentation_selected_rank": selected_index + 1,
            "segmentation_backup_depth_m": backup_depth_m,
            "segmentation_backup_depth_source": backup_depth_source,
            "segmentation_backup_rank": backup_index + 1,
            "segmentation_blob_count": len(candidates),
            "segmentation_blob_selection": selected_candidate.get("selection_mode"),
            "segmentation_blob_depth_m": primary_candidate.get("depth_m"),
            "segmentation_promoted": promoted,
            "segmentation_promote_depth_threshold": self.segmentation_promote_depth_threshold,
            "segmentation_blob_backup_depth_m": backup_depth_m,
            "segmentation_target_airsim": selected_target_airsim,
            "segmentation_backup_target_airsim": backup_target_airsim,
            "camera_intrinsics": intr,
        }
        return selected_target_airsim, aux

    def ground_truth_gate_target_point_airsim(self, position_airsim):
        if not self.gate_poses_ground_truth:
            return None

        active_idx = min(self.active_gate_index, len(self.gate_poses_ground_truth) - 1)
        active_gate = self.gate_poses_ground_truth[active_idx].position
        gate_pos_airsim = np.array(
            [active_gate.x_val, active_gate.y_val, active_gate.z_val], dtype=np.float32
        )

        distance = np.linalg.norm(gate_pos_airsim - position_airsim)
        if distance < 2.0 and active_idx < len(self.gate_poses_ground_truth) - 1:
            self.active_gate_index += 1
            active_idx = self.active_gate_index
            active_gate = self.gate_poses_ground_truth[active_idx].position
            gate_pos_airsim = np.array(
                [active_gate.x_val, active_gate.y_val, active_gate.z_val], dtype=np.float32
            )

        return gate_pos_airsim

    def preprocess_depth(self, depth):
        depth = depth.copy()
        depth[~np.isfinite(depth)] = 24.0
        depth[depth <= 0.0] = 24.0
        depth = 3.0 / np.clip(depth, 0.3, 24.0) - 0.6
        h, w = depth.shape
        crop_ratio = 0.82
        crop_h = max(1, int(round(h * crop_ratio)))
        crop_w = max(1, int(round(w * crop_ratio)))
        start_h = max(0, (h - crop_h) // 2)
        start_w = max(0, (w - crop_w) // 2)
        depth = depth[start_h : start_h + crop_h, start_w : start_w + crop_w]

        if torch is None:
            raise ImportError("torch is required to preprocess depth for the model.")
        depth_tensor = torch.as_tensor(depth, dtype=torch.float32)[None, None]
        depth_tensor = F.interpolate(depth_tensor, (48, 64), mode="area")
        depth_tensor = F.max_pool2d(depth_tensor, (4, 4))
        return depth_tensor

    def visualize_depth_input(self, depth_tensor):
        if cv2 is None:
            return
        depth_map = depth_tensor.detach().cpu().numpy()[0, 0]
        depth_map = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX)
        depth_map = depth_map.astype(np.uint8)
        depth_map = cv2.resize(
            depth_map,
            (depth_map.shape[1] * 16, depth_map.shape[0] * 16),
            interpolation=cv2.INTER_NEAREST,
        )
        depth_map = self.augment_depth_input_viz(depth_map)
        cv2.imshow("depth", depth_map)
        cv2.waitKey(1)

    def visualize_depth_raw(self, depth):
        if cv2 is None:
            return
        depth = np.clip(depth, 0.0, 50.0)
        depth_map = cv2.normalize(depth, None, 0, 255, cv2.NORM_MINMAX)
        depth_map = depth_map.astype(np.uint8)
        #depth_map = self.augment_depth_raw_viz(depth_map)
        cv2.imshow("depth_raw", depth_map)
        cv2.waitKey(1)

    def build_viz_overlay(self, aux: dict, frame_id: int) -> dict | None: #build the overlay data from control thread 
        return None

    def _put_viz_overlay(self, frame_id: int, overlay: dict) -> None: #put the overlay data into frame ids 
        with self._viz_overlay_cond:
            self._viz_overlay_by_frame_id[int(frame_id)] = overlay
            if len(self._viz_overlay_by_frame_id) > 20:
                for key in sorted(self._viz_overlay_by_frame_id.keys())[:-10]:
                    self._viz_overlay_by_frame_id.pop(key, None)
            self._viz_overlay_cond.notify_all()

    def _get_viz_overlay(self, frame_id: int, wait: bool) -> dict | None: #get the overlay data from certain frame id 
        with self._viz_overlay_cond:
            if not wait:
                # Best-effort: if the exact overlay for this frame_id isn't ready yet,
                # fall back to the most recent overlay so visualization stays responsive
                # when sync_viz_to_control is disabled.
                overlay = self._viz_overlay_by_frame_id.get(int(frame_id))
                if overlay is not None:
                    return overlay
                if not self._viz_overlay_by_frame_id:
                    return None
                latest_frame_id = max(self._viz_overlay_by_frame_id.keys())
                return self._viz_overlay_by_frame_id.get(int(latest_frame_id))
            deadline = time.time() + max(0.0, float(self.sync_viz_timeout_sec))
            while int(frame_id) not in self._viz_overlay_by_frame_id:
                remaining = deadline - time.time()
                if remaining <= 0.0:
                    break
                self._viz_overlay_cond.wait(timeout=remaining)
            return self._viz_overlay_by_frame_id.get(int(frame_id))

    def image_callback(self):
        depth, segmentation, rgb, segmentation_response = self.get_sensor_images()
        now = time.time()
        depth_updated = False
        frame_id = None

        with self._sensor_cond:
            if depth is not None:
                self.last_depth = depth
                self.last_depth_timestamp = now
                self.last_depth_frame_id += 1
                frame_id = int(self.last_depth_frame_id)
                depth_updated = True
                if self.debug_print and (self.last_depth_frame_id % self.debug_print_every == 0):
                    if self._depth_fps_last_time is not None:
                        dt = float(now - self._depth_fps_last_time)
                        frames = int(self.last_depth_frame_id - self._depth_fps_last_frame_id)
                        if dt > 1e-6 and frames > 0:
                            fps = frames / dt
                            print("[sensor]", "depth_fps=", round(fps, 2), "frame_id=", self.last_depth_frame_id)
                    self._depth_fps_last_time = float(now)
                    self._depth_fps_last_frame_id = int(self.last_depth_frame_id)
            if rgb is not None:
                self.last_rgb = rgb
            else:
                self.last_rgb = None
            if segmentation is not None:
                self.last_segmentation = segmentation
                self.last_segmentation_response = segmentation_response
                self.last_segmentation_mask = self.build_segmentation_mask(segmentation)
                self.last_segmentation_timestamp = now
            else:
                self.last_segmentation = None
                self.last_segmentation_response = None
                self.last_segmentation_mask = None

            if depth_updated:
                self._sensor_cond.notify_all() #wakes up all threads currently blocked in cond.wait() on the same Condition. 

        # Visualization uses the local copies from this callback (not shared state).
        if frame_id is None:
            frame_id = int(self.last_depth_frame_id)
        if self.sync_viz_to_control:
            self._viz_overlay_active = self._get_viz_overlay(frame_id, wait=True)
        else:
            self._viz_overlay_active = self._get_viz_overlay(frame_id, wait=False)
        if depth is not None:
            if self.viz_depth_raw:
                self.visualize_depth_raw(depth)
            if self.viz_depth:
                depth_input = self.preprocess_depth(depth)
                self.visualize_depth_input(depth_input)
        if rgb is not None:
            if self.viz_rgb and cv2 is not None:
                rgb_viz = rgb
                if hasattr(self, "swap_rb") and getattr(self, "swap_rb"):
                    # Internally some racers keep RGB for model inputs; OpenCV expects BGR for display.
                    rgb_viz = rgb_viz[..., ::-1]
                rgb_viz = self.augment_rgb_viz(rgb_viz)
                cv2.imshow("rgb", rgb_viz)
                cv2.waitKey(1)
        self._viz_overlay_active = None

        if segmentation is not None:
            did_show_segmentation = False
            if self.viz_segmentation_map and cv2 is not None:
                seg_viz = cv2.cvtColor(segmentation, cv2.COLOR_RGB2BGR)
                cv2.imshow("segmentation_map", seg_viz)
                cv2.setMouseCallback(
                    "segmentation_map", self.segmentation_map_mouse_callback
                )
                did_show_segmentation = True
            if self.viz_gate_mask and cv2 is not None:
                mask = self.build_segmentation_mask(segmentation)
                if mask is not None:
                    mask_viz = (mask.astype(np.uint8) * 255)
                    cv2.imshow("gate_mask", mask_viz)
                    did_show_segmentation = True
            if did_show_segmentation:
                cv2.waitKey(1)

    def segmentation_map_mouse_callback(self, event, x, y, flags, param):
        if cv2 is None:
            return
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        if self.last_segmentation is None:
            return
        if x < 0 or y < 0:
            return
        if y >= self.last_segmentation.shape[0] or x >= self.last_segmentation.shape[1]:
            return

        rgb = self.last_segmentation[y, x]
        print(
            "[segmentation_map]",
            "pixel=(",
            int(x),
            ",",
            int(y),
            ")",
            "rgb=",
            [int(v) for v in rgb.tolist()],
        )

    def augment_rgb_viz(self, rgb_viz):
        return rgb_viz

    def augment_depth_raw_viz(self, depth_viz):
        return depth_viz

    def augment_depth_input_viz(self, depth_viz):
        return depth_viz

    def fetch_state(self):
        drone_state = self.airsim_client_odom.getMultirotorState(
            vehicle_name=self.drone_name
        )
        kin = drone_state.kinematics_estimated
        position = np.array(
            [kin.position.x_val, kin.position.y_val, kin.position.z_val], dtype=np.float32
        )
        orientation = np.array(
            [
                kin.orientation.w_val,
                kin.orientation.x_val,
                kin.orientation.y_val,
                kin.orientation.z_val,
            ],
            dtype=np.float32,
        )
        linear_velocity = np.array(
            [
                kin.linear_velocity.x_val,
                kin.linear_velocity.y_val,
                kin.linear_velocity.z_val,
            ],
            dtype=np.float32,
        )
        return {
            "position": position,
            "orientation": orientation,
            "linear_velocity": linear_velocity,
        }

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
        segmentation_aux = {}

        gt_target_airsim = None
        if self.target_source == "ground_truth" and self.gate_poses_ground_truth:
            gt_target_airsim = self.ground_truth_gate_target_point_airsim(position)

        if self.target_source == "segmentation":
            p_target_airsim, segmentation_aux = self.estimate_segmentation_target_point_airsim()
            if p_target_airsim is not None:
                if segmentation_aux.get("segmentation_target_cache_used"):
                    cache_rank = int(segmentation_aux.get("segmentation_target_cache_rank", 2))
                    self.last_target_source = f"segmentation_cache_rank_{cache_rank}"
                else:
                    selected_rank = int(segmentation_aux.get("segmentation_selected_rank", 1))
                    if segmentation_aux.get("segmentation_promoted"):
                        self.last_target_source = f"segmentation_promoted_rank_{selected_rank}"
                    else:
                        self.last_target_source = f"segmentation_rank_{selected_rank}"
        else:
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
        aux.update(segmentation_aux)
        return state_tensor, yaw_only_rot, aux

    def infer_acceleration(self, depth, state_dict):
        if self.model is None:
            raise RuntimeError(
                "No model loaded. Pass --model_path with trained model weights."
            )
        depth_tensor = self.preprocess_depth(depth)
        state_tensor, yaw_only_rot, aux = self.build_state_tensor(state_dict)
        act = self.model.predict_action(depth_tensor, state_tensor)
        act_fm = yaw_only_rot @ act.reshape(3, -1)
        a_pred_fm = act_fm[:, 0] - act_fm[:, 1]
        a_pred_airsim = flightmare_to_airsim_vector(a_pred_fm)
        target_v_airsim = flightmare_to_airsim_vector(aux["target_v_fm"])
        return a_pred_airsim.astype(np.float32), target_v_airsim.astype(np.float32), aux

    def acceleration_to_velocity_command(self, accel_world, state_dict):
        current_velocity = state_dict["linear_velocity"].astype(np.float32)
        dt = 0.01
        accel_cmd = accel_world.astype(np.float32).copy()
        accel_cmd[0] *= self.velocity_gain_xy
        accel_cmd[1] *= self.velocity_gain_xy
        accel_cmd[2] *= self.velocity_gain_z
        velocity_cmd = current_velocity + accel_cmd * dt

        horizontal_speed = np.linalg.norm(velocity_cmd[:2])
        if horizontal_speed > self.max_velocity:
            velocity_cmd[:2] = velocity_cmd[:2] / horizontal_speed * self.max_velocity

        velocity_cmd[2] = float(
            np.clip(
                velocity_cmd[2],
                -self.max_vertical_velocity,
                self.max_vertical_velocity,
            )
        )
        return velocity_cmd

    def acceleration_to_attitude_command(self, accel_world, aux):
        a_setpoint = airsim_to_flightmare_vector(accel_world).astype(np.float32)
        a_setpoint[2] += self.gravity
        thrust_mag = float(np.linalg.norm(a_setpoint))
        if thrust_mag < 1e-6:
            current_forward = aux["env_rot_fm"][:, 0]
            yaw = math.atan2(float(current_forward[1]), float(current_forward[0]))
            return 0.0, 0.0, yaw, float(self.hover_throttle), thrust_mag

        up_vec = a_setpoint / thrust_mag
        throttle = thrust_mag + float(
            aux["local_velocity_fm"][2] * abs(aux["local_velocity_fm"][2]) * 0.01
        )

        if aux["p_target_fm"] is not None:
            target_delta = aux["p_target_fm"] - aux["position_fm"]
        else:
            target_delta = aux["target_v_fm"]
        forward_vec = aux["env_rot_fm"][:, 0] * 5.0 + target_delta
        if abs(up_vec[2]) > 1e-6:
            forward_vec[2] = -(
                forward_vec[0] * up_vec[0] + forward_vec[1] * up_vec[1]
            ) / up_vec[2]
        else:
            forward_vec[2] = 0.0
        forward_vec = normalize(forward_vec)
        left_vec = normalize(np.cross(up_vec, forward_vec))
        forward_vec = normalize(np.cross(left_vec, up_vec))

        roll = math.atan2(float(left_vec[2]), float(up_vec[2]))
        pitch = math.asin(float(np.clip(-forward_vec[2], -1.0, 1.0)))
        yaw = math.atan2(float(forward_vec[1]), float(forward_vec[0]))
        throttle = float(np.clip(throttle / 9.8 * self.hover_throttle, 0.0, 1.0))
        return float(roll), float(pitch), float(yaw), throttle, thrust_mag

    def control_callback(self):
        if self.model is None:
            return

        depth = None
        depth_frame_id = -1
        if self.sync_control_to_depth:
            deadline = time.time() + max(0.0, self.sync_depth_timeout_sec)
            with self._sensor_cond:
                while True:
                    if self.last_depth is not None and self.last_depth_frame_id != self.last_control_depth_frame_id:
                        break
                    remaining = deadline - time.time()
                    if remaining <= 0.0:
                        break
                    self._sensor_cond.wait(timeout=remaining)
                depth = self.last_depth
                depth_frame_id = self.last_depth_frame_id
        else:
            depth = self.last_depth
            depth_frame_id = self.last_depth_frame_id

        self.current_state = self.fetch_state()

        if depth is None:
            if self.debug_print and (self.debug_counter % self.debug_print_every == 0):
                print("[debug]", "target_src=", "depth_missing", "fallback=", "hover")
            self.debug_counter += 1
            self.airsim_client.hoverAsync(vehicle_name=self.drone_name)
            return

        if (not self.sync_control_to_depth) and depth_frame_id == self.last_control_depth_frame_id:
            if self.debug_print and (self.debug_counter % self.debug_print_every == 0):
                print(
                    "[debug]",
                    "target_src=",
                    "depth_reused",
                    "frame_id=",
                    depth_frame_id,
                    "fallback=",
                    "none",
                )
        try:
            accel_world, target_v, aux = self.infer_acceleration(
                depth, self.current_state
            )
            overlay = self.build_viz_overlay(aux, int(depth_frame_id))
            if overlay is not None:
                self._put_viz_overlay(int(depth_frame_id), overlay)
            current_velocity = self.current_state["linear_velocity"]
            seg_center = aux.get("segmentation_center_px")
            seg_depth = aux.get("segmentation_depth_m")
            seg_depth_source = aux.get("segmentation_depth_source")
            seg_blob_count = aux.get("segmentation_blob_count")
            seg_blob_depth = aux.get("segmentation_blob_depth_m")
            seg_blob_selection = aux.get("segmentation_blob_selection")
            seg_selected_depth = aux.get("segmentation_selected_depth_m")
            seg_promoted = aux.get("segmentation_promoted")
            seg_selected_rank = aux.get("segmentation_selected_rank")
            seg_backup_rank = aux.get("segmentation_backup_rank")
            seg_primary_rank = aux.get("segmentation_primary_rank")
            seg_promo_thr = aux.get("segmentation_promote_depth_threshold")
            seg_blob_depth_display = (
                None
                if seg_blob_depth is None or not np.isfinite(float(seg_blob_depth))
                else round(float(seg_blob_depth), 3)
            )
            seg_selected_depth_display = (
                None
                if seg_selected_depth is None or not np.isfinite(float(seg_selected_depth))
                else round(float(seg_selected_depth), 3)
            )
            seg_depth_display = (
                None
                if seg_depth is None or not np.isfinite(float(seg_depth))
                else round(float(seg_depth), 3)
            )
            depth_shape = None if self.last_depth is None else tuple(self.last_depth.shape)
            seg_mask_shape = (
                None
                if self.last_segmentation_mask is None
                else tuple(self.last_segmentation_mask.shape)
            )
            seg_img_shape = (
                None if self.last_segmentation is None else tuple(self.last_segmentation.shape)
            )
            rpg_rgb_shape = None
            if hasattr(self, "last_rgb") and self.last_rgb is not None:
                rpg_rgb_shape = tuple(self.last_rgb.shape)
            seg_rgb = self.target_rgb
            if self.control_mode == "attitude":
                roll, pitch, yaw, throttle, thrust_mag = self.acceleration_to_attitude_command(
                    accel_world, aux
                )
                if self.debug_print and (self.debug_counter % self.debug_print_every == 0):
                    print(
                        "[debug]",
                        "target_src=", self.last_target_source,
                        "target_v=", np.round(target_v, 3),
                        "a_pred=", np.round(accel_world, 3),
                        "v_cur=", np.round(current_velocity, 3),
                        "seg_center=", None if seg_center is None else np.round(seg_center, 1),
                        "seg_depth=", seg_depth_display,
                        "seg_depth_src=", seg_depth_source,
                        "seg_selected_depth=", seg_selected_depth_display,
                        "seg_blob_depth=", seg_blob_depth_display,
                        "seg_blob_selection=", seg_blob_selection,
                        "seg_blob_count=", seg_blob_count,
                        "seg_primary_rank=", seg_primary_rank,
                        "seg_selected_rank=", seg_selected_rank,
                        "seg_backup_rank=", seg_backup_rank,
                        "seg_promoted=", seg_promoted,
                        "seg_thr=", seg_promo_thr,
                        "rpg_rgb_shape=", rpg_rgb_shape,
                        "seg_img_shape=", seg_img_shape,
                        "depth_shape=", depth_shape,
                        "seg_mask_shape=", seg_mask_shape,
                        "seg_rgb=", None if seg_rgb is None else seg_rgb.tolist(),
                        "rpy=", np.round([roll, pitch, yaw], 3),
                        "throttle=", round(throttle, 3),
                        "thrust=", round(thrust_mag, 3),
                    )
                self.debug_counter += 1
                self.airsim_client.moveByRollPitchYawThrottleAsync(
                    roll,
                    pitch,
                    yaw,
                    throttle,
                    self.control_period,
                    vehicle_name=self.drone_name,
                )
                self.last_control_depth_frame_id = depth_frame_id
            else:
                velocity_cmd = self.acceleration_to_velocity_command(
                    accel_world, self.current_state
                )
                if self.debug_print and (self.debug_counter % self.debug_print_every == 0):
                    print(
                        "[debug]",
                        "target_src=", self.last_target_source,
                        "target_v=", np.round(target_v, 3),
                        "a_pred=", np.round(accel_world, 3),
                        "v_cur=", np.round(current_velocity, 3),
                        "seg_center=", None if seg_center is None else np.round(seg_center, 1),
                        "seg_depth=", seg_depth_display,
                        "seg_depth_src=", seg_depth_source,
                        "seg_selected_depth=", seg_selected_depth_display,
                        "seg_blob_depth=", seg_blob_depth_display,
                        "seg_blob_selection=", seg_blob_selection,
                        "seg_blob_count=", seg_blob_count,
                        "seg_primary_rank=", seg_primary_rank,
                        "seg_selected_rank=", seg_selected_rank,
                        "seg_backup_rank=", seg_backup_rank,
                        "seg_promoted=", seg_promoted,
                        "seg_thr=", seg_promo_thr,
                        "rpg_rgb_shape=", rpg_rgb_shape,
                        "seg_img_shape=", seg_img_shape,
                        "depth_shape=", depth_shape,
                        "seg_mask_shape=", seg_mask_shape,
                        "seg_rgb=", None if seg_rgb is None else seg_rgb.tolist(),
                        "v_cmd=", np.round(velocity_cmd, 3),
                    )
                self.debug_counter += 1
                self.airsim_client.moveByVelocityAsync(
                    float(velocity_cmd[0]),
                    float(velocity_cmd[1]),
                    float(velocity_cmd[2]),
                    self.control_period,
                    vehicle_name=self.drone_name,
                )
                self.last_control_depth_frame_id = depth_frame_id
        except Exception as exc:  # pragma: no cover
            print("control_callback failed:", exc)

    def repeat_timer_image_callback(self, task, period):
        while self.is_image_thread_active:
            task()
            time.sleep(period)

    def repeat_timer_control_callback(self, task, period):
        while self.is_control_thread_active:
            task()
            if not self.sync_control_to_depth:
                time.sleep(period)

    def start_threads(self):
        if not self.is_image_thread_active:
            self.is_image_thread_active = True
            self.image_callback_thread.start()
        if not self.is_control_thread_active:
            self.is_control_thread_active = True
            self.control_loop_thread.start()

    def stop_threads(self):
        if self.is_image_thread_active:
            self.is_image_thread_active = False
            self.image_callback_thread.join()
        if self.is_control_thread_active:
            self.is_control_thread_active = False
            self.control_loop_thread.join()
        if cv2 is not None:
            if self.viz_rgb:
                cv2.destroyWindow("rgb")
            if self.viz_depth_raw:
                cv2.destroyWindow("depth_raw")
            if self.viz_depth:
                cv2.destroyWindow("depth")
            if self.viz_segmentation_map:
                cv2.destroyWindow("segmentation_map")
            if self.viz_gate_mask:
                cv2.destroyWindow("gate_mask")


def main():
    parser = ArgumentParser()
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
    parser.add_argument("--sync_control_to_depth", action="store_true", default=False)
    parser.add_argument("--sync_depth_timeout_sec", type=float, default=0.5)
    parser.add_argument(
        "--model_path",
        type=str,
        default=DEFAULT_MODEL_PATH,
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
    args = parser.parse_args()

    racer = DiffSimRacer(
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
        sync_control_to_depth=args.sync_control_to_depth,
        sync_depth_timeout_sec=args.sync_depth_timeout_sec,
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
