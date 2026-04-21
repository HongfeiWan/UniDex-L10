#!/usr/bin/env python3
"""
PyBullet smoke test for the L10 right-hand URDF.

Examples:
    python scripts/test_l10_right_hand_pybullet.py
    python scripts/test_l10_right_hand_pybullet.py --no-gui
    python scripts/test_l10_right_hand_pybullet.py --wrist-camera --camera-save-dir debug_wrist_camera
    python scripts/test_l10_right_hand_pybullet.py --wrist-camera --camera-fov 45 --camera-offset 0.1 0 -0.2
"""

from __future__ import annotations

import argparse
import math
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

try:
    import pybullet as p
    import pybullet_data
except ImportError as exc:
    raise SystemExit(
        "PyBullet is not installed. Install it with "
        "`python3 -m pip install pybullet-arm64` on Apple Silicon macOS, "
        "or `python3 -m pip install pybullet` on other platforms."
    ) from exc


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_URDF = REPO_ROOT / "assets" / "right-L10hand" / "linkerhand_l10_right.urdf"
DEFAULT_HAND_BASE_POSITION = [0.0, 0.0, 0.5]
DEFAULT_HAND_BASE_EULER = [0.0, -np.pi / 2, 0.0]
DEFAULT_MOTOR_FORCE = 5.0
PRIMARY_DOF_NAMES = [
    "thumb_cmc_roll",
    "thumb_cmc_yaw",
    "thumb_cmc_pitch",
    "index_mcp_roll",
    "index_mcp_pitch",
    "middle_mcp_pitch",
    "ring_mcp_roll",
    "ring_mcp_pitch",
    "pinky_mcp_roll",
    "pinky_mcp_pitch",
]


@dataclass(frozen=True)
class JointRecord:
    index: int
    name: str
    joint_type: int
    lower: float
    upper: float
    parent_link: str
    child_link: str


@dataclass(frozen=True)
class JointSlider:
    joint: JointRecord
    slider_id: int
    default_value: float


@dataclass(frozen=True)
class WristCameraConfig:
    link_name: str
    width: int
    height: int
    fov: float
    near: float
    far: float
    offset_local: np.ndarray
    forward_axis_local: np.ndarray
    up_axis_local: np.ndarray
    target_distance: float
    log_every: int
    save_dir: Path | None
    visualize: bool
    axis_length: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Test the L10 right-hand URDF with PyBullet.")
    parser.add_argument(
        "--urdf",
        type=Path,
        default=DEFAULT_URDF,
        help="Path to the URDF file to load.",
    )
    parser.add_argument(
        "--gui",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Launch the PyBullet GUI for interactive inspection (default: enabled).",
    )
    parser.add_argument(
        "--dt",
        type=float,
        default=1.0 / 240.0,
        help="Simulation step size in seconds.",
    )
    parser.add_argument(
        "--fixed-base",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Load the hand as a fixed-base robot (default: enabled).",
    )
    parser.add_argument(
        "--wrist-camera",
        action="store_true",
        help="Attach a virtual RGB-D camera to the wrist/base and capture images.",
    )
    parser.add_argument(
        "--camera-link",
        type=str,
        default="base",
        help="Link name to attach the virtual camera to. Use 'base' for the hand base/wrist.",
    )
    parser.add_argument("--camera-width", type=int, default=320, help="Virtual camera image width.")
    parser.add_argument("--camera-height", type=int, default=240, help="Virtual camera image height.")
    parser.add_argument(
        "--camera-fov",
        type=float,
        default=50.0,
        help="Virtual camera field of view in degrees. Smaller values zoom in on the hand.",
    )
    parser.add_argument("--camera-near", type=float, default=0.01, help="Virtual camera near plane in meters.")
    parser.add_argument("--camera-far", type=float, default=1.5, help="Virtual camera far plane in meters.")
    parser.add_argument(
        "--camera-offset",
        type=float,
        nargs=3,
        default=(0.1, 0.0, -0.2),
        metavar=("X", "Y", "Z"),
        help="Camera origin offset in the chosen link's local frame, in meters. Default moves the camera closer to the fingers.",
    )
    parser.add_argument(
        "--camera-forward-axis",
        type=float,
        nargs=3,
        default=(0.0, 0.0, 1.0),
        metavar=("X", "Y", "Z"),
        help="Camera forward axis in the chosen link's local frame.",
    )
    parser.add_argument(
        "--camera-up-axis",
        type=float,
        nargs=3,
        default=(0.0, -1.0, 0.0),
        metavar=("X", "Y", "Z"),
        help="Camera up axis in the chosen link's local frame.",
    )
    parser.add_argument(
        "--camera-target-distance",
        type=float,
        default=0.1,
        help="Distance from camera origin to its look-at target, in meters.",
    )
    parser.add_argument(
        "--camera-log-every",
        type=int,
        default=120,
        help="Capture/log wrist camera output every N GUI loop steps.",
    )
    parser.add_argument(
        "--camera-save-dir",
        type=Path,
        default=None,
        help="Optional directory to save captured wrist camera frames as .npz files.",
    )
    parser.add_argument(
        "--camera-visualize",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Draw the wrist camera origin and axes in the PyBullet GUI.",
    )
    parser.add_argument(
        "--camera-axis-length",
        type=float,
        default=0.05,
        help="Length of the debug axes used to visualize the wrist camera, in meters.",
    )
    return parser.parse_args()


def decode_name(value: bytes) -> str:
    return value.decode("utf-8")


def collect_actuated_joints(body_id: int) -> list[JointRecord]:
    joint_records: list[JointRecord] = []
    for joint_index in range(p.getNumJoints(body_id)):
        info = p.getJointInfo(body_id, joint_index)
        joint_type = info[2]
        if joint_type not in (p.JOINT_REVOLUTE, p.JOINT_PRISMATIC):
            continue

        parent_index = info[16]
        parent_link = "base" if parent_index == -1 else decode_name(p.getJointInfo(body_id, parent_index)[12])
        joint_records.append(
            JointRecord(
                index=joint_index,
                name=decode_name(info[1]),
                joint_type=joint_type,
                lower=float(info[8]),
                upper=float(info[9]),
                parent_link=parent_link,
                child_link=decode_name(info[12]),
            )
        )
    return joint_records


def build_joint_lookup(joints: list[JointRecord]) -> dict[str, JointRecord]:
    return {joint.name: joint for joint in joints}


def build_link_lookup(joints: list[JointRecord]) -> dict[str, JointRecord]:
    return {joint.child_link: joint for joint in joints}


def clamp_target(joint: JointRecord, ratio: float) -> float:
    lower = joint.lower
    upper = joint.upper
    if not math.isfinite(lower) or not math.isfinite(upper) or upper <= lower:
        return 0.0
    return lower + ratio * (upper - lower)


def clamp_joint_position(joint: JointRecord, value: float) -> float:
    lower = joint.lower
    upper = joint.upper
    if math.isfinite(lower) and math.isfinite(upper) and upper > lower:
        return min(max(value, lower), upper)
    return value


def expand_l10_action_to_joint_targets(action: list[float]) -> dict[str, float]:
    if len(action) != len(PRIMARY_DOF_NAMES):
        raise ValueError(f"Expected {len(PRIMARY_DOF_NAMES)} values, got {len(action)}")

    q: dict[str, float] = {}

    q["thumb_cmc_roll"] = action[0]
    q["thumb_cmc_yaw"] = action[1]
    q["thumb_cmc_pitch"] = action[2]
    q["index_mcp_roll"] = action[3]
    q["index_mcp_pitch"] = action[4]
    q["middle_mcp_pitch"] = action[5]
    q["ring_mcp_roll"] = action[6]
    q["ring_mcp_pitch"] = action[7]
    q["pinky_mcp_roll"] = action[8]
    q["pinky_mcp_pitch"] = action[9]

    q["thumb_mcp"] = 1.3898 * q["thumb_cmc_pitch"]
    q["thumb_ip"] = 1.5080 * q["thumb_cmc_pitch"]

    q["index_pip"] = 1.3462 * q["index_mcp_pitch"]
    q["index_dip"] = 0.4616 * q["index_mcp_pitch"]

    q["middle_pip"] = 1.3462 * q["middle_mcp_pitch"]
    q["middle_dip"] = 0.4616 * q["middle_mcp_pitch"]

    q["ring_pip"] = 1.3462 * q["ring_mcp_pitch"]
    q["ring_dip"] = 0.4616 * q["ring_mcp_pitch"]

    q["pinky_pip"] = 1.3462 * q["pinky_mcp_pitch"]
    q["pinky_dip"] = 0.4616 * q["pinky_mcp_pitch"]

    return q


def apply_joint_targets(
    body_id: int,
    joint_lookup: dict[str, JointRecord],
    joint_targets: dict[str, float],
    reset_state: bool = False,
) -> None:
    for joint_name, target in joint_targets.items():
        joint = joint_lookup[joint_name]
        clamped_target = clamp_joint_position(joint, target)
        if reset_state:
            p.resetJointState(body_id, joint.index, targetValue=clamped_target)
        p.setJointMotorControl2(
            bodyIndex=body_id,
            jointIndex=joint.index,
            controlMode=p.POSITION_CONTROL,
            targetPosition=clamped_target,
            force=DEFAULT_MOTOR_FORCE,
        )


def set_rest_pose(body_id: int, joint_lookup: dict[str, JointRecord]) -> None:
    action: list[float] = []
    for order, joint_name in enumerate(PRIMARY_DOF_NAMES):
        joint = joint_lookup[joint_name]
        ratio = 0.2 + 0.6 * ((order % 4) / 3.0)
        action.append(clamp_target(joint, ratio))
    apply_joint_targets(
        body_id,
        joint_lookup,
        expand_l10_action_to_joint_targets(action),
        reset_state=True,
    )


def print_joint_summary(body_id: int, joints: list[JointRecord]) -> None:
    print(f"Loaded body id: {body_id}")
    print(f"Total joints: {p.getNumJoints(body_id)}")
    print(f"Actuated joints: {len(joints)}")
    print("Actuated joint summary:")
    for joint in joints:
        state = p.getJointState(body_id, joint.index)
        print(
            "  "
            f"[{joint.index:02d}] {joint.name:<18} "
            f"range=({joint.lower:.4f}, {joint.upper:.4f}) "
            f"state={state[0]:.4f} "
            f"{joint.parent_link} -> {joint.child_link}"
        )


def verify_link_states(body_id: int) -> None:
    for joint_index in range(p.getNumJoints(body_id)):
        link_state = p.getLinkState(body_id, joint_index, computeForwardKinematics=True)
        if link_state is None:
            raise RuntimeError(f"Failed to query link state for joint index {joint_index}")
    print("All link states queried successfully.")


def setup_gui_camera() -> None:
    p.resetDebugVisualizerCamera(
        cameraDistance=0.35,
        cameraYaw=135,
        cameraPitch=-20,
        cameraTargetPosition=[0.0, 0.0, 0.08],
    )


def step_simulation(steps: int, dt: float | None = None) -> None:
    for _ in range(steps):
        p.stepSimulation()
        if dt is not None:
            time.sleep(dt)


def validate_joint_configuration(joint_lookup: dict[str, JointRecord]) -> None:
    expected_joint_names = expand_l10_action_to_joint_targets([0.0] * len(PRIMARY_DOF_NAMES)).keys()
    missing = [joint_name for joint_name in expected_joint_names if joint_name not in joint_lookup]
    if missing:
        missing_text = ", ".join(missing)
        raise RuntimeError(f"Missing expected joints in URDF: {missing_text}")


def _normalize_vector(vector: np.ndarray, name: str) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if norm <= 1e-8:
        raise ValueError(f"{name} must be a non-zero 3D vector.")
    return vector / norm


def build_wrist_camera_config(args: argparse.Namespace) -> WristCameraConfig:
    return WristCameraConfig(
        link_name=args.camera_link,
        width=args.camera_width,
        height=args.camera_height,
        fov=args.camera_fov,
        near=args.camera_near,
        far=args.camera_far,
        offset_local=np.asarray(args.camera_offset, dtype=np.float32),
        forward_axis_local=_normalize_vector(
            np.asarray(args.camera_forward_axis, dtype=np.float32),
            "camera_forward_axis",
        ),
        up_axis_local=_normalize_vector(
            np.asarray(args.camera_up_axis, dtype=np.float32),
            "camera_up_axis",
        ),
        target_distance=args.camera_target_distance,
        log_every=max(1, args.camera_log_every),
        save_dir=args.camera_save_dir.resolve() if args.camera_save_dir is not None else None,
        visualize=args.camera_visualize,
        axis_length=args.camera_axis_length,
    )


def _quat_to_rotation_matrix(quaternion: tuple[float, float, float, float] | list[float]) -> np.ndarray:
    return np.asarray(p.getMatrixFromQuaternion(quaternion), dtype=np.float32).reshape(3, 3)


def get_link_pose(
    body_id: int,
    link_name: str,
    link_lookup: dict[str, JointRecord],
) -> tuple[np.ndarray, tuple[float, float, float, float]]:
    if link_name == "base":
        position, orientation = p.getBasePositionAndOrientation(body_id)
        return np.asarray(position, dtype=np.float32), orientation

    if link_name not in link_lookup:
        known_names = ", ".join(sorted(["base", *link_lookup.keys()]))
        raise ValueError(f"Unknown camera link '{link_name}'. Available names: {known_names}")

    link_state = p.getLinkState(
        body_id,
        link_lookup[link_name].index,
        computeForwardKinematics=True,
    )
    return np.asarray(link_state[4], dtype=np.float32), link_state[5]


def get_wrist_camera_pose(
    body_id: int,
    link_lookup: dict[str, JointRecord],
    camera_config: WristCameraConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    link_position, link_orientation = get_link_pose(body_id, camera_config.link_name, link_lookup)
    rotation = _quat_to_rotation_matrix(link_orientation)

    camera_position = link_position + rotation @ camera_config.offset_local
    camera_forward = _normalize_vector(rotation @ camera_config.forward_axis_local, "camera_forward_world")
    camera_up = _normalize_vector(rotation @ camera_config.up_axis_local, "camera_up_world")
    camera_target = camera_position + camera_forward * camera_config.target_distance
    return camera_position, camera_forward, camera_up, camera_target


def capture_wrist_camera(
    body_id: int,
    link_lookup: dict[str, JointRecord],
    camera_config: WristCameraConfig,
) -> dict[str, np.ndarray]:
    camera_position, camera_forward, camera_up, camera_target = get_wrist_camera_pose(
        body_id,
        link_lookup,
        camera_config,
    )

    aspect = camera_config.width / camera_config.height
    view_matrix = p.computeViewMatrix(
        cameraEyePosition=camera_position.tolist(),
        cameraTargetPosition=camera_target.tolist(),
        cameraUpVector=camera_up.tolist(),
    )
    projection_matrix = p.computeProjectionMatrixFOV(
        fov=camera_config.fov,
        aspect=aspect,
        nearVal=camera_config.near,
        farVal=camera_config.far,
    )
    _, _, rgba, depth_buffer, segmentation = p.getCameraImage(
        width=camera_config.width,
        height=camera_config.height,
        viewMatrix=view_matrix,
        projectionMatrix=projection_matrix,
        renderer=p.ER_TINY_RENDERER,
    )

    rgba_image = np.asarray(rgba, dtype=np.uint8).reshape(camera_config.height, camera_config.width, 4)
    depth_buffer = np.asarray(depth_buffer, dtype=np.float32).reshape(camera_config.height, camera_config.width)
    segmentation = np.asarray(segmentation, dtype=np.int32).reshape(camera_config.height, camera_config.width)

    depth_m = (
        camera_config.far
        * camera_config.near
        / (
            camera_config.far
            - (camera_config.far - camera_config.near) * np.clip(depth_buffer, 0.0, 1.0)
        )
    ).astype(np.float32)

    return {
        "camera_position": camera_position.astype(np.float32),
        "camera_target": camera_target.astype(np.float32),
        "camera_up": camera_up.astype(np.float32),
        "rgba": rgba_image,
        "rgb": rgba_image[..., :3].copy(),
        "depth_m": depth_m,
        "segmentation": segmentation,
    }


def summarize_wrist_camera_capture(capture: dict[str, np.ndarray]) -> str:
    depth = capture["depth_m"]
    finite_mask = np.isfinite(depth)
    if not np.any(finite_mask):
        return "depth contains no finite values"
    depth_values = depth[finite_mask]
    return (
        f"rgb shape={capture['rgb'].shape}, "
        f"depth range=[{depth_values.min():.4f}, {depth_values.max():.4f}] m, "
        f"camera_pos={np.round(capture['camera_position'], 4).tolist()}"
    )


def save_wrist_camera_capture(
    capture: dict[str, np.ndarray],
    save_dir: Path,
    frame_index: int,
) -> Path:
    save_dir.mkdir(parents=True, exist_ok=True)
    output_path = save_dir / f"wrist_camera_{frame_index:06d}.npz"
    np.savez_compressed(
        output_path,
        rgb=capture["rgb"],
        rgba=capture["rgba"],
        depth_m=capture["depth_m"],
        segmentation=capture["segmentation"],
        camera_position=capture["camera_position"],
        camera_target=capture["camera_target"],
        camera_up=capture["camera_up"],
    )
    return output_path


def update_wrist_camera_debug(
    body_id: int,
    link_lookup: dict[str, JointRecord],
    camera_config: WristCameraConfig,
    debug_item_ids: dict[str, int] | None = None,
) -> dict[str, int]:
    if not camera_config.visualize:
        return debug_item_ids or {}

    camera_position, camera_forward, camera_up, camera_target = get_wrist_camera_pose(
        body_id,
        link_lookup,
        camera_config,
    )
    camera_right = _normalize_vector(np.cross(camera_forward, camera_up), "camera_right_world")
    axis_length = camera_config.axis_length

    debug_item_ids = {} if debug_item_ids is None else dict(debug_item_ids)
    debug_item_ids["forward"] = p.addUserDebugLine(
        camera_position.tolist(),
        (camera_position + camera_forward * axis_length).tolist(),
        [1.0, 0.0, 0.0],
        lineWidth=3.0,
        lifeTime=0.0,
        replaceItemUniqueId=debug_item_ids.get("forward", -1),
    )
    debug_item_ids["up"] = p.addUserDebugLine(
        camera_position.tolist(),
        (camera_position + camera_up * axis_length).tolist(),
        [0.0, 1.0, 0.0],
        lineWidth=3.0,
        lifeTime=0.0,
        replaceItemUniqueId=debug_item_ids.get("up", -1),
    )
    debug_item_ids["right"] = p.addUserDebugLine(
        camera_position.tolist(),
        (camera_position + camera_right * axis_length).tolist(),
        [0.0, 0.4, 1.0],
        lineWidth=3.0,
        lifeTime=0.0,
        replaceItemUniqueId=debug_item_ids.get("right", -1),
    )
    debug_item_ids["view"] = p.addUserDebugLine(
        camera_position.tolist(),
        camera_target.tolist(),
        [1.0, 1.0, 0.0],
        lineWidth=2.0,
        lifeTime=0.0,
        replaceItemUniqueId=debug_item_ids.get("view", -1),
    )
    debug_item_ids["label"] = p.addUserDebugText(
        "wrist_camera",
        camera_position.tolist(),
        textColorRGB=[1.0, 1.0, 1.0],
        textSize=1.2,
        lifeTime=0.0,
        replaceItemUniqueId=debug_item_ids.get("label", -1),
    )
    return debug_item_ids


def create_joint_sliders(body_id: int, joint_lookup: dict[str, JointRecord]) -> list[JointSlider]:
    sliders: list[JointSlider] = []
    print("Creating 10 primary-DOF sliders in the PyBullet GUI...")
    for joint_name in PRIMARY_DOF_NAMES:
        joint = joint_lookup[joint_name]
        lower = joint.lower if math.isfinite(joint.lower) else -math.pi
        upper = joint.upper if math.isfinite(joint.upper) else math.pi
        if upper <= lower:
            lower, upper = -1.0, 1.0

        current = p.getJointState(body_id, joint.index)[0]
        default_value = min(max(current, lower), upper)
        slider_id = p.addUserDebugParameter(joint.name, lower, upper, default_value)
        if slider_id < 0:
            raise RuntimeError(
                f"Failed to create debug slider for joint `{joint.name}`. "
                "Please make sure the PyBullet GUI debug panel is available."
            )

        sliders.append(
            JointSlider(
                joint=joint,
                slider_id=slider_id,
                default_value=default_value,
            )
        )
        print(f"  slider: {joint.name:<18} [{lower:.4f}, {upper:.4f}] id={slider_id}")

    step_simulation(10, dt=0.01)
    return sliders


def read_slider_value(slider: JointSlider, fallback_value: float) -> tuple[bool, float]:
    for _ in range(5):
        try:
            return True, p.readUserDebugParameter(slider.slider_id)
        except p.error:
            time.sleep(0.01)
    return False, fallback_value


def run_gui_joint_control(
    body_id: int,
    joint_lookup: dict[str, JointRecord],
    link_lookup: dict[str, JointRecord],
    sliders: list[JointSlider],
    dt: float,
    wrist_camera: WristCameraConfig | None = None,
) -> None:
    print("GUI is running. Drag the 10 primary sliders to control the hand.")
    print("Coupled joints are updated automatically from the 10 primary sliders.")
    print("Close the PyBullet window or press Ctrl+C in the terminal to stop.")
    action = [slider.default_value for slider in sliders]
    warned_slider_ids: set[int] = set()
    frame_index = 0
    camera_debug_ids: dict[str, int] = {}
    while p.isConnected():
        for idx, slider in enumerate(sliders):
            ok, value = read_slider_value(slider, fallback_value=action[idx])
            if not ok:
                if slider.slider_id not in warned_slider_ids:
                    print(
                        f"Warning: failed to read slider `{slider.joint.name}` "
                        f"(id={slider.slider_id}); keeping previous value {action[idx]:.4f}."
                    )
                    warned_slider_ids.add(slider.slider_id)
                continue
            action[idx] = value
        joint_targets = expand_l10_action_to_joint_targets(action)
        apply_joint_targets(body_id, joint_lookup, joint_targets)
        p.stepSimulation()
        frame_index += 1
        if wrist_camera is not None:
            camera_debug_ids = update_wrist_camera_debug(
                body_id,
                link_lookup,
                wrist_camera,
                debug_item_ids=camera_debug_ids,
            )
        if wrist_camera is not None and frame_index % wrist_camera.log_every == 0:
            capture = capture_wrist_camera(body_id, link_lookup, wrist_camera)
            print(f"[wrist-camera frame={frame_index}] {summarize_wrist_camera_capture(capture)}")
            if wrist_camera.save_dir is not None:
                saved_path = save_wrist_camera_capture(capture, wrist_camera.save_dir, frame_index)
                print(f"  saved wrist camera frame to: {saved_path}")
        time.sleep(dt)


def main() -> None:
    args = parse_args()
    urdf_path = args.urdf.resolve()
    if not urdf_path.exists():
        raise FileNotFoundError(f"URDF not found: {urdf_path}")
    wrist_camera = build_wrist_camera_config(args) if args.wrist_camera else None

    connection_mode = p.GUI if args.gui else p.DIRECT
    physics_client = p.connect(connection_mode)

    try:
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setPhysicsEngineParameter(enableFileCaching=0)
        p.setGravity(0, 0, 0)
        p.setTimeStep(args.dt)

        plane_id = p.loadURDF("plane.urdf")
        hand_id = p.loadURDF(
            str(urdf_path),
            basePosition=DEFAULT_HAND_BASE_POSITION,
            baseOrientation=p.getQuaternionFromEuler(DEFAULT_HAND_BASE_EULER),
            useFixedBase=args.fixed_base,
            flags=p.URDF_USE_INERTIA_FROM_FILE,
        )

        joints = collect_actuated_joints(hand_id)
        if not joints:
            raise RuntimeError("No actuated joints were found after loading the URDF.")
        joint_lookup = build_joint_lookup(joints)
        link_lookup = build_link_lookup(joints)
        validate_joint_configuration(joint_lookup)

        set_rest_pose(hand_id, joint_lookup)
        step_simulation(60)

        print(f"PyBullet mode: {'GUI' if args.gui else 'DIRECT'}")
        print(f"Loaded plane id: {plane_id}, hand URDF: {urdf_path}")
        print_joint_summary(hand_id, joints)
        verify_link_states(hand_id)
        if wrist_camera is not None:
            print(
                "Wrist camera:",
                f"link={wrist_camera.link_name}, "
                f"offset={np.round(wrist_camera.offset_local, 4).tolist()}, "
                f"forward={np.round(wrist_camera.forward_axis_local, 4).tolist()}, "
                f"fov={wrist_camera.fov:.1f}, "
                f"target_distance={wrist_camera.target_distance:.3f}",
            )
            if args.gui:
                update_wrist_camera_debug(hand_id, link_lookup, wrist_camera)
            capture = capture_wrist_camera(hand_id, link_lookup, wrist_camera)
            print(f"Initial wrist camera capture: {summarize_wrist_camera_capture(capture)}")
            if wrist_camera.save_dir is not None:
                saved_path = save_wrist_camera_capture(capture, wrist_camera.save_dir, frame_index=0)
                print(f"Saved initial wrist camera frame to: {saved_path}")
        print("URDF smoke test passed.")

        if args.gui:
            setup_gui_camera()
            sliders = create_joint_sliders(hand_id, joint_lookup)
            run_gui_joint_control(
                hand_id,
                joint_lookup,
                link_lookup,
                sliders,
                dt=args.dt,
                wrist_camera=wrist_camera,
            )
    finally:
        p.disconnect(physics_client)

if __name__ == "__main__":
    main()
