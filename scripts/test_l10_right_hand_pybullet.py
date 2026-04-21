#!/usr/bin/env python3
"""
PyBullet smoke test for the L10 right-hand URDF.

Examples:
    python scripts/test_l10_right_hand_pybullet.py
    python scripts/test_l10_right_hand_pybullet.py --no-gui
"""

from __future__ import annotations

import argparse
import math
import time
from dataclasses import dataclass
from pathlib import Path

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
            force=5.0,
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


def validate_joint_configuration(joint_lookup: dict[str, JointRecord]) -> None:
    expected_joint_names = expand_l10_action_to_joint_targets([0.0] * len(PRIMARY_DOF_NAMES)).keys()
    missing = [joint_name for joint_name in expected_joint_names if joint_name not in joint_lookup]
    if missing:
        missing_text = ", ".join(missing)
        raise RuntimeError(f"Missing expected joints in URDF: {missing_text}")


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

    # Give the GUI a moment to register the controls before the first read.
    for _ in range(10):
        p.stepSimulation()
        time.sleep(0.01)
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
    sliders: list[JointSlider],
    dt: float,
) -> None:
    print("GUI is running. Drag the 10 primary sliders to control the hand.")
    print("The remaining joints are updated from the coupling formulas automatically.")
    print("Close the PyBullet window or press Ctrl+C in the terminal to stop.")
    action = [slider.default_value for slider in sliders]
    warned_slider_ids: set[int] = set()
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
        time.sleep(dt)


def main() -> None:
    args = parse_args()
    urdf_path = args.urdf.resolve()
    if not urdf_path.exists():
        raise FileNotFoundError(f"URDF not found: {urdf_path}")

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
            basePosition=[0.0, 0.0, 0.0],
            baseOrientation=p.getQuaternionFromEuler([0.0, 0.0, 0.0]),
            useFixedBase=args.fixed_base,
            flags=p.URDF_USE_INERTIA_FROM_FILE,
        )

        joints = collect_actuated_joints(hand_id)
        if not joints:
            raise RuntimeError("No actuated joints were found after loading the URDF.")
        joint_lookup = build_joint_lookup(joints)
        validate_joint_configuration(joint_lookup)

        set_rest_pose(hand_id, joint_lookup)
        for _ in range(60):
            p.stepSimulation()

        print(f"PyBullet connected in {'GUI' if args.gui else 'DIRECT'} mode.")
        print(f"Loaded plane id: {plane_id}")
        print(f"Loaded URDF: {urdf_path}")
        print_joint_summary(hand_id, joints)
        verify_link_states(hand_id)
        print("URDF smoke test passed.")

        if args.gui:
            setup_gui_camera()
            sliders = create_joint_sliders(hand_id, joint_lookup)
            run_gui_joint_control(hand_id, joint_lookup, sliders, dt=args.dt)
    finally:
        p.disconnect(physics_client)

if __name__ == "__main__":
    main()
