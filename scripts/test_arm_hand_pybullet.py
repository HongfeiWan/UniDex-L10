#!/usr/bin/env python3
"""
PyBullet interactive test for the combined arm+hand URDF.

- Loads `assets/arm_hand.urdf`
- Provides 6 sliders for the arm (joint1..joint6)
- Provides 10 primary-DOF sliders for the hand, keeping the same coupling rules
  as `scripts/test_l10_right_hand_pybullet.py`

Example:
    python scripts/test_arm_hand_pybullet.py
    python scripts/test_arm_hand_pybullet.py --no-gui
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
DEFAULT_URDF = REPO_ROOT / "assets" / "arm_hand.urdf"
DEFAULT_BASE_POSITION = (0.0, 0.0, 0.0)
DEFAULT_BASE_EULER = (0.0, 0.0, 0.0)
DEFAULT_DT = 1.0 / 240.0
DEFAULT_MOTOR_FORCE = 30.0

HAND_PREFIX = "hand_"

ARM_JOINT_NAMES = ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6"]
HAND_PRIMARY_DOF_NAMES = [
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
    parser = argparse.ArgumentParser(description="Test the combined arm+hand URDF with PyBullet.")
    parser.add_argument("--urdf", type=Path, default=DEFAULT_URDF, help="Path to the combined URDF.")
    parser.add_argument("--gui", action=argparse.BooleanOptionalAction, default=True, help="Launch the PyBullet GUI.")
    parser.add_argument("--dt", type=float, default=DEFAULT_DT, help="Simulation step size in seconds.")
    parser.add_argument(
        "--fixed-base",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Load the combined model as fixed base (default: enabled).",
    )
    parser.add_argument(
        "--motor-force",
        type=float,
        default=DEFAULT_MOTOR_FORCE,
        help="Max force used by the position controllers.",
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


def clamp_joint_position(joint: JointRecord, value: float) -> float:
    lower = joint.lower
    upper = joint.upper
    if math.isfinite(lower) and math.isfinite(upper) and upper > lower:
        return min(max(value, lower), upper)
    return value


def hand_expand_primary_action_to_targets(primary_action: list[float]) -> dict[str, float]:
    if len(primary_action) != len(HAND_PRIMARY_DOF_NAMES):
        raise ValueError(f"Expected {len(HAND_PRIMARY_DOF_NAMES)} values, got {len(primary_action)}")

    q: dict[str, float] = {}
    q[HAND_PREFIX + "thumb_cmc_roll"] = primary_action[0]
    q[HAND_PREFIX + "thumb_cmc_yaw"] = primary_action[1]
    q[HAND_PREFIX + "thumb_cmc_pitch"] = primary_action[2]
    q[HAND_PREFIX + "index_mcp_roll"] = primary_action[3]
    q[HAND_PREFIX + "index_mcp_pitch"] = primary_action[4]
    q[HAND_PREFIX + "middle_mcp_pitch"] = primary_action[5]
    q[HAND_PREFIX + "ring_mcp_roll"] = primary_action[6]
    q[HAND_PREFIX + "ring_mcp_pitch"] = primary_action[7]
    q[HAND_PREFIX + "pinky_mcp_roll"] = primary_action[8]
    q[HAND_PREFIX + "pinky_mcp_pitch"] = primary_action[9]

    # Couplings (mirrors `expand_l10_action_to_joint_targets` but with the `hand_` prefix).
    q[HAND_PREFIX + "thumb_mcp"] = 1.3898 * q[HAND_PREFIX + "thumb_cmc_pitch"]
    q[HAND_PREFIX + "thumb_ip"] = 1.5080 * q[HAND_PREFIX + "thumb_cmc_pitch"]

    q[HAND_PREFIX + "index_pip"] = 1.3462 * q[HAND_PREFIX + "index_mcp_pitch"]
    q[HAND_PREFIX + "index_dip"] = 0.4616 * q[HAND_PREFIX + "index_mcp_pitch"]

    q[HAND_PREFIX + "middle_pip"] = 1.3462 * q[HAND_PREFIX + "middle_mcp_pitch"]
    q[HAND_PREFIX + "middle_dip"] = 0.4616 * q[HAND_PREFIX + "middle_mcp_pitch"]

    q[HAND_PREFIX + "ring_pip"] = 1.3462 * q[HAND_PREFIX + "ring_mcp_pitch"]
    q[HAND_PREFIX + "ring_dip"] = 0.4616 * q[HAND_PREFIX + "ring_mcp_pitch"]

    q[HAND_PREFIX + "pinky_pip"] = 1.3462 * q[HAND_PREFIX + "pinky_mcp_pitch"]
    q[HAND_PREFIX + "pinky_dip"] = 0.4616 * q[HAND_PREFIX + "pinky_mcp_pitch"]

    return q


def apply_joint_targets(
    body_id: int,
    joint_lookup: dict[str, JointRecord],
    joint_targets: dict[str, float],
    motor_force: float,
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
            force=motor_force,
        )


def create_named_sliders(
    body_id: int,
    joint_lookup: dict[str, JointRecord],
    joint_names: list[str],
    label_prefix: str,
) -> list[JointSlider]:
    sliders: list[JointSlider] = []
    for name in joint_names:
        joint = joint_lookup[name]
        lower = joint.lower if math.isfinite(joint.lower) else -math.pi
        upper = joint.upper if math.isfinite(joint.upper) else math.pi
        if upper <= lower:
            lower, upper = -math.pi, math.pi
        current = float(p.getJointState(body_id, joint.index)[0])
        default_value = min(max(current, lower), upper)
        slider_id = p.addUserDebugParameter(f"{label_prefix}{joint.name}", lower, upper, default_value)
        sliders.append(JointSlider(joint=joint, slider_id=slider_id, default_value=default_value))
    return sliders


def read_slider_value(slider: JointSlider, fallback_value: float) -> tuple[bool, float]:
    for _ in range(5):
        try:
            return True, float(p.readUserDebugParameter(slider.slider_id))
        except p.error:
            time.sleep(0.01)
    return False, fallback_value


def setup_gui_camera() -> None:
    p.resetDebugVisualizerCamera(
        cameraDistance=0.7,
        cameraYaw=135,
        cameraPitch=-25,
        cameraTargetPosition=[0.0, 0.0, 0.15],
    )


def validate_expected_joints(joint_lookup: dict[str, JointRecord]) -> None:
    missing: list[str] = []
    for name in ARM_JOINT_NAMES:
        if name not in joint_lookup:
            missing.append(name)
    for name in HAND_PRIMARY_DOF_NAMES:
        prefixed = HAND_PREFIX + name
        if prefixed not in joint_lookup:
            missing.append(prefixed)

    # also validate coupled joints exist
    coupled = hand_expand_primary_action_to_targets([0.0] * len(HAND_PRIMARY_DOF_NAMES)).keys()
    for name in coupled:
        if name not in joint_lookup:
            missing.append(name)

    if missing:
        known = ", ".join(sorted(joint_lookup.keys())[:60])
        raise RuntimeError(
            "Missing expected joints in the combined URDF:\n"
            + "\n".join(f"  - {name}" for name in sorted(set(missing)))
            + f"\nKnown joint names (first 60): {known}"
        )


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
        p.loadURDF("plane.urdf")

        body_id = p.loadURDF(
            str(urdf_path),
            basePosition=list(DEFAULT_BASE_POSITION),
            baseOrientation=p.getQuaternionFromEuler(list(DEFAULT_BASE_EULER)),
            useFixedBase=args.fixed_base,
            flags=p.URDF_USE_INERTIA_FROM_FILE,
        )

        joints = collect_actuated_joints(body_id)
        if not joints:
            raise RuntimeError("No actuated joints were found after loading the combined URDF.")
        joint_lookup = build_joint_lookup(joints)
        validate_expected_joints(joint_lookup)

        # initialize hand to open pose (all 10 primary = 0)
        hand_primary_action = [0.0] * len(HAND_PRIMARY_DOF_NAMES)
        apply_joint_targets(
            body_id,
            joint_lookup,
            hand_expand_primary_action_to_targets(hand_primary_action),
            motor_force=args.motor_force,
            reset_state=True,
        )

        print(f"PyBullet mode: {'GUI' if args.gui else 'DIRECT'}")
        print(f"Loaded combined URDF: {urdf_path}")
        print("GUI controls:")
        print("  - arm sliders: joint1..joint6")
        print("  - hand sliders: 10 primary DOFs (coupled joints updated automatically)")

        if args.gui:
            setup_gui_camera()
            arm_sliders = create_named_sliders(body_id, joint_lookup, ARM_JOINT_NAMES, label_prefix="arm::")
            hand_sliders = create_named_sliders(
                body_id,
                joint_lookup,
                [HAND_PREFIX + name for name in HAND_PRIMARY_DOF_NAMES],
                label_prefix="hand::",
            )

            warned_slider_ids: set[int] = set()
            hand_action = [slider.default_value for slider in hand_sliders]

            while p.isConnected():
                # arm
                arm_joint_targets: dict[str, float] = {}
                for slider in arm_sliders:
                    ok, value = read_slider_value(slider, fallback_value=slider.default_value)
                    if not ok:
                        continue
                    arm_joint_targets[slider.joint.name] = value
                if arm_joint_targets:
                    apply_joint_targets(
                        body_id,
                        joint_lookup,
                        arm_joint_targets,
                        motor_force=args.motor_force,
                        reset_state=False,
                    )

                # hand (10 primary => expanded with couplings)
                for idx, slider in enumerate(hand_sliders):
                    ok, value = read_slider_value(slider, fallback_value=hand_action[idx])
                    if not ok:
                        if slider.slider_id not in warned_slider_ids:
                            print(
                                f"Warning: failed to read slider `{slider.joint.name}` "
                                f"(id={slider.slider_id}); keeping previous value {hand_action[idx]:.4f}."
                            )
                            warned_slider_ids.add(slider.slider_id)
                        continue
                    hand_action[idx] = value

                hand_joint_targets = hand_expand_primary_action_to_targets(hand_action)
                apply_joint_targets(
                    body_id,
                    joint_lookup,
                    hand_joint_targets,
                    motor_force=args.motor_force,
                    reset_state=False,
                )

                p.stepSimulation()
                time.sleep(args.dt)
        else:
            # headless: just verify we can step without error
            for _ in range(240):
                p.stepSimulation()
                time.sleep(args.dt)
            print("Headless smoke test finished.")
    finally:
        p.disconnect(physics_client)


if __name__ == "__main__":
    main()

