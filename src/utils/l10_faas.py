from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.utils import hand_utils, pose as pose_utils


L10_JOINT_DIM = 10
WRIST_POSE_DIM = 9
FAAS_HAND_STATE_DIM = hand_utils.MAPPED_JOINT_DIM
FAAS_PROPRIO_STATE_DIM = 18 + 2 * FAAS_HAND_STATE_DIM

# The training pipeline in this repo uses a canonical 82D state layout:
# [right_wrist_9d, left_wrist_9d, right_hand_32, left_hand_32]
#
# The slot indices below follow the FAAS-style 32D hand layout already used by
# the repo. We only fill the ten bend DoFs that can be read from L10 `getStateArc()`.
# Unused slots remain zero by default.

# TODO: 仔细检查这张映射表是否正确

L10_TO_FAAS_SLOT = {
    0: 1,   # thumb proximal bend / opposition main axis
    1: 3,   # thumb distal / coupled bend
    2: 6,   # index proximal bend
    3: 8,   # index distal / coupled bend
    4: 11,  # middle proximal bend
    5: 13,  # middle distal / coupled bend
    6: 16,  # ring proximal bend
    7: 18,  # ring distal / coupled bend
    8: 21,  # pinky proximal bend
    9: 23,  # pinky distal / coupled bend
}

L10_JOINT_LABELS = {
    0: "thumb_proximal_bend",
    1: "thumb_distal_coupled_bend",
    2: "index_proximal_bend",
    3: "index_distal_coupled_bend",
    4: "middle_proximal_bend",
    5: "middle_distal_coupled_bend",
    6: "ring_proximal_bend",
    7: "ring_distal_coupled_bend",
    8: "pinky_proximal_bend",
    9: "pinky_distal_coupled_bend",
}


def _as_float32_array(values, expected_last_dim: int, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=np.float32)
    if array.shape[-1] != expected_last_dim:
        raise ValueError(
            f"{name} must have last dimension {expected_last_dim}, got {array.shape}"
        )
    return array


def _identity_pose9d() -> np.ndarray:
    return pose_utils.mat_to_pose9d(np.eye(4, dtype=np.float32))


def _to_pose9d(pose: np.ndarray | list[float] | None, name: str) -> np.ndarray:
    if pose is None:
        return _identity_pose9d()

    pose_array = np.asarray(pose, dtype=np.float32)
    if pose_array.shape[-2:] == (4, 4):
        return pose_utils.mat_to_pose9d(pose_array)
    if pose_array.shape[-1] == WRIST_POSE_DIM:
        return pose_array

    raise ValueError(
        f"{name} must be a 4x4 pose matrix or 9D pose vector, got shape {pose_array.shape}"
    )


def _broadcast_pose9d(pose9d: np.ndarray, batch_shape: tuple[int, ...]) -> np.ndarray:
    if pose9d.shape[-1] != WRIST_POSE_DIM:
        raise ValueError(f"pose9d must end with dim {WRIST_POSE_DIM}, got {pose9d.shape}")

    target_shape = batch_shape + (WRIST_POSE_DIM,)
    if pose9d.shape == target_shape:
        return pose9d.astype(np.float32, copy=False)

    if pose9d.ndim == 1:
        return np.broadcast_to(pose9d, target_shape).astype(np.float32, copy=False)

    try:
        return np.broadcast_to(pose9d, target_shape).astype(np.float32, copy=False)
    except ValueError as exc:
        raise ValueError(
            f"Cannot broadcast pose with shape {pose9d.shape} to target shape {target_shape}"
        ) from exc


def l10_to_faas_hand_state(
    joint_angles_rad: np.ndarray | list[float],
    fill_value: float = 0.0,
) -> np.ndarray:
    """
    Map L10 10D joint angles to the repo's 32D FAAS hand-state layout.

    Args:
        joint_angles_rad: L10 joint angles from `getStateArc()`, shape (..., 10).
        fill_value: Default value for FAAS slots that L10 does not provide.

    Returns:
        np.ndarray of shape (..., 32), dtype float32.
    """
    joint_angles = _as_float32_array(joint_angles_rad, L10_JOINT_DIM, "joint_angles_rad")
    faas_state = np.full(
        joint_angles.shape[:-1] + (FAAS_HAND_STATE_DIM,),
        fill_value,
        dtype=np.float32,
    )

    for src_idx, dst_slot in L10_TO_FAAS_SLOT.items():
        faas_state[..., dst_slot] = joint_angles[..., src_idx]

    return faas_state


def l10_to_faas_proprio_state(
    joint_angles_rad: np.ndarray | list[float],
    hand_side: str = "right",
    active_wrist_pose: np.ndarray | list[float] | None = None,
    inactive_wrist_pose: np.ndarray | list[float] | None = None,
    unmapped_slot_fill: float = 0.0,
    inactive_hand_fill: float = 0.0,
) -> np.ndarray:
    """
    Convert L10 joint angles into the 82D proprio state expected by this repo.

    The canonical output layout is:
    [right_wrist_9d, left_wrist_9d, right_hand_32, left_hand_32]

    Args:
        joint_angles_rad: L10 joint angles from `getStateArc()`, shape (..., 10).
        hand_side: Which side the current L10 belongs to, either `right` or `left`.
        active_wrist_pose: Active hand wrist pose as 4x4 matrix or 9D pose.
            Defaults to identity.
        inactive_wrist_pose: Inactive hand wrist pose as 4x4 matrix or 9D pose.
            Defaults to identity.
        unmapped_slot_fill: Fill value for FAAS slots that L10 does not expose.
        inactive_hand_fill: Fill value for the missing hand-state slots.

    Returns:
        np.ndarray of shape (..., 82), dtype float32.
    """
    hand_side = hand_side.lower()
    if hand_side not in {"right", "left"}:
        raise ValueError(f"hand_side must be 'right' or 'left', got {hand_side}")

    active_hand = l10_to_faas_hand_state(
        joint_angles_rad,
        fill_value=unmapped_slot_fill,
    )
    batch_shape = active_hand.shape[:-1]

    inactive_hand = np.full(
        batch_shape + (FAAS_HAND_STATE_DIM,),
        inactive_hand_fill,
        dtype=np.float32,
    )

    active_wrist_9d = _broadcast_pose9d(
        _to_pose9d(active_wrist_pose, "active_wrist_pose"),
        batch_shape,
    )
    inactive_wrist_9d = _broadcast_pose9d(
        _to_pose9d(inactive_wrist_pose, "inactive_wrist_pose"),
        batch_shape,
    )

    if hand_side == "right":
        right_wrist_9d, left_wrist_9d = active_wrist_9d, inactive_wrist_9d
        right_hand, left_hand = active_hand, inactive_hand
    else:
        right_wrist_9d, left_wrist_9d = inactive_wrist_9d, active_wrist_9d
        right_hand, left_hand = inactive_hand, active_hand

    return np.concatenate(
        [right_wrist_9d, left_wrist_9d, right_hand, left_hand],
        axis=-1,
    )


@dataclass
class L10FAASStateAdapter:
    """
    Thin adapter around the L10 -> FAAS proprio mapping.

    Example:
        adapter = L10FAASStateAdapter(hand_side="right")
        hand_state = adapter.hand_state(joint_angles_rad)
        proprio = adapter.proprio_state(joint_angles_rad)
    """

    hand_side: str = "right"
    unmapped_slot_fill: float = 0.0
    inactive_hand_fill: float = 0.0
    active_wrist_pose: np.ndarray | list[float] | None = None
    inactive_wrist_pose: np.ndarray | list[float] | None = None

    def hand_state(self, joint_angles_rad: np.ndarray | list[float]) -> np.ndarray:
        return l10_to_faas_hand_state(
            joint_angles_rad=joint_angles_rad,
            fill_value=self.unmapped_slot_fill,
        )

    def proprio_state(self, joint_angles_rad: np.ndarray | list[float]) -> np.ndarray:
        return l10_to_faas_proprio_state(
            joint_angles_rad=joint_angles_rad,
            hand_side=self.hand_side,
            active_wrist_pose=self.active_wrist_pose,
            inactive_wrist_pose=self.inactive_wrist_pose,
            unmapped_slot_fill=self.unmapped_slot_fill,
            inactive_hand_fill=self.inactive_hand_fill,
        )


__all__ = [
    "FAAS_HAND_STATE_DIM",
    "FAAS_PROPRIO_STATE_DIM",
    "L10FAASStateAdapter",
    "L10_JOINT_DIM",
    "L10_JOINT_LABELS",
    "L10_TO_FAAS_SLOT",
    "l10_to_faas_hand_state",
    "l10_to_faas_proprio_state",
]
