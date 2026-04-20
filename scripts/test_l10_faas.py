#!/usr/bin/env python3
"""
Minimal validation script for `src/utils/l10_faas.py`.

This script fabricates one L10 `getStateArc()`-style 10D joint-angle vector,
maps it into the FAAS hand-state / proprio-state layouts, and prints the key
results so the mapping can be checked quickly by eye.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.utils.l10_faas import (  # noqa: E402
    FAAS_HAND_STATE_DIM,
    FAAS_PROPRIO_STATE_DIM,
    L10_TO_FAAS_SLOT,
    L10FAASStateAdapter,
    l10_to_faas_hand_state,
    l10_to_faas_proprio_state,
)


def build_example_l10_joint_angles() -> np.ndarray:
    """
    Fabricate a plausible L10 `getStateArc()` input in radians.

    The values represent a lightly flexed hand:
    thumb bent moderately, index/middle more closed, ring/pinky slightly less.
    """
    return np.array(
        [
            0.45,  # q0 thumb proximal bend
            0.20,  # q1 thumb distal / coupled bend
            0.95,  # q2 index proximal bend
            0.55,  # q3 index distal / coupled bend
            0.90,  # q4 middle proximal bend
            0.50,  # q5 middle distal / coupled bend
            0.80,  # q6 ring proximal bend
            0.42,  # q7 ring distal / coupled bend
            0.72,  # q8 pinky proximal bend
            0.35,  # q9 pinky distal / coupled bend
        ],
        dtype=np.float32,
    )


def build_example_wrist_pose() -> np.ndarray:
    """Create a simple 4x4 wrist pose for the active hand."""
    pose = np.eye(4, dtype=np.float32)
    pose[:3, 3] = np.array([0.35, -0.12, 0.58], dtype=np.float32)
    return pose


def print_nonzero_faas_slots(hand_state: np.ndarray) -> None:
    print("Mapped FAAS slots:")
    for src_idx, dst_slot in L10_TO_FAAS_SLOT.items():
        print(f"  q{src_idx:02d} -> slot {dst_slot:02d}: {hand_state[dst_slot]:.4f}")


def validate_single_hand_mapping() -> None:
    joint_angles = build_example_l10_joint_angles()
    wrist_pose = build_example_wrist_pose()

    hand_state = l10_to_faas_hand_state(joint_angles)
    proprio_state = l10_to_faas_proprio_state(
        joint_angles,
        hand_side="right",
        active_wrist_pose=wrist_pose,
    )

    assert hand_state.shape == (FAAS_HAND_STATE_DIM,)
    assert proprio_state.shape == (FAAS_PROPRIO_STATE_DIM,)

    for src_idx, dst_slot in L10_TO_FAAS_SLOT.items():
        assert np.isclose(
            hand_state[dst_slot], joint_angles[src_idx]
        ), f"Slot {dst_slot} should equal q{src_idx}"

    right_hand = proprio_state[18 : 18 + FAAS_HAND_STATE_DIM]
    left_hand = proprio_state[18 + FAAS_HAND_STATE_DIM :]
    assert np.allclose(right_hand, hand_state)
    assert np.allclose(left_hand, 0.0)

    print("Single right-hand mapping passed.")
    print(f"hand_state shape: {hand_state.shape}")
    print(f"proprio_state shape: {proprio_state.shape}")
    print_nonzero_faas_slots(hand_state)
    print("Right wrist 9D slice:", np.round(proprio_state[:9], 4).tolist())
    print("Left wrist 9D slice:", np.round(proprio_state[9:18], 4).tolist())


def validate_batch_mapping() -> None:
    joint_angles = np.stack(
        [
            build_example_l10_joint_angles(),
            build_example_l10_joint_angles() * np.float32(0.5),
        ],
        axis=0,
    )
    adapter = L10FAASStateAdapter(hand_side="left")

    hand_state = adapter.hand_state(joint_angles)
    proprio_state = adapter.proprio_state(joint_angles)

    assert hand_state.shape == (2, FAAS_HAND_STATE_DIM)
    assert proprio_state.shape == (2, FAAS_PROPRIO_STATE_DIM)

    right_hand = proprio_state[:, 18 : 18 + FAAS_HAND_STATE_DIM]
    left_hand = proprio_state[:, 18 + FAAS_HAND_STATE_DIM :]
    assert np.allclose(right_hand, 0.0)
    assert np.allclose(left_hand, hand_state)

    print("Batch left-hand mapping passed.")
    print(f"batch hand_state shape: {hand_state.shape}")
    print(f"batch proprio_state shape: {proprio_state.shape}")


def main() -> None:
    print("=== L10 -> FAAS adapter smoke test ===")
    validate_single_hand_mapping()
    print()
    validate_batch_mapping()
    print()
    print("All checks passed.")


if __name__ == "__main__":
    main()
