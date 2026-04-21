#!/usr/bin/env python3
"""
Minimal UniDex checkpoint inference runner.

This script is intended to smoke-test the end-to-end inference pipeline:
    pointcloud + prompt + L10/82D state -> UniDex checkpoint -> predicted 82D action chunk

Examples:
    python scripts/run_unidex_ckpt_inference.py \
        --checkpoint "/path/to/32-epochs.ckpt" \
        --prompt "Use L10 hands to grasp the object" \
        --dummy-pointcloud

    python scripts/run_unidex_ckpt_inference.py \
        --checkpoint "/path/to/32-epochs.ckpt" \
        --prompt "Use L10 hands to pick up the cup" \
        --pointcloud-npy data/debug/example_pointcloud.npy \
        --l10-joints 0.1 0.0 0.2 0.0 0.4 0.3 0.0 0.2 0.0 0.1
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import hydra
import numpy as np
import torch
from omegaconf import OmegaConf


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.utils.l10_faas import (  # noqa: E402
    FAAS_HAND_STATE_DIM,
    FAAS_PROPRIO_STATE_DIM,
    l10_to_faas_proprio_state,
)
from src.utils.normalizers import Normalizer  # noqa: E402


DEFAULT_NORMALIZER_YAML = REPO_ROOT / "config" / "dataset" / "normalizer" / "base.yaml"
DEFAULT_NUM_POINTS = 10000


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run UniDex checkpoint inference.")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="Path to the UniDex checkpoint (.ckpt).",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        required=True,
        help="Single text prompt for the model.",
    )
    parser.add_argument(
        "--pointcloud-npy",
        type=Path,
        default=None,
        help="Optional .npy/.npz pointcloud path. Expected xyzrgb with shape (N, 6), (1, N, 6), or (1, 1, N, 6).",
    )
    parser.add_argument(
        "--dummy-pointcloud",
        action="store_true",
        help="Use a synthetic xyzrgb pointcloud if no real pointcloud is available.",
    )
    parser.add_argument(
        "--num-points",
        type=int,
        default=DEFAULT_NUM_POINTS,
        help="Point count to use after loading/sampling.",
    )
    parser.add_argument(
        "--state-npy",
        type=Path,
        default=None,
        help="Optional prebuilt 82D state file (.npy/.npz). Overrides --l10-joints if provided.",
    )
    parser.add_argument(
        "--l10-joints",
        type=float,
        nargs=10,
        default=None,
        help="Ten L10 joint values in radians. Used to build the 82D right-hand state.",
    )
    parser.add_argument(
        "--hand-side",
        type=str,
        default="right",
        choices=["right", "left"],
        help="Which hand side the L10 state belongs to.",
    )
    parser.add_argument(
        "--normalizer-yaml",
        type=Path,
        default=DEFAULT_NORMALIZER_YAML,
        help="Normalizer YAML used for state/action scaling.",
    )
    parser.add_argument(
        "--skip-normalizer",
        action="store_true",
        help="Skip state normalization and action unnormalization.",
    )
    parser.add_argument(
        "--pretrained-model-path",
        type=str,
        default=None,
        help="Optional override for model.pretrained_model_path/tokenizer path, useful when using a local PaliGemma copy.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Torch device, e.g. cuda, cuda:0, cpu.",
    )
    parser.add_argument(
        "--output-npy",
        type=Path,
        default=None,
        help="Optional path to save the predicted action chunk as .npy.",
    )
    return parser.parse_args()


def load_normalizer(path: Path) -> Normalizer:
    cfg = OmegaConf.load(path)
    norm_stats = OmegaConf.to_container(cfg.norm_stats, resolve=True)
    norm_type = OmegaConf.to_container(cfg.norm_type, resolve=True)
    return Normalizer(norm_stats=norm_stats, norm_type=norm_type)


def load_array(path: Path) -> np.ndarray:
    if path.suffix == ".npz":
        data = np.load(path)
        if len(data.files) != 1:
            raise ValueError(f"{path} contains multiple arrays; please keep only one.")
        return np.asarray(data[data.files[0]], dtype=np.float32)
    return np.asarray(np.load(path), dtype=np.float32)


def sample_or_pad_pointcloud(pointcloud: np.ndarray, num_points: int) -> np.ndarray:
    if pointcloud.ndim != 2 or pointcloud.shape[1] != 6:
        raise ValueError(f"Expected pointcloud shape (N, 6), got {pointcloud.shape}")

    num_current = pointcloud.shape[0]
    if num_current == num_points:
        return pointcloud.astype(np.float32, copy=False)

    if num_current > num_points:
        idx = np.linspace(0, num_current - 1, num_points, dtype=np.int64)
        return pointcloud[idx].astype(np.float32, copy=False)

    pad_count = num_points - num_current
    pad = np.repeat(pointcloud[-1:, :], pad_count, axis=0)
    return np.concatenate([pointcloud, pad], axis=0).astype(np.float32, copy=False)


def build_dummy_pointcloud(num_points: int) -> np.ndarray:
    xyz = np.random.uniform(low=-0.08, high=0.08, size=(num_points, 3)).astype(np.float32)
    rgb = np.full((num_points, 3), 0.5, dtype=np.float32)
    return np.concatenate([xyz, rgb], axis=1)


def load_pointcloud(args: argparse.Namespace) -> np.ndarray:
    if args.pointcloud_npy is not None:
        pointcloud = load_array(args.pointcloud_npy)
        if pointcloud.ndim == 4:
            if pointcloud.shape[0] != 1 or pointcloud.shape[1] != 1:
                raise ValueError(f"Expected (1, 1, N, 6) pointcloud, got {pointcloud.shape}")
            pointcloud = pointcloud[0, 0]
        elif pointcloud.ndim == 3:
            if pointcloud.shape[0] != 1:
                raise ValueError(f"Expected (1, N, 6) pointcloud, got {pointcloud.shape}")
            pointcloud = pointcloud[0]
        return sample_or_pad_pointcloud(pointcloud, args.num_points)

    if args.dummy_pointcloud:
        return build_dummy_pointcloud(args.num_points)

    raise ValueError("Please provide --pointcloud-npy or use --dummy-pointcloud.")


def load_state(args: argparse.Namespace) -> np.ndarray:
    if args.state_npy is not None:
        state = load_array(args.state_npy)
        if state.ndim == 3:
            if state.shape[0] != 1 or state.shape[1] != 1:
                raise ValueError(f"Expected state shape (1, 1, 82), got {state.shape}")
            state = state[0, 0]
        elif state.ndim == 2:
            if state.shape[0] != 1:
                raise ValueError(f"Expected state shape (1, 82), got {state.shape}")
            state = state[0]
        if state.shape != (FAAS_PROPRIO_STATE_DIM,):
            raise ValueError(f"Expected flat 82D state, got {state.shape}")
        return state.astype(np.float32, copy=False)

    if args.l10_joints is None:
        raise ValueError("Please provide either --state-npy or --l10-joints.")

    l10 = np.asarray(args.l10_joints, dtype=np.float32)
    return l10_to_faas_proprio_state(l10, hand_side=args.hand_side).astype(np.float32, copy=False)


def build_model_config(pretrained_model_path: str | None):
    overrides = ["model=unidex_inference"]
    if pretrained_model_path:
        overrides.extend(
            [
                f"model.pretrained_model_path={pretrained_model_path}",
                f"model.tokenizer_path={pretrained_model_path}",
            ]
        )

    with hydra.initialize_config_dir(
        config_dir=str(REPO_ROOT / "config"),
        version_base=None,
    ):
        cfg = hydra.compose(config_name="finetune", overrides=overrides)
    return cfg.model


def strip_policy_prefix(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    cleaned = {}
    for key, value in state_dict.items():
        if key.startswith("policy."):
            cleaned[key[len("policy."):]] = value
        else:
            cleaned[key] = value
    return cleaned


def load_model(
    checkpoint_path: Path,
    device: torch.device,
    pretrained_model_path: str | None,
) -> torch.nn.Module:
    model_cfg = build_model_config(pretrained_model_path)
    model = hydra.utils.instantiate(model_cfg)

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state_dict = checkpoint["state_dict"] if "state_dict" in checkpoint else checkpoint
    missing, unexpected = model.load_state_dict(strip_policy_prefix(state_dict), strict=False)

    if missing:
        print(f"Warning: missing keys when loading checkpoint: {len(missing)}")
    if unexpected:
        print(f"Warning: unexpected keys when loading checkpoint: {len(unexpected)}")

    model.eval()
    model.to(device)
    return model


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)

    checkpoint_path = args.checkpoint.resolve()
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    pointcloud_np = load_pointcloud(args)
    state_np = load_state(args)

    normalizer = None if args.skip_normalizer else load_normalizer(args.normalizer_yaml.resolve())
    if normalizer is not None:
        normalized = normalizer.normalize(
            {
                "state": state_np[None, :],
                "pointcloud": pointcloud_np[None, :, :],
            }
        )
        state_np = np.asarray(normalized["state"][0], dtype=np.float32)

    pointcloud_tensor = torch.from_numpy(pointcloud_np[None, None, :, :]).to(device=device, dtype=torch.float32)
    state_tensor = torch.from_numpy(state_np[None, None, :]).to(device=device, dtype=torch.float32)
    prompt_batch = [args.prompt]

    model = load_model(
        checkpoint_path=checkpoint_path,
        device=device,
        pretrained_model_path=args.pretrained_model_path,
    )

    with torch.inference_mode():
        pred_action = model.infer_action(
            pointcloud=pointcloud_tensor,
            state=state_tensor,
            prompt=prompt_batch,
        )

    if normalizer is not None:
        pred_action = normalizer.unnormalize({"action": pred_action})["action"]

    pred_action_np = pred_action.detach().cpu().numpy().astype(np.float32, copy=False)

    print("=== UniDex inference completed ===")
    print(f"checkpoint: {checkpoint_path}")
    print(f"pointcloud shape: {tuple(pointcloud_tensor.shape)}")
    print(f"state shape: {tuple(state_tensor.shape)}")
    print(f"predicted action shape: {pred_action_np.shape}")
    print(f"prompt: {args.prompt}")

    first_step = pred_action_np[0, 0]
    right_hand = first_step[18 : 18 + FAAS_HAND_STATE_DIM]
    left_hand = first_step[18 + FAAS_HAND_STATE_DIM :]

    print("first action step, right wrist 9d:", np.round(first_step[:9], 4).tolist())
    print("first action step, left wrist 9d:", np.round(first_step[9:18], 4).tolist())
    print("first action step, right hand nonzero idx:", np.where(np.abs(right_hand) > 1e-6)[0].tolist())
    print("first action step, left hand nonzero idx:", np.where(np.abs(left_hand) > 1e-6)[0].tolist())

    if args.output_npy is not None:
        output_path = args.output_npy.resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(output_path, pred_action_np)
        print(f"saved predicted action chunk to: {output_path}")


if __name__ == "__main__":
    main()
