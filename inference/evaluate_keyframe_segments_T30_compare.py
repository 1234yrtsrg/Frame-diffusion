import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
CSDI_DIR = REPO_ROOT / "CSDI"
if str(CSDI_DIR) not in sys.path:
    sys.path.insert(0, str(CSDI_DIR))

from baseline_express4d import cubic_interpolation, linear_interpolation
from dataset_keyframe_segments_t30 import get_dataloader
from main_model import CSDI_Express4D, CSDI_KeyframeSegmentsT30

DEFAULT_T30_CONFIG = "CSDI/config/keyframe_segments_T30.yaml"
DEFAULT_T30_CHECKPOINT = "save/keyframe_segments_T30/checkpoint_step_10000.pth"
DEFAULT_EXPRESS4D_CONFIG = "CSDI/config/express4d.yaml"

ARKIT_52_NAMES = [
    "browDownLeft",
    "browDownRight",
    "browInnerUp",
    "browOuterUpLeft",
    "browOuterUpRight",
    "cheekPuff",
    "cheekSquintLeft",
    "cheekSquintRight",
    "eyeBlinkLeft",
    "eyeBlinkRight",
    "eyeLookDownLeft",
    "eyeLookDownRight",
    "eyeLookInLeft",
    "eyeLookInRight",
    "eyeLookOutLeft",
    "eyeLookOutRight",
    "eyeLookUpLeft",
    "eyeLookUpRight",
    "eyeSquintLeft",
    "eyeSquintRight",
    "eyeWideLeft",
    "eyeWideRight",
    "jawForward",
    "jawLeft",
    "jawOpen",
    "jawRight",
    "mouthClose",
    "mouthDimpleLeft",
    "mouthDimpleRight",
    "mouthFrownLeft",
    "mouthFrownRight",
    "mouthFunnel",
    "mouthLeft",
    "mouthLowerDownLeft",
    "mouthLowerDownRight",
    "mouthPressLeft",
    "mouthPressRight",
    "mouthPucker",
    "mouthRight",
    "mouthRollLower",
    "mouthRollUpper",
    "mouthShrugLower",
    "mouthShrugUpper",
    "mouthSmileLeft",
    "mouthSmileRight",
    "mouthStretchLeft",
    "mouthStretchRight",
    "mouthUpperUpLeft",
    "mouthUpperUpRight",
    "noseSneerLeft",
    "noseSneerRight",
    "tongueOut",
]


def resolve_path(path):
    path = Path(path)
    if path.is_absolute():
        return path
    return REPO_ROOT / path


def load_config(config_path):
    config_path = resolve_path(config_path)
    if not config_path.is_file():
        raise FileNotFoundError(f"Config not found: {config_path}")
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_state_dict(checkpoint_path):
    checkpoint_path = resolve_path(checkpoint_path)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    state = torch.load(checkpoint_path, map_location="cpu")
    if isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]
    elif isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]

    if not isinstance(state, dict):
        raise ValueError(f"Unsupported checkpoint format: {checkpoint_path}")

    if all(key.startswith("module.") for key in state.keys()):
        state = {key.removeprefix("module."): value for key, value in state.items()}
    return state


def load_model(model_cls, config, checkpoint_path, device):
    model = model_cls(
        config,
        device,
        target_dim=config["dataset"].get("num_features", 52),
    ).to(device)
    model.load_state_dict(load_state_dict(checkpoint_path))
    model.eval()
    return model


def ensure_tensor(x, device):
    return torch.as_tensor(x, device=device).float()


def metric_dict(pred_middle, gt_middle, start, end):
    pred_full = torch.cat([start[:, None, :], pred_middle, end[:, None, :]], dim=1)
    gt_full = torch.cat([start[:, None, :], gt_middle, end[:, None, :]], dim=1)
    pred_vel = pred_full[:, 1:] - pred_full[:, :-1]
    gt_vel = gt_full[:, 1:] - gt_full[:, :-1]
    pred_acc = pred_full[:, 2:] - 2 * pred_full[:, 1:-1] + pred_full[:, :-2]
    gt_acc = gt_full[:, 2:] - 2 * gt_full[:, 1:-1] + gt_full[:, :-2]
    return {
        "middle_l1": torch.mean(torch.abs(pred_middle - gt_middle)).item(),
        "middle_mse": torch.mean((pred_middle - gt_middle) ** 2).item(),
        "velocity_l1": torch.mean(torch.abs(pred_vel - gt_vel)).item(),
        "acceleration_l1": torch.mean(torch.abs(pred_acc - gt_acc)).item(),
        "endpoint_continuity_start": torch.mean(torch.abs(pred_full[:, 1] - pred_full[:, 0])).item(),
        "endpoint_continuity_end": torch.mean(torch.abs(pred_full[:, -1] - pred_full[:, -2])).item(),
    }


def average_metrics(items):
    if not items:
        raise ValueError("No batches were evaluated. Check --max_batches and the selected split.")
    keys = items[0].keys()
    return {key: float(np.mean([item[key] for item in items])) for key in keys}


def resample_sequence(sequence, target_len):
    """Linearly resample [B,L,K] sequence to [B,target_len,K]."""
    if sequence.dim() != 3:
        raise ValueError(f"sequence must have shape [B,L,K], got {tuple(sequence.shape)}")
    source_len = sequence.shape[1]
    if source_len == target_len:
        return sequence

    positions = torch.linspace(0, source_len - 1, target_len, device=sequence.device)
    left = torch.floor(positions).long()
    right = torch.ceil(positions).long()
    weight = (positions - left.float()).view(1, target_len, 1)
    left_values = sequence[:, left, :]
    right_values = sequence[:, right, :]
    return (1.0 - weight) * left_values + weight * right_values


def predict_t30_middle(model, start, end, duration, num_samples):
    pred = model.generate_middle(start, end, duration, num_samples=num_samples)
    if pred.dim() == 3:
        return pred
    if pred.dim() == 4:
        return pred.mean(dim=1)
    raise ValueError(f"Unexpected T30 prediction shape: {tuple(pred.shape)}")


def predict_express4d_upsampled_middle(model, start, end, duration, num_samples, target_len=30):
    pred = model.generate_middle(start, end, duration, num_samples=num_samples)
    if pred.dim() == 3:
        pred_middle_10 = pred
    elif pred.dim() == 4:
        pred_middle_10 = pred.mean(dim=1)
    else:
        raise ValueError(f"Unexpected Express4D prediction shape: {tuple(pred.shape)}")

    full_12 = torch.cat([start[:, None, :], pred_middle_10, end[:, None, :]], dim=1)
    full_30 = resample_sequence(full_12, target_len)
    return full_30[:, 1:-1, :]


def run_eval_split(t30_model, express4d_model, loader, device, num_samples, max_batches=None):
    t30_model.eval()
    express4d_model.eval()
    t30_metrics = []
    express4d_upsampled_metrics = []
    linear_metrics = []
    cubic_metrics = []
    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            if max_batches is not None and batch_idx >= max_batches:
                break
            start = ensure_tensor(batch["start"], device)
            end = ensure_tensor(batch["end"], device)
            gt_middle = ensure_tensor(batch["middle"], device)
            duration = ensure_tensor(batch["duration"], device) if "duration" in batch else None

            t30_middle = predict_t30_middle(t30_model, start, end, duration, num_samples)
            t30_metrics.append(metric_dict(t30_middle, gt_middle, start, end))

            express4d_middle = predict_express4d_upsampled_middle(
                express4d_model,
                start,
                end,
                duration,
                num_samples,
                target_len=30,
            )
            express4d_upsampled_metrics.append(metric_dict(express4d_middle, gt_middle, start, end))

            linear = np.stack(
                [linear_interpolation(s, e, num_middle=28) for s, e in zip(start.cpu().numpy(), end.cpu().numpy())],
                axis=0,
            )
            linear = torch.from_numpy(linear).to(device)
            linear_metrics.append(metric_dict(linear, gt_middle, start, end))

            cubic = np.stack(
                [cubic_interpolation(s, e, num_middle=28) for s, e in zip(start.cpu().numpy(), end.cpu().numpy())],
                axis=0,
            )
            cubic = torch.from_numpy(cubic).to(device)
            cubic_metrics.append(metric_dict(cubic, gt_middle, start, end))

    return {
        "linear_baseline_30": average_metrics(linear_metrics),
        "cubic_baseline_30": average_metrics(cubic_metrics),
        "express4d_10_upsample_to_30": average_metrics(express4d_upsampled_metrics),
        "keyframe_segments_T30": average_metrics(t30_metrics),
    }


def main():
    parser = argparse.ArgumentParser(description="Compare 30-frame keyframe interpolation methods on the T30 split.")
    parser.add_argument("--t30_config", default=DEFAULT_T30_CONFIG)
    parser.add_argument("--t30_checkpoint", default=DEFAULT_T30_CHECKPOINT)
    parser.add_argument("--express4d_config", default=DEFAULT_EXPRESS4D_CONFIG)
    parser.add_argument("--express4d_checkpoint", required=True)
    parser.add_argument("--split", choices=["train", "valid", "test"], default="test")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--num_samples", type=int, default=1)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output_dir", default="outputs/keyframe_segments_T30_compare")
    parser.add_argument("--max_batches", type=int, default=None)
    args = parser.parse_args()

    device = args.device
    if device == "auto":
        device = "cuda:0" if torch.cuda.is_available() else "cpu"

    t30_config = load_config(args.t30_config)
    t30_config["seed"] = args.seed
    train_loader, valid_loader, test_loader = get_dataloader(
        t30_config,
        seed=args.seed,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    split_loader = {
        "train": train_loader,
        "valid": valid_loader,
        "test": test_loader,
    }[args.split]

    express4d_config = load_config(args.express4d_config)
    t30_model = load_model(CSDI_KeyframeSegmentsT30, t30_config, args.t30_checkpoint, device)
    express4d_model = load_model(
        CSDI_Express4D,
        express4d_config,
        args.express4d_checkpoint,
        device,
    )
    metrics = run_eval_split(
        t30_model,
        express4d_model,
        split_loader,
        device,
        num_samples=args.num_samples,
        max_batches=args.max_batches,
    )

    output_dir = resolve_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "t30_checkpoint": str(resolve_path(args.t30_checkpoint)),
        "t30_config": str(resolve_path(args.t30_config)),
        "express4d_checkpoint": str(resolve_path(args.express4d_checkpoint)),
        "express4d_config": str(resolve_path(args.express4d_config)),
        "split": args.split,
        "batch_size": args.batch_size,
        "num_samples": args.num_samples,
        "max_batches": args.max_batches,
        "seed": args.seed,
        "device": device,
        "arkit_52_names": ARKIT_52_NAMES,
        "metrics": metrics,
    }
    with open(output_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print(json.dumps(metrics, indent=2))
    print(f"saved metrics to {output_dir / 'metrics.json'}")


if __name__ == "__main__":
    main()
