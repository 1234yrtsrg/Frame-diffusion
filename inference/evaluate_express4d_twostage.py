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
from dataset_express4d import get_dataloader
from main_model import CSDI_Express4D


DEFAULT_CONFIG = "CSDI/config/express4d.yaml"
DEFAULT_OUTPUT_DIR = "outputs/express4d_twostage_eval"


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


def load_model(config, checkpoint_path, device):
    model = CSDI_Express4D(
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


def predict_middle(model, start, end, duration, num_samples):
    pred = model.generate_middle(start, end, duration, num_samples=num_samples)
    if pred.dim() == 3:
        return pred
    if pred.dim() == 4:
        return pred.mean(dim=1)
    raise ValueError(f"Unexpected prediction shape: {tuple(pred.shape)}")


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
    return (1.0 - weight) * sequence[:, left, :] + weight * sequence[:, right, :]


def predict_twostage_downsampled(model, start, end, duration, num_samples, coarse_len=12, target_len=12):
    coarse_middle = predict_middle(model, start, end, duration, num_samples)
    coarse_full = torch.cat([start[:, None, :], coarse_middle, end[:, None, :]], dim=1)
    if coarse_full.shape[1] != coarse_len:
        raise ValueError(f"Expected coarse length {coarse_len}, got {coarse_full.shape[1]}")

    interval_duration = duration.float() / float(coarse_len - 1)
    refined_parts = []
    for pair_index in range(coarse_len - 1):
        pair_middle = predict_middle(
            model,
            coarse_full[:, pair_index],
            coarse_full[:, pair_index + 1],
            interval_duration,
            num_samples,
        )
        segment = torch.cat(
            [
                coarse_full[:, pair_index : pair_index + 1],
                pair_middle,
                coarse_full[:, pair_index + 1 : pair_index + 2],
            ],
            dim=1,
        )
        refined_parts.append(segment if pair_index == 0 else segment[:, 1:])

    refined_full = torch.cat(refined_parts, dim=1)
    downsampled_full = resample_sequence(refined_full, target_len)
    return downsampled_full[:, 1:-1], refined_full.shape[1]


def run_eval(model, loader, device, num_samples, max_batches=None):
    model.eval()
    one_stage_metrics = []
    twostage_metrics = []
    linear_metrics = []
    cubic_metrics = []
    refined_lengths = []

    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            if max_batches is not None and batch_idx >= max_batches:
                break

            start = ensure_tensor(batch["start"], device)
            end = ensure_tensor(batch["end"], device)
            gt_middle = ensure_tensor(batch["middle"], device)
            duration = ensure_tensor(batch["duration"], device)

            one_stage_middle = predict_middle(model, start, end, duration, num_samples)
            one_stage_metrics.append(metric_dict(one_stage_middle, gt_middle, start, end))

            twostage_middle, refined_len = predict_twostage_downsampled(
                model,
                start,
                end,
                duration,
                num_samples,
                coarse_len=12,
                target_len=12,
            )
            twostage_metrics.append(metric_dict(twostage_middle, gt_middle, start, end))
            refined_lengths.append(refined_len)

            linear = np.stack(
                [linear_interpolation(s, e, num_middle=10) for s, e in zip(start.cpu().numpy(), end.cpu().numpy())],
                axis=0,
            )
            linear = torch.from_numpy(linear).to(device)
            linear_metrics.append(metric_dict(linear, gt_middle, start, end))

            cubic = np.stack(
                [cubic_interpolation(s, e, num_middle=10) for s, e in zip(start.cpu().numpy(), end.cpu().numpy())],
                axis=0,
            )
            cubic = torch.from_numpy(cubic).to(device)
            cubic_metrics.append(metric_dict(cubic, gt_middle, start, end))

    return {
        "linear_baseline_12": average_metrics(linear_metrics),
        "cubic_baseline_12": average_metrics(cubic_metrics),
        "express4d_one_stage_12": average_metrics(one_stage_metrics),
        "express4d_twostage_downsample_to_12": average_metrics(twostage_metrics),
        "twostage_refined_length_before_downsample": int(refined_lengths[0]) if refined_lengths else None,
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate Express4D two-stage inference on the Express4D test split.")
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", choices=["train", "test"], default="test")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--num_samples", type=int, default=1)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output_dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--max_batches", type=int, default=None)
    args = parser.parse_args()

    device = args.device
    if device == "auto":
        device = "cuda:0" if torch.cuda.is_available() else "cpu"

    config = load_config(args.config)
    train_loader, test_loader = get_dataloader(
        config,
        seed=args.seed,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    loader = train_loader if args.split == "train" else test_loader

    model = load_model(config, args.checkpoint, device)
    metrics = run_eval(
        model,
        loader,
        device,
        num_samples=args.num_samples,
        max_batches=args.max_batches,
    )

    output_dir = resolve_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "checkpoint": str(resolve_path(args.checkpoint)),
        "config": str(resolve_path(args.config)),
        "split": args.split,
        "batch_size": args.batch_size,
        "num_samples": args.num_samples,
        "max_batches": args.max_batches,
        "seed": args.seed,
        "device": device,
        "metrics": metrics,
    }
    with open(output_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print(json.dumps(metrics, indent=2))
    print(f"saved metrics to {output_dir / 'metrics.json'}")


if __name__ == "__main__":
    main()
