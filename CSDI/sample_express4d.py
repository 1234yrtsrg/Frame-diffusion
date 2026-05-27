import argparse
from pathlib import Path

import numpy as np
import torch
import yaml

from baseline_express4d import cubic_interpolation, linear_interpolation
from dataset_express4d import get_dataloader, load_vector_52
from main_model import CSDI_Express4D


def resolve_config_path(path):
    candidate = Path(path)
    if candidate.is_file():
        return candidate
    script_relative = Path(__file__).resolve().parent / path
    if script_relative.is_file():
        return script_relative
    raise FileNotFoundError(f"Config file not found: {path}")


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
    keys = items[0].keys()
    return {key: float(np.mean([item[key] for item in items])) for key in keys}


def run_eval_test(model, config, device, num_samples):
    _, test_loader = get_dataloader(
        config,
        batch_size=config["train"]["batch_size"],
        num_workers=config["train"].get("num_workers", 0),
    )
    model.eval()
    model_metrics = []
    linear_metrics = []
    cubic_metrics = []
    with torch.no_grad():
        for batch in test_loader:
            samples, observed_data, _, _, _ = model.evaluate(batch, num_samples)
            gt_full = observed_data.permute(0, 2, 1)
            pred_full = gt_full.clone()
            pred_samples = samples.mean(dim=1).permute(0, 2, 1)
            pred_full[:, 1:11] = pred_samples[:, 1:11]

            start = gt_full[:, 0]
            end = gt_full[:, -1]
            gt_middle = gt_full[:, 1:11]
            pred_middle = pred_full[:, 1:11]
            model_metrics.append(metric_dict(pred_middle, gt_middle, start, end))

            linear = np.stack(
                [linear_interpolation(s, e) for s, e in zip(start.cpu().numpy(), end.cpu().numpy())],
                axis=0,
            )
            linear = torch.from_numpy(linear).to(device)
            linear_metrics.append(metric_dict(linear, gt_middle, start, end))

            cubic = np.stack(
                [cubic_interpolation(s, e) for s, e in zip(start.cpu().numpy(), end.cpu().numpy())],
                axis=0,
            )
            cubic = torch.from_numpy(cubic).to(device)
            cubic_metrics.append(metric_dict(cubic, gt_middle, start, end))

    print("model:", average_metrics(model_metrics))
    print("linear_baseline:", average_metrics(linear_metrics))
    print("cubic_baseline:", average_metrics(cubic_metrics))


def main():
    parser = argparse.ArgumentParser(description="Sample Express4D middle blendshape frames")
    parser.add_argument("--config", type=str, default="config/express4d.yaml")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--input_start", type=str, default="")
    parser.add_argument("--input_end", type=str, default="")
    parser.add_argument("--duration", "--duraction", dest="duration", type=float, default=None)
    parser.add_argument("--output", type=str, default="output_middle.npy")
    parser.add_argument("--num_samples", type=int, default=1)
    parser.add_argument("--eval_test", action="store_true")
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    config_path = resolve_config_path(args.config)
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    model = CSDI_Express4D(
        config,
        args.device,
        target_dim=config["dataset"].get("num_features", 52),
    ).to(args.device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=args.device))

    if args.eval_test:
        run_eval_test(model, config, args.device, args.num_samples)
        return

    if not args.input_start or not args.input_end or args.duration is None:
        raise ValueError("--input_start, --input_end, and --duration are required unless --eval_test is set")

    dataset_config = config["dataset"]
    start = load_vector_52(
        args.input_start,
        clamp=dataset_config.get("clamp", True),
        clamp_min=dataset_config.get("clamp_min", 0.0),
        clamp_max=dataset_config.get("clamp_max", 1.0),
    )
    end = load_vector_52(
        args.input_end,
        clamp=dataset_config.get("clamp", True),
        clamp_min=dataset_config.get("clamp_min", 0.0),
        clamp_max=dataset_config.get("clamp_max", 1.0),
    )

    model.eval()
    middle = model.generate_middle(
        torch.from_numpy(start),
        torch.from_numpy(end),
        args.duration,
        num_samples=args.num_samples,
    )
    middle = middle.cpu().numpy()
    if args.num_samples == 1:
        middle = middle[0]
    np.save(args.output, middle.astype(np.float32))
    print(f"saved {middle.shape} to {args.output}")


if __name__ == "__main__":
    main()
