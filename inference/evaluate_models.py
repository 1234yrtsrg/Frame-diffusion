import argparse
import importlib
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
CSDI_DIR = REPO_ROOT / "CSDI"
if str(CSDI_DIR) not in sys.path:
    sys.path.insert(0, str(CSDI_DIR))

from main_model import CSDI_Express4D  # noqa: E402


METHODS = {
    "express4d_duration": {
        "config": "CSDI/config/express4d.yaml",
        "dataset_module": "dataset_express4d",
    },
    "express4d_condition": {
        "config": "CSDI/config/express4d_condition.yaml",
        "dataset_module": "dataset_express4d_condition",
    },
    "keyframe_dataset_60fps": {
        "config": "CSDI/config/keyframe_dataset_60fps.yaml",
        "dataset_module": "dataset_keyframe_dataset_60fps",
    },
}
EVAL_DATASET_METHOD = "keyframe_dataset_60fps"

DEFAULT_CONDITION_GAPS = {
    1: 240,
    3: 24,
}


def resolve_path(path):
    path = Path(path)
    if path.is_absolute():
        return path
    return REPO_ROOT / path


def load_config(path):
    path = resolve_path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Config not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
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

    if all(key.startswith("module.") for key in state):
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


def get_dataset(method, config, split, seed, batch_size, num_workers):
    module = importlib.import_module(METHODS[method]["dataset_module"])
    train_loader, test_loader = module.get_dataloader(
        config,
        seed=seed,
        batch_size=batch_size,
        num_workers=num_workers,
    )
    if split == "train":
        return module, train_loader.dataset
    if split == "test":
        return module, test_loader.dataset
    raise ValueError(f"Unsupported split: {split}")


def apply_eval_dataset_overrides(config, args):
    if args.dataset_root is not None:
        config["dataset"]["root"] = args.dataset_root
    if args.data_dirs is not None:
        config["dataset"]["data_dirs"] = [
            item.strip() for item in args.data_dirs.split(",") if item.strip()
        ]
    return config


def _global_ssim_parts(pred, target, data_range):
    pred = pred.reshape(pred.shape[0], -1)
    target = target.reshape(target.shape[0], -1)

    c1 = (0.01 * data_range) ** 2
    c2 = (0.03 * data_range) ** 2

    mu_x = pred.mean(dim=1)
    mu_y = target.mean(dim=1)
    x_centered = pred - mu_x[:, None]
    y_centered = target - mu_y[:, None]

    sigma_x = (x_centered * x_centered).mean(dim=1)
    sigma_y = (y_centered * y_centered).mean(dim=1)
    sigma_xy = (x_centered * y_centered).mean(dim=1)

    luminance = (2 * mu_x * mu_y + c1) / (mu_x * mu_x + mu_y * mu_y + c1)
    contrast_structure = (2 * sigma_xy + c2) / (sigma_x + sigma_y + c2)
    ssim = luminance * contrast_structure
    return luminance.clamp_min(1e-6), contrast_structure.clamp_min(1e-6), ssim


def ssim(pred, target, data_range):
    _, _, values = _global_ssim_parts(pred, target, data_range)
    return values.clamp(-1.0, 1.0)


def ms_ssim(pred, target, data_range):
    pred = pred.unsqueeze(1)
    target = target.unsqueeze(1)
    weights = torch.tensor(
        [0.0448, 0.2856, 0.3001, 0.2363, 0.1333],
        device=pred.device,
        dtype=pred.dtype,
    )

    luminance_values = []
    contrast_structure_values = []
    while True:
        luminance, contrast_structure, _ = _global_ssim_parts(pred, target, data_range)
        luminance_values.append(luminance)
        contrast_structure_values.append(contrast_structure)
        if len(luminance_values) == len(weights) or min(pred.shape[-2:]) < 2:
            break
        pred = torch.nn.functional.avg_pool2d(pred, kernel_size=2, stride=2)
        target = torch.nn.functional.avg_pool2d(target, kernel_size=2, stride=2)

    scale_count = len(luminance_values)
    scale_weights = weights[:scale_count]
    scale_weights = scale_weights / scale_weights.sum()

    score = luminance_values[-1].pow(scale_weights[-1])
    for idx in range(scale_count - 1):
        score = score * contrast_structure_values[idx].pow(scale_weights[idx])
    return score.clamp(0.0, 1.0)


class MetricAccumulator:
    def __init__(self, data_range):
        self.data_range = float(data_range)
        self.abs_sum = 0.0
        self.sq_sum = 0.0
        self.count = 0
        self.ssim_sum = 0.0
        self.ms_ssim_sum = 0.0
        self.num_sequences = 0

    def update(self, pred, target):
        pred = pred.float()
        target = target.float()
        diff = pred - target
        self.abs_sum += torch.abs(diff).sum().item()
        self.sq_sum += (diff * diff).sum().item()
        self.count += diff.numel()

        self.ssim_sum += ssim(pred, target, self.data_range).sum().item()
        self.ms_ssim_sum += ms_ssim(pred, target, self.data_range).sum().item()
        self.num_sequences += pred.shape[0]

    def compute(self):
        if self.count == 0 or self.num_sequences == 0:
            raise ValueError("No samples were evaluated")

        mae = self.abs_sum / self.count
        mse = self.sq_sum / self.count
        psnr = math.inf if mse == 0 else 20.0 * math.log10(self.data_range) - 10.0 * math.log10(mse)
        return {
            "psnr": float(psnr),
            "ssim": float(self.ssim_sum / self.num_sequences),
            "ms_ssim": float(self.ms_ssim_sum / self.num_sequences),
            "mae_l1": float(mae),
            "mse_l2": float(mse),
        }


def default_data_range(config):
    dataset_config = config.get("dataset", {})
    clamp_min = float(dataset_config.get("clamp_min", 0.0))
    clamp_max = float(dataset_config.get("clamp_max", 1.0))
    value = clamp_max - clamp_min
    return value if value > 0 else 1.0


def condition_gap(config, condition, override=None):
    if override is not None:
        return int(override)
    condition_gaps = config.get("dataset", {}).get("condition_gaps", {})
    normalized = {int(key): int(value) for key, value in condition_gaps.items()}
    if int(condition) in normalized:
        return normalized[int(condition)]
    if int(condition) in DEFAULT_CONDITION_GAPS:
        return DEFAULT_CONDITION_GAPS[int(condition)]
    raise ValueError(f"No gap configured for condition={condition}")


def sample_tuple_parts(sample):
    if len(sample) == 4:
        sequence_id, start_idx, end_idx, gap = sample
        return int(sequence_id), int(start_idx), int(end_idx), int(gap), None
    if len(sample) == 5:
        sequence_id, start_idx, end_idx, gap, condition = sample
        return int(sequence_id), int(start_idx), int(end_idx), int(gap), int(condition)
    raise ValueError(f"Unsupported dataset sample tuple: {sample}")


def keyframes_for_sequence(dataset, sequence_id, sequence):
    if hasattr(dataset, "sequence_keyframes"):
        return dataset.sequence_keyframes[sequence_id]
    return sequence.get("keyframes")


def sample_coarse_ground_truth(module, dataset, sample_index):
    sequence_id, start_idx, end_idx, _, _ = sample_tuple_parts(dataset.samples[sample_index])
    sequence = dataset.sequences[sequence_id]
    data = sequence["data"]
    keyframes = keyframes_for_sequence(dataset, sequence_id, sequence)
    seq_len = int(getattr(dataset, "seq_len", 12))

    if keyframes is None:
        sequence_gt = module.sample_linear_sequence(data, start_idx, end_idx, seq_len)
        positions = np.linspace(start_idx, end_idx, seq_len, dtype=np.float32)
    else:
        sequence_gt, positions, _ = module.sample_nearest_sequence_with_keyframes(
            data,
            start_idx,
            end_idx,
            keyframes,
            seq_len=seq_len,
        )
    return data, keyframes, sequence_gt.astype(np.float32), positions.astype(np.float32)


def sample_fine_ground_truth(module, data, keyframes, coarse_positions, seq_len=12):
    parts = []
    for pair_index in range(len(coarse_positions) - 1):
        start_pos = float(coarse_positions[pair_index])
        end_pos = float(coarse_positions[pair_index + 1])
        if keyframes is None:
            segment = module.sample_linear_sequence(data, start_pos, end_pos, seq_len)
        else:
            segment, _, _ = module.sample_nearest_sequence_with_keyframes(
                data,
                int(round(start_pos)),
                int(round(end_pos)),
                keyframes,
                seq_len=seq_len,
            )
        parts.append(segment if pair_index == 0 else segment[1:])
    return np.concatenate(parts, axis=0).astype(np.float32)


def selected_sample_indices(dataset, coarse_condition, coarse_gap):
    indices = []
    for index, sample in enumerate(dataset.samples):
        _, _, _, gap, condition = sample_tuple_parts(sample)
        if condition is None:
            keep = gap == coarse_gap
        else:
            keep = condition == coarse_condition
        if keep:
            indices.append(index)
    if not indices:
        raise ValueError(
            f"No evaluation samples found for condition={coarse_condition} / gap={coarse_gap}"
        )
    return indices


def generated_middle(model, start, end, duration, condition, num_samples):
    if model.use_condition:
        output = model.generate_middle(
            start,
            end,
            None,
            num_samples=num_samples,
            condition=condition,
        )
    else:
        output = model.generate_middle(
            start,
            end,
            duration,
            num_samples=num_samples,
        )
    if output.dim() == 4:
        return output.mean(dim=1)
    if output.dim() == 3:
        return output
    raise ValueError(f"Unexpected generated middle shape: {tuple(output.shape)}")


def stitch_segments(segments):
    first = segments[:, 0]
    rest = segments[:, 1:, 1:].reshape(segments.shape[0], -1, segments.shape[-1])
    return torch.cat([first, rest], dim=1)


def two_stage_predict(
    model,
    coarse_gt,
    coarse_positions,
    fps,
    coarse_condition,
    fine_condition,
    num_samples,
    device,
):
    batch_size = coarse_gt.shape[0]
    start = coarse_gt[:, 0].to(device)
    end = coarse_gt[:, -1].to(device)

    coarse_duration = (coarse_positions[:, -1] - coarse_positions[:, 0]).to(device).float() / float(fps)
    coarse_condition_tensor = torch.full(
        (batch_size,),
        float(coarse_condition),
        device=device,
    )
    coarse_middle = generated_middle(
        model,
        start,
        end,
        duration=coarse_duration,
        condition=coarse_condition_tensor,
        num_samples=num_samples,
    )
    coarse_pred = torch.cat([start[:, None], coarse_middle, end[:, None]], dim=1)

    segment_start = coarse_pred[:, :-1].reshape(-1, coarse_pred.shape[-1])
    segment_end = coarse_pred[:, 1:].reshape(-1, coarse_pred.shape[-1])
    interval_durations = (
        (coarse_positions[:, 1:] - coarse_positions[:, :-1]).reshape(-1).to(device).float()
        / float(fps)
    )
    fine_condition_tensor = torch.full(
        (segment_start.shape[0],),
        float(fine_condition),
        device=device,
    )
    fine_middle = generated_middle(
        model,
        segment_start,
        segment_end,
        duration=interval_durations,
        condition=fine_condition_tensor,
        num_samples=num_samples,
    )
    fine_segments = torch.cat(
        [segment_start[:, None], fine_middle, segment_end[:, None]],
        dim=1,
    ).reshape(batch_size, coarse_pred.shape[1] - 1, -1, coarse_pred.shape[-1])
    return stitch_segments(fine_segments)


def build_eval_context(args):
    config = apply_eval_dataset_overrides(load_config(args.eval_config), args)
    config["seed"] = args.seed
    module, dataset = get_dataset(
        EVAL_DATASET_METHOD,
        config,
        split=args.split,
        seed=args.seed,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    coarse_condition = int(args.coarse_condition)
    fine_condition = int(args.fine_condition)
    coarse_gap = condition_gap(config, coarse_condition, args.coarse_gap)
    selected_indices = selected_sample_indices(dataset, coarse_condition, coarse_gap)
    return {
        "config": config,
        "module": module,
        "dataset": dataset,
        "coarse_condition": coarse_condition,
        "fine_condition": fine_condition,
        "coarse_gap": coarse_gap,
        "selected_indices": selected_indices,
        "fps": float(config.get("dataset", {}).get("fps", 60)),
        "seq_len": int(config.get("dataset", {}).get("seq_len", 12)),
    }


def evaluate_checkpoint(method, checkpoint, config_path, eval_context, args, device):
    model_config = load_config(config_path)
    model_config["seed"] = args.seed
    model = load_model(model_config, checkpoint, device)
    accumulator = MetricAccumulator(
        data_range=args.data_range
        if args.data_range is not None
        else default_data_range(eval_context["config"])
    )
    module = eval_context["module"]
    dataset = eval_context["dataset"]
    selected_indices = eval_context["selected_indices"]

    with torch.no_grad():
        for batch_idx, start_offset in enumerate(range(0, len(selected_indices), args.batch_size)):
            if args.max_batches is not None and batch_idx >= args.max_batches:
                break
            batch_indices = selected_indices[start_offset : start_offset + args.batch_size]
            coarse_gt_items = []
            coarse_position_items = []
            fine_gt_items = []
            for sample_index in batch_indices:
                data, keyframes, coarse_gt_np, coarse_positions_np = sample_coarse_ground_truth(
                    module,
                    dataset,
                    sample_index,
                )
                fine_gt_np = sample_fine_ground_truth(
                    module,
                    data,
                    keyframes,
                    coarse_positions_np,
                    seq_len=eval_context["seq_len"],
                )
                coarse_gt_items.append(coarse_gt_np)
                coarse_position_items.append(coarse_positions_np)
                fine_gt_items.append(fine_gt_np)

            coarse_gt = torch.from_numpy(np.stack(coarse_gt_items, axis=0)).float()
            coarse_positions = torch.from_numpy(np.stack(coarse_position_items, axis=0)).float()
            fine_gt = torch.from_numpy(np.stack(fine_gt_items, axis=0)).to(device).float()
            fine_pred = two_stage_predict(
                model,
                coarse_gt,
                coarse_positions,
                fps=eval_context["fps"],
                coarse_condition=eval_context["coarse_condition"],
                fine_condition=eval_context["fine_condition"],
                num_samples=args.num_samples,
                device=device,
            )

            accumulator.update(fine_pred[:, 1:-1], fine_gt[:, 1:-1])

    return {
        "checkpoint": str(resolve_path(checkpoint)),
        "model_config": str(resolve_path(config_path)),
        "model_dataset": model_config["dataset"].get("name", method),
        "eval_dataset": eval_context["config"]["dataset"].get("name", EVAL_DATASET_METHOD),
        "eval_config": str(resolve_path(args.eval_config)),
        "split": args.split,
        "num_samples": args.num_samples,
        "coarse_condition": eval_context["coarse_condition"],
        "fine_condition": eval_context["fine_condition"],
        "coarse_gap": eval_context["coarse_gap"],
        "num_selected_windows": len(selected_indices),
        "refined_sequence_length": int((eval_context["seq_len"] - 1) ** 2 + 1),
        "metrics": accumulator.compute(),
    }


def checkpoint_args(args):
    return {
        "express4d_duration": args.express4d_duration_checkpoint,
        "express4d_condition": args.express4d_condition_checkpoint,
        "keyframe_dataset_60fps": args.keyframe_dataset_60fps_checkpoint,
    }


def config_args(args):
    return {
        "express4d_duration": args.express4d_duration_config,
        "express4d_condition": args.express4d_condition_config,
        "keyframe_dataset_60fps": args.keyframe_dataset_60fps_config,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate the three training methods with two-stage condition=1 -> condition=3 inference."
    )
    parser.add_argument("--express4d_duration_checkpoint", default="")
    parser.add_argument("--express4d_condition_checkpoint", default="")
    parser.add_argument("--keyframe_dataset_60fps_checkpoint", default="")
    parser.add_argument("--express4d_duration_config", default=METHODS["express4d_duration"]["config"])
    parser.add_argument("--express4d_condition_config", default=METHODS["express4d_condition"]["config"])
    parser.add_argument("--keyframe_dataset_60fps_config", default=METHODS["keyframe_dataset_60fps"]["config"])
    parser.add_argument(
        "--eval_config",
        default=METHODS[EVAL_DATASET_METHOD]["config"],
        help="Evaluation dataset config. Defaults to keyframe_dataset_60fps.",
    )
    parser.add_argument("--split", choices=["train", "test"], default="test")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--num_samples", type=int, default=1)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--data_range", type=float, default=None)
    parser.add_argument("--coarse_condition", type=int, default=1)
    parser.add_argument("--fine_condition", type=int, default=3)
    parser.add_argument("--coarse_gap", type=int, default=None)
    parser.add_argument("--max_batches", type=int, default=None)
    parser.add_argument("--output", default="outputs/model_eval/metrics.json")
    parser.add_argument("--dataset_root", default=None, help="Override keyframe_dataset_60fps dataset root.")
    parser.add_argument("--data_dirs", default=None, help="Override keyframe_dataset_60fps data_dirs, comma-separated.")
    args = parser.parse_args()

    device = args.device
    if device == "auto":
        device = "cuda:0" if torch.cuda.is_available() else "cpu"

    checkpoints = {key: value for key, value in checkpoint_args(args).items() if value}
    if not checkpoints:
        raise ValueError("Provide at least one *_checkpoint argument.")

    configs = config_args(args)
    eval_context = build_eval_context(args)
    payload = {
        "device": device,
        "seed": args.seed,
        "split": args.split,
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "max_batches": args.max_batches,
        "eval_dataset": eval_context["config"]["dataset"].get("name", EVAL_DATASET_METHOD),
        "eval_config": str(resolve_path(args.eval_config)),
        "coarse_condition": eval_context["coarse_condition"],
        "fine_condition": eval_context["fine_condition"],
        "coarse_gap": eval_context["coarse_gap"],
        "num_selected_windows": len(eval_context["selected_indices"]),
        "methods": {},
    }

    for method, checkpoint in checkpoints.items():
        payload["methods"][method] = evaluate_checkpoint(
            method,
            checkpoint,
            configs[method],
            eval_context,
            args,
            device,
        )

    output_path = resolve_path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print(json.dumps(payload["methods"], indent=2))
    print(f"saved metrics to {output_path}")


if __name__ == "__main__":
    main()
