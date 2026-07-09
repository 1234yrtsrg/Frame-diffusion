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


class TimelineMetricAccumulator:
    def __init__(self, num_features=52, writeback_known_frames=True):
        self.num_features = int(num_features)
        self.writeback_known_frames = bool(writeback_known_frames)
        self.num_sequences = 0
        self.num_frames = 0
        self.num_non_keyframes = 0
        self.num_known_frames = 0

        self.timeline_abs_sum = 0.0
        self.timeline_sq_sum = 0.0
        self.timeline_count = 0

        self.non_keyframe_abs_sum = 0.0
        self.non_keyframe_sq_sum = 0.0
        self.non_keyframe_count = 0
        self.non_keyframe_feature_abs_sum = np.zeros(self.num_features, dtype=np.float64)
        self.non_keyframe_feature_count = 0

        self.velocity_abs_sum = 0.0
        self.velocity_sq_sum = 0.0
        self.velocity_count = 0

        self.acceleration_abs_sum = 0.0
        self.acceleration_sq_sum = 0.0
        self.acceleration_count = 0

        self.boundary_velocity_abs_sum = 0.0
        self.boundary_velocity_sq_sum = 0.0
        self.boundary_velocity_count = 0
        self.num_boundary_velocity_intervals = 0

        self.known_abs_before_writeback = 0.0
        self.known_sq_before_writeback = 0.0
        self.known_count = 0
        self.known_abs_after_writeback = 0.0
        self.known_sq_after_writeback = 0.0

        self.endpoint_abs_before_writeback = 0.0
        self.endpoint_sq_before_writeback = 0.0
        self.endpoint_count = 0
        self.endpoint_abs_after_writeback = 0.0
        self.endpoint_sq_after_writeback = 0.0

    @staticmethod
    def _add_abs_sq(diff):
        return torch.abs(diff).sum().item(), (diff * diff).sum().item(), diff.numel()

    @staticmethod
    def _mean(abs_sum, count):
        return float(abs_sum / count) if count else None

    @staticmethod
    def _mse(sq_sum, count):
        return float(sq_sum / count) if count else None

    def update(self, pred_raw, target, known_mask):
        pred_raw = pred_raw.float()
        target = target.float()
        known_mask = known_mask.to(device=pred_raw.device, dtype=torch.bool)
        if pred_raw.shape != target.shape:
            raise ValueError(f"pred/target shape mismatch: {tuple(pred_raw.shape)} vs {tuple(target.shape)}")
        if known_mask.shape[0] != pred_raw.shape[0]:
            raise ValueError(f"known_mask length mismatch: {known_mask.shape[0]} vs {pred_raw.shape[0]}")

        self.num_sequences += 1
        self.num_frames += pred_raw.shape[0]
        self.num_known_frames += int(known_mask.sum().item())

        known_diff_before = pred_raw[known_mask] - target[known_mask]
        if known_diff_before.numel() > 0:
            abs_sum, sq_sum, count = self._add_abs_sq(known_diff_before)
            self.known_abs_before_writeback += abs_sum
            self.known_sq_before_writeback += sq_sum
            self.known_count += count

        endpoint_mask = torch.zeros_like(known_mask)
        endpoint_mask[0] = True
        endpoint_mask[-1] = True
        endpoint_diff_before = pred_raw[endpoint_mask] - target[endpoint_mask]
        abs_sum, sq_sum, count = self._add_abs_sq(endpoint_diff_before)
        self.endpoint_abs_before_writeback += abs_sum
        self.endpoint_sq_before_writeback += sq_sum
        self.endpoint_count += count

        pred = pred_raw.clone()
        if self.writeback_known_frames:
            pred[known_mask] = target[known_mask]

        known_diff_after = pred[known_mask] - target[known_mask]
        if known_diff_after.numel() > 0:
            abs_sum, sq_sum, _ = self._add_abs_sq(known_diff_after)
            self.known_abs_after_writeback += abs_sum
            self.known_sq_after_writeback += sq_sum

        endpoint_diff_after = pred[endpoint_mask] - target[endpoint_mask]
        abs_sum, sq_sum, _ = self._add_abs_sq(endpoint_diff_after)
        self.endpoint_abs_after_writeback += abs_sum
        self.endpoint_sq_after_writeback += sq_sum

        diff = pred - target
        abs_sum, sq_sum, count = self._add_abs_sq(diff)
        self.timeline_abs_sum += abs_sum
        self.timeline_sq_sum += sq_sum
        self.timeline_count += count

        non_keyframe_mask = ~known_mask
        self.num_non_keyframes += int(non_keyframe_mask.sum().item())
        non_keyframe_diff = diff[non_keyframe_mask]
        if non_keyframe_diff.numel() > 0:
            abs_sum, sq_sum, count = self._add_abs_sq(non_keyframe_diff)
            self.non_keyframe_abs_sum += abs_sum
            self.non_keyframe_sq_sum += sq_sum
            self.non_keyframe_count += count
            self.non_keyframe_feature_abs_sum += torch.abs(non_keyframe_diff).sum(dim=0).cpu().numpy()
            self.non_keyframe_feature_count += int(non_keyframe_diff.shape[0])

        if pred.shape[0] >= 2:
            pred_velocity = pred[1:] - pred[:-1]
            target_velocity = target[1:] - target[:-1]
            velocity_diff = pred_velocity - target_velocity
            abs_sum, sq_sum, count = self._add_abs_sq(velocity_diff)
            self.velocity_abs_sum += abs_sum
            self.velocity_sq_sum += sq_sum
            self.velocity_count += count

            boundary_interval_mask = known_mask[1:] | known_mask[:-1]
            boundary_velocity_diff = velocity_diff[boundary_interval_mask]
            if boundary_velocity_diff.numel() > 0:
                abs_sum, sq_sum, count = self._add_abs_sq(boundary_velocity_diff)
                self.boundary_velocity_abs_sum += abs_sum
                self.boundary_velocity_sq_sum += sq_sum
                self.boundary_velocity_count += count
                self.num_boundary_velocity_intervals += int(boundary_interval_mask.sum().item())

        if pred.shape[0] >= 3:
            pred_acceleration = pred[2:] - 2.0 * pred[1:-1] + pred[:-2]
            target_acceleration = target[2:] - 2.0 * target[1:-1] + target[:-2]
            acceleration_diff = pred_acceleration - target_acceleration
            abs_sum, sq_sum, count = self._add_abs_sq(acceleration_diff)
            self.acceleration_abs_sum += abs_sum
            self.acceleration_sq_sum += sq_sum
            self.acceleration_count += count

    def compute(self):
        if self.timeline_count == 0 or self.num_sequences == 0:
            raise ValueError("No samples were evaluated")

        per_feature = None
        if self.non_keyframe_feature_count:
            per_feature = (
                self.non_keyframe_feature_abs_sum / float(self.non_keyframe_feature_count)
            ).astype(float).tolist()

        return {
            "timeline_resampled_mae_l1": self._mean(self.timeline_abs_sum, self.timeline_count),
            "timeline_resampled_mse_l2": self._mse(self.timeline_sq_sum, self.timeline_count),
            "non_keyframe_mae_l1": self._mean(self.non_keyframe_abs_sum, self.non_keyframe_count),
            "non_keyframe_mse_l2": self._mse(self.non_keyframe_sq_sum, self.non_keyframe_count),
            "timeline_resampled_velocity_mae": self._mean(self.velocity_abs_sum, self.velocity_count),
            "timeline_resampled_velocity_mse": self._mse(self.velocity_sq_sum, self.velocity_count),
            "timeline_resampled_acceleration_mae": self._mean(
                self.acceleration_abs_sum,
                self.acceleration_count,
            ),
            "timeline_resampled_acceleration_mse": self._mse(
                self.acceleration_sq_sum,
                self.acceleration_count,
            ),
            "boundary_velocity_mae": self._mean(
                self.boundary_velocity_abs_sum,
                self.boundary_velocity_count,
            ),
            "boundary_velocity_mse": self._mse(
                self.boundary_velocity_sq_sum,
                self.boundary_velocity_count,
            ),
            "per_feature_non_keyframe_mae": per_feature,
            "known_frame_mae_before_writeback": self._mean(
                self.known_abs_before_writeback,
                self.known_count,
            ),
            "known_frame_mse_before_writeback": self._mse(
                self.known_sq_before_writeback,
                self.known_count,
            ),
            "known_frame_mae_after_writeback": self._mean(
                self.known_abs_after_writeback,
                self.known_count,
            ),
            "known_frame_mse_after_writeback": self._mse(
                self.known_sq_after_writeback,
                self.known_count,
            ),
            "endpoint_mae_before_writeback": self._mean(
                self.endpoint_abs_before_writeback,
                self.endpoint_count,
            ),
            "endpoint_mse_before_writeback": self._mse(
                self.endpoint_sq_before_writeback,
                self.endpoint_count,
            ),
            "endpoint_mae_after_writeback": self._mean(
                self.endpoint_abs_after_writeback,
                self.endpoint_count,
            ),
            "endpoint_mse_after_writeback": self._mse(
                self.endpoint_sq_after_writeback,
                self.endpoint_count,
            ),
            "counts": {
                "num_sequences": int(self.num_sequences),
                "num_frames": int(self.num_frames),
                "num_non_keyframes": int(self.num_non_keyframes),
                "num_known_frames": int(self.num_known_frames),
                "num_boundary_velocity_intervals": int(self.num_boundary_velocity_intervals),
            },
        }


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


def known_frame_mask(keyframes, start_idx, end_idx):
    length = int(end_idx) - int(start_idx) + 1
    if length <= 0:
        raise ValueError(f"Invalid frame interval: start={start_idx}, end={end_idx}")
    mask = np.zeros(length, dtype=bool)
    mask[0] = True
    mask[-1] = True
    if keyframes is not None:
        for keyframe in keyframes:
            keyframe = int(keyframe)
            if int(start_idx) <= keyframe <= int(end_idx):
                mask[keyframe - int(start_idx)] = True
    return mask


def resample_sequence(sequence, target_len):
    sequence = sequence.float()
    target_len = int(target_len)
    if target_len <= 0:
        raise ValueError(f"target_len must be positive, got {target_len}")
    if sequence.shape[0] == target_len:
        return sequence.clone()
    if sequence.shape[0] == 1:
        return sequence.expand(target_len, -1).clone()
    return torch.nn.functional.interpolate(
        sequence.T.unsqueeze(0),
        size=target_len,
        mode="linear",
        align_corners=True,
    ).squeeze(0).T


def timeline_item(module, dataset, sample_index):
    data, keyframes, coarse_gt_np, coarse_positions_np, start_idx, end_idx = sample_coarse_ground_truth(
        module,
        dataset,
        sample_index,
    )
    return {
        "coarse_gt": coarse_gt_np,
        "coarse_positions": coarse_positions_np,
        "target": data[start_idx : end_idx + 1].astype(np.float32, copy=False),
        "known_mask": known_frame_mask(keyframes, start_idx, end_idx),
        "start_idx": start_idx,
        "end_idx": end_idx,
    }


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
    return (
        data,
        keyframes,
        sequence_gt.astype(np.float32),
        positions.astype(np.float32),
        int(start_idx),
        int(end_idx),
    )


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


def progress_print(args, message):
    if not args.quiet:
        print(f"[shard {args.shard_index + 1}/{args.num_shards}] {message}", flush=True)


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
    fine_segment_batch_size=None,
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
    fine_segment_batch_size = int(fine_segment_batch_size or segment_start.shape[0])
    fine_middle_items = []
    for start_offset in range(0, segment_start.shape[0], fine_segment_batch_size):
        end_offset = start_offset + fine_segment_batch_size
        fine_middle_items.append(
            generated_middle(
                model,
                segment_start[start_offset:end_offset],
                segment_end[start_offset:end_offset],
                duration=interval_durations[start_offset:end_offset],
                condition=fine_condition_tensor[start_offset:end_offset],
                num_samples=num_samples,
            )
        )
    fine_middle = torch.cat(fine_middle_items, dim=0)
    fine_segments = torch.cat(
        [segment_start[:, None], fine_middle, segment_end[:, None]],
        dim=1,
    ).reshape(batch_size, coarse_pred.shape[1] - 1, -1, coarse_pred.shape[-1])
    return stitch_segments(fine_segments)


def linear_interpolate_known_frames(target, known_mask):
    target = target.float()
    known_mask = known_mask.to(device=target.device, dtype=torch.bool)
    known_indices = torch.nonzero(known_mask, as_tuple=False).flatten()
    if known_indices.numel() < 2:
        raise ValueError("Linear interpolation baseline requires at least two known frames")

    pred = torch.empty_like(target)
    for pair_index in range(known_indices.numel() - 1):
        start_idx = int(known_indices[pair_index].item())
        end_idx = int(known_indices[pair_index + 1].item())
        span = end_idx - start_idx
        if span <= 0:
            continue
        alpha = torch.linspace(0.0, 1.0, span + 1, device=target.device, dtype=target.dtype).unsqueeze(1)
        start = target[start_idx].unsqueeze(0)
        end = target[end_idx].unsqueeze(0)
        segment = start * (1.0 - alpha) + end * alpha
        pred[start_idx : end_idx + 1] = segment
    return pred


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
    if args.num_shards <= 0:
        raise ValueError("--num_shards must be positive")
    if args.shard_index < 0 or args.shard_index >= args.num_shards:
        raise ValueError("--shard_index must satisfy 0 <= shard_index < num_shards")
    total_selected_windows = len(selected_indices)
    selected_indices = selected_indices[args.shard_index :: args.num_shards]
    if not selected_indices:
        raise ValueError(
            f"No evaluation samples assigned to shard {args.shard_index}/{args.num_shards}"
        )
    return {
        "config": config,
        "module": module,
        "dataset": dataset,
        "coarse_condition": coarse_condition,
        "fine_condition": fine_condition,
        "coarse_gap": coarse_gap,
        "selected_indices": selected_indices,
        "total_selected_windows": total_selected_windows,
        "fps": float(config.get("dataset", {}).get("fps", 60)),
        "seq_len": int(config.get("dataset", {}).get("seq_len", 12)),
    }


def evaluate_checkpoint(method, checkpoint, config_path, eval_context, args, device):
    model_config = load_config(config_path)
    model_config["seed"] = args.seed
    model = load_model(model_config, checkpoint, device)
    accumulator = TimelineMetricAccumulator(
        num_features=model_config["dataset"].get("num_features", 52),
        writeback_known_frames=True,
    )
    module = eval_context["module"]
    dataset = eval_context["dataset"]
    selected_indices = eval_context["selected_indices"]
    total_batches = math.ceil(len(selected_indices) / args.batch_size)
    if args.max_batches is not None:
        total_batches = min(total_batches, args.max_batches)
    progress_print(
        args,
        f"{method}: start {len(selected_indices)} windows, {total_batches} batches, checkpoint={checkpoint}",
    )

    with torch.no_grad():
        for batch_idx, start_offset in enumerate(range(0, len(selected_indices), args.batch_size)):
            if args.max_batches is not None and batch_idx >= args.max_batches:
                break
            progress_print(args, f"{method}: batch {batch_idx + 1}/{total_batches}")
            batch_indices = selected_indices[start_offset : start_offset + args.batch_size]
            coarse_gt_items = []
            coarse_position_items = []
            timeline_items = []
            for sample_index in batch_indices:
                item = timeline_item(module, dataset, sample_index)
                coarse_gt_items.append(item["coarse_gt"])
                coarse_position_items.append(item["coarse_positions"])
                timeline_items.append(item)

            coarse_gt = torch.from_numpy(np.stack(coarse_gt_items, axis=0)).float()
            coarse_positions = torch.from_numpy(np.stack(coarse_position_items, axis=0)).float()
            fine_pred = two_stage_predict(
                model,
                coarse_gt,
                coarse_positions,
                fps=eval_context["fps"],
                coarse_condition=eval_context["coarse_condition"],
                fine_condition=eval_context["fine_condition"],
                num_samples=args.num_samples,
                device=device,
                fine_segment_batch_size=args.fine_segment_batch_size,
            )

            for item_index, item in enumerate(timeline_items):
                target = torch.from_numpy(item["target"]).to(device).float()
                known_mask = torch.from_numpy(item["known_mask"]).to(device)
                pred_resampled = resample_sequence(fine_pred[item_index], target.shape[0])
                accumulator.update(pred_resampled, target, known_mask)

    progress_print(args, f"{method}: done")
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
        "generated_sequence_length_before_timeline_resample": int((eval_context["seq_len"] - 1) ** 2 + 1),
        "timeline_resampled_to_original_60fps": True,
        "known_frames_written_back_before_metrics": True,
        "metrics": accumulator.compute(),
    }


def evaluate_linear_baseline(eval_context, args, device):
    accumulator = TimelineMetricAccumulator(
        num_features=eval_context["config"]["dataset"].get("num_features", 52),
        writeback_known_frames=True,
    )
    module = eval_context["module"]
    dataset = eval_context["dataset"]
    selected_indices = eval_context["selected_indices"]
    total_batches = math.ceil(len(selected_indices) / args.batch_size)
    if args.max_batches is not None:
        total_batches = min(total_batches, args.max_batches)
    progress_print(args, f"linear_interpolation: start {len(selected_indices)} windows, {total_batches} batches")

    for batch_idx, start_offset in enumerate(range(0, len(selected_indices), args.batch_size)):
        if args.max_batches is not None and batch_idx >= args.max_batches:
            break
        progress_print(args, f"linear_interpolation: batch {batch_idx + 1}/{total_batches}")
        batch_indices = selected_indices[start_offset : start_offset + args.batch_size]
        for sample_index in batch_indices:
            item = timeline_item(module, dataset, sample_index)
            target = torch.from_numpy(item["target"]).to(device).float()
            known_mask = torch.from_numpy(item["known_mask"]).to(device)
            pred = linear_interpolate_known_frames(target, known_mask)
            accumulator.update(pred, target, known_mask)

    progress_print(args, "linear_interpolation: done")
    return {
        "model_dataset": "linear_interpolation_baseline",
        "eval_dataset": eval_context["config"]["dataset"].get("name", EVAL_DATASET_METHOD),
        "eval_config": str(resolve_path(args.eval_config)),
        "split": args.split,
        "coarse_condition": eval_context["coarse_condition"],
        "fine_condition": eval_context["fine_condition"],
        "coarse_gap": eval_context["coarse_gap"],
        "num_selected_windows": len(selected_indices),
        "timeline_resampled_to_original_60fps": True,
        "known_frames_used_for_piecewise_interpolation": True,
        "known_frames_written_back_before_metrics": True,
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
    parser.add_argument("--coarse_condition", type=int, default=1)
    parser.add_argument("--fine_condition", type=int, default=3)
    parser.add_argument("--coarse_gap", type=int, default=None)
    parser.add_argument("--max_batches", type=int, default=None)
    parser.add_argument("--fine_segment_batch_size", type=int, default=512)
    parser.add_argument("--num_shards", type=int, default=1)
    parser.add_argument("--shard_index", type=int, default=0)
    parser.add_argument("--output", default="outputs/model_eval/metrics.json")
    parser.add_argument("--dataset_root", default=None, help="Override keyframe_dataset_60fps dataset root.")
    parser.add_argument("--data_dirs", default=None, help="Override keyframe_dataset_60fps data_dirs, comma-separated.")
    parser.add_argument("--quiet", action="store_true", help="Disable per-method and per-batch progress logs.")
    parser.add_argument(
        "--no_linear_baseline",
        action="store_true",
        help="Disable the piecewise linear interpolation baseline.",
    )
    args = parser.parse_args()

    device = args.device
    if device == "auto":
        device = "cuda:0" if torch.cuda.is_available() else "cpu"

    checkpoints = {key: value for key, value in checkpoint_args(args).items() if value}
    include_linear_baseline = not args.no_linear_baseline
    if not checkpoints and not include_linear_baseline:
        raise ValueError("Provide at least one *_checkpoint argument or enable the linear baseline.")

    configs = config_args(args)
    eval_context = build_eval_context(args)
    payload = {
        "device": device,
        "seed": args.seed,
        "split": args.split,
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "max_batches": args.max_batches,
        "fine_segment_batch_size": args.fine_segment_batch_size,
        "eval_dataset": eval_context["config"]["dataset"].get("name", EVAL_DATASET_METHOD),
        "eval_config": str(resolve_path(args.eval_config)),
        "coarse_condition": eval_context["coarse_condition"],
        "fine_condition": eval_context["fine_condition"],
        "coarse_gap": eval_context["coarse_gap"],
        "num_shards": args.num_shards,
        "shard_index": args.shard_index,
        "total_selected_windows": eval_context["total_selected_windows"],
        "num_selected_windows": len(eval_context["selected_indices"]),
        "include_linear_baseline": include_linear_baseline,
        "methods": {},
    }

    if include_linear_baseline:
        payload["methods"]["linear_interpolation"] = evaluate_linear_baseline(
            eval_context,
            args,
            device,
        )

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
