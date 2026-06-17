from collections import Counter
from pathlib import Path
import sys

import math
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Sampler

REPO_ROOT = Path(__file__).resolve().parents[1]
CSDI_DIR = REPO_ROOT / "CSDI"
if str(CSDI_DIR) not in sys.path:
    sys.path.insert(0, str(CSDI_DIR))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dataset_express4d import (
    _read_list,
    _resolve_entry,
    load_blendshape_file,
    resolve_dataset_root,
)
from train.express4d_condition.detect_keyframes import detect_blendshape_keyframes


def _normalize_int_mapping(mapping, default):
    if mapping is None:
        mapping = default
    return {int(key): int(value) for key, value in mapping.items()}


def _normalize_float_mapping(mapping, default):
    if mapping is None:
        mapping = default
    values = {int(key): float(value) for key, value in mapping.items()}
    total = sum(values.values())
    if total <= 0:
        raise ValueError("condition ratios must sum to a positive value")
    return {key: value / total for key, value in values.items()}


def sample_nearest_sequence_with_keyframes(data, start_idx, end_idx, keyframes, seq_len=12):
    """Sample 12 real frames and force in-window keyframes onto nearest slots."""
    interval = float(end_idx - start_idx) / float(seq_len - 1)
    base_positions = np.rint(start_idx + np.arange(seq_len, dtype=np.float32) * interval).astype(np.int64)
    base_positions[0] = int(start_idx)
    base_positions[-1] = int(end_idx)

    positions = base_positions.copy()
    keyframe_slots = np.zeros(seq_len, dtype=np.float32)
    available_slots = set(range(1, seq_len - 1))
    in_window_keyframes = [
        int(keyframe)
        for keyframe in keyframes
        if int(start_idx) < int(keyframe) < int(end_idx)
    ]

    for keyframe in in_window_keyframes:
        if not available_slots:
            break
        slot = min(
            available_slots,
            key=lambda candidate: (abs(int(base_positions[candidate]) - keyframe), candidate),
        )
        positions[slot] = keyframe
        keyframe_slots[slot] = 1.0
        available_slots.remove(slot)

    positions = np.clip(positions, 0, len(data) - 1).astype(np.int64)
    sampled = data[positions].astype(np.float32, copy=False)
    return sampled, positions.astype(np.float32), keyframe_slots


class Express4DConditionDataset(Dataset):
    """Express4D 12-frame windows with discrete temporal scale conditions."""

    def __init__(self, config, split="train"):
        dataset_config = config["dataset"]
        self.root = resolve_dataset_root(dataset_config["root"])
        self.data_dir = self.root / dataset_config.get("data_dir", "data")
        list_name = dataset_config["train_list"] if split == "train" else dataset_config["test_list"]
        self.list_path = self.root / list_name
        self.fps = float(dataset_config.get("fps", 60))
        self.seq_len = int(dataset_config.get("seq_len", 12))
        self.num_middle = int(dataset_config.get("num_middle", self.seq_len - 2))
        self.window_stride = int(dataset_config.get("window_stride", 5))
        self.condition_gaps = _normalize_int_mapping(
            dataset_config.get("condition_gaps"),
            {1: 240, 2: 120, 3: 24, 4: 12},
        )
        self.condition_ratios = _normalize_float_mapping(
            dataset_config.get("condition_ratios"),
            {1: 0.4, 2: 0.1, 3: 0.4, 4: 0.1},
        )
        self.use_npy_first = bool(dataset_config.get("use_npy_first", True))
        self.clamp = bool(dataset_config.get("clamp", True))
        self.clamp_min = float(dataset_config.get("clamp_min", 0.0))
        self.clamp_max = float(dataset_config.get("clamp_max", 1.0))
        keyframe_config = dataset_config.get("keyframe_detection", {})
        self.keyframe_eps = float(keyframe_config.get("eps", 0.05))
        self.keyframe_min_gap = int(keyframe_config.get("min_gap", 6))
        self.keyframe_lam = float(keyframe_config.get("lam", 0.5))
        self.keyframe_prominence_percentile = float(keyframe_config.get("prominence_percentile", 75))
        self.keyframe_smooth = bool(keyframe_config.get("smooth", True))

        missing_ratio_keys = set(self.condition_gaps) - set(self.condition_ratios)
        if missing_ratio_keys:
            raise ValueError(f"Missing ratios for conditions: {sorted(missing_ratio_keys)}")
        if self.seq_len != self.num_middle + 2:
            raise ValueError("Express4D expects seq_len == num_middle + 2")
        if self.window_stride <= 0:
            raise ValueError("window_stride must be positive")
        if not self.root.is_dir():
            raise FileNotFoundError(f"Express4D root not found: {self.root}")
        if not self.data_dir.is_dir():
            raise FileNotFoundError(f"Express4D data directory not found: {self.data_dir}")

        entries = _read_list(self.list_path)
        self.sequences = []
        self.sequence_keyframes = []
        self.samples = []
        for entry in entries:
            path = _resolve_entry(entry, self.root, self.data_dir, self.use_npy_first)
            data = load_blendshape_file(path, self.clamp, self.clamp_min, self.clamp_max)
            sequence_id = len(self.sequences)
            keyframes, _ = detect_blendshape_keyframes(
                data,
                eps=self.keyframe_eps,
                min_gap=self.keyframe_min_gap,
                lam=self.keyframe_lam,
                prominence_percentile=self.keyframe_prominence_percentile,
                smooth=self.keyframe_smooth,
            )
            self.sequences.append({"name": path.stem, "path": path, "data": data})
            self.sequence_keyframes.append(np.asarray(keyframes, dtype=np.int64))
            total_frames = data.shape[0]
            for condition, gap in sorted(self.condition_gaps.items()):
                if total_frames <= gap:
                    continue
                for start_idx in range(0, total_frames - gap, self.window_stride):
                    self.samples.append((sequence_id, start_idx, start_idx + gap, gap, condition))

        if not self.samples:
            raise ValueError(
                f"No Express4D condition samples could be built from {self.list_path}. "
                f"Check sequence lengths and condition_gaps={self.condition_gaps}."
            )
        self.condition_counts = Counter(sample[-1] for sample in self.samples)
        self.condition_indices = {}
        for index, sample in enumerate(self.samples):
            condition = sample[-1]
            self.condition_indices.setdefault(condition, []).append(index)

    def target_epoch_counts(self, base_condition=1):
        if base_condition not in self.condition_counts:
            raise ValueError(f"Base condition {base_condition} not found in dataset")
        base_count = int(self.condition_counts[base_condition])
        base_ratio = float(self.condition_ratios[base_condition])
        if base_ratio <= 0:
            raise ValueError(f"Base condition ratio must be positive, got {base_ratio}")

        total_target = int(round(base_count / base_ratio))
        targets = {}
        for condition, ratio in sorted(self.condition_ratios.items()):
            if condition == base_condition:
                targets[condition] = base_count
            else:
                targets[condition] = int(round(total_target * ratio))
        targets[base_condition] = base_count
        return targets

    def __getitem__(self, index):
        sequence_id, start_idx, end_idx, gap, condition = self.samples[index]
        sequence = self.sequences[sequence_id]
        keyframes = self.sequence_keyframes[sequence_id]
        seq, sample_positions, keyframe_slots = sample_nearest_sequence_with_keyframes(
            sequence["data"], start_idx, end_idx, keyframes, self.seq_len
        )

        observed_mask = np.zeros_like(seq, dtype=np.float32)
        observed_mask[0] = 1.0
        observed_mask[-1] = 1.0
        keyframe_mask = np.zeros_like(seq, dtype=np.float32)
        keyframe_mask[keyframe_slots.astype(bool)] = 1.0
        observed_mask[keyframe_slots.astype(bool)] = 1.0

        target_mask = 1.0 - observed_mask

        return {
            "observed_data": seq,
            "data": seq,
            "observed_mask": observed_mask,
            "gt_mask": target_mask,
            "target_mask": target_mask,
            "timepoints": np.arange(self.seq_len, dtype=np.float32),
            "condition": np.float32(condition),
            "keyframe_mask": keyframe_mask,
            "sample_positions": sample_positions.astype(np.float32),
            "start": seq[0],
            "end": seq[-1],
            "middle": seq[1:-1],
            "sequence_name": sequence["name"],
            "start_idx": np.int64(start_idx),
            "end_idx": np.int64(end_idx),
            "gap": np.int64(gap),
        }

    def __len__(self):
        return len(self.samples)


class BalancedDeterministicConditionSampler(Sampler):
    def __init__(
        self,
        dataset,
        base_condition=1,
        seed=1,
    ):
        self.dataset = dataset
        self.base_condition = int(base_condition)
        self.seed = int(seed)
        self.epoch = 0
        self.target_counts = dataset.target_epoch_counts(base_condition=self.base_condition)

        missing_conditions = sorted(set(dataset.condition_indices) - set(self.target_counts))
        if missing_conditions:
            raise ValueError(f"Missing target counts for conditions: {missing_conditions}")

    def __len__(self):
        return int(sum(self.target_counts.values()))

    def set_epoch(self, epoch):
        self.epoch = int(epoch)

    def _pick_without_replacement(self, indices, target_count, rng):
        if target_count <= 0:
            return np.empty((0,), dtype=np.int64)
        if len(indices) == 0:
            raise ValueError("Cannot sample from an empty condition bucket")

        permuted = np.array(indices, dtype=np.int64)
        rng.shuffle(permuted)
        if target_count <= len(permuted):
            return permuted[:target_count]

        repeats = int(math.ceil(target_count / len(permuted)))
        tiled = np.tile(permuted, repeats)
        return tiled[:target_count]

    def __iter__(self):
        rng = np.random.default_rng(self.seed + self.epoch)
        epoch_indices = []
        for condition, target_count in sorted(self.target_counts.items()):
            indices = self.dataset.condition_indices.get(condition, [])
            if condition == self.base_condition:
                selected = self._pick_without_replacement(indices, len(indices), rng)
            else:
                selected = self._pick_without_replacement(indices, target_count, rng)
            epoch_indices.extend(selected.tolist())

        epoch_indices = np.array(epoch_indices, dtype=np.int64)
        rng.shuffle(epoch_indices)
        self.epoch += 1
        return iter(epoch_indices.tolist())


def get_dataloader(config, seed=1, batch_size=16, num_workers=0):
    train_dataset = Express4DConditionDataset(config, split="train")
    test_dataset = Express4DConditionDataset(config, split="test")

    sampler = BalancedDeterministicConditionSampler(
        train_dataset,
        base_condition=int(config["dataset"].get("balance_base_condition", 1)),
        seed=seed,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
        drop_last=False,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        drop_last=False,
    )
    return train_loader, test_loader
