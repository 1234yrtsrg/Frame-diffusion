from collections import Counter
from pathlib import Path
import json
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


def _as_list(value):
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return list(value)
    if isinstance(value, str):
        return [item.strip() for item in value.split(",") if item.strip()]
    return [value]


def _as_ratio_pair(split_ratios):
    ratios = np.asarray(split_ratios, dtype=np.float64)
    if ratios.shape != (2,):
        raise ValueError(f"split_ratios must contain train/test ratios, got {split_ratios}")
    if np.any(ratios < 0) or ratios.sum() <= 0:
        raise ValueError(f"split_ratios must be non-negative and non-zero, got {split_ratios}")
    return ratios / ratios.sum()


def _load_keyframe_indices(payload, path, total_frames):
    raw_keyframes = payload.get("keyframe_indices", payload.get("keyframes"))
    if raw_keyframes is None:
        raise ValueError(f"Missing keyframe_indices in annotated keyframe JSON: {path}")

    keyframes = []
    for item in raw_keyframes:
        if isinstance(item, dict):
            item = item.get("frame_index", item.get("index"))
        if item is None:
            continue
        keyframe = int(item)
        if 0 <= keyframe < total_frames:
            keyframes.append(keyframe)
    return np.asarray(sorted(set(keyframes)), dtype=np.int64)


def _load_annotated_json(path, clamp=True, clamp_min=0.0, clamp_max=1.0):
    path = Path(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    frames = payload.get("frames")
    if not frames:
        raise ValueError(f"Missing frames in annotated keyframe JSON: {path}")

    data = []
    for frame in frames:
        blendshapes = frame.get("blendshapes")
        if blendshapes is None:
            raise ValueError(f"Missing blendshapes in frame for {path}")
        data.append(blendshapes)

    data = np.asarray(data, dtype=np.float32)
    if data.ndim != 2 or data.shape[1] != 52:
        raise ValueError(f"{path} must contain frames[].blendshapes with shape [T,52], got {data.shape}")
    data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32, copy=False)
    if clamp:
        data = np.clip(data, clamp_min, clamp_max)

    return data, _load_keyframe_indices(payload, path, len(data))


def _resolve_keyframe_json(entry, data_path, root, data_dir, keyframe_dir=None):
    data_path = Path(data_path)
    if data_path.suffix.lower() == ".json":
        return data_path

    raw = entry.strip().replace("\\", "/")
    rel = Path(raw)
    candidates = []

    if keyframe_dir:
        keyframe_base = Path(keyframe_dir)
        if not keyframe_base.is_absolute():
            keyframe_base = root / keyframe_base
        if rel.suffix:
            candidates.append(keyframe_base / rel.with_suffix(".json"))
        else:
            candidates.append(keyframe_base / rel.with_suffix(".json"))
        candidates.append(keyframe_base / f"{data_path.stem}.json")

    candidates.append(data_path.with_suffix(".json"))
    if rel.suffix:
        candidates.append((root / rel).with_suffix(".json"))
    else:
        candidates.append((data_dir / rel).with_suffix(".json"))
        candidates.append((root / rel).with_suffix(".json"))

    seen = set()
    unique_candidates = []
    for candidate in candidates:
        candidate = Path(candidate)
        key = str(candidate)
        if key not in seen:
            seen.add(key)
            unique_candidates.append(candidate)

    for candidate in unique_candidates:
        if candidate.is_file():
            return candidate

    searched = ", ".join(str(path) for path in unique_candidates)
    raise FileNotFoundError(f"Could not find annotated keyframe JSON for '{entry}'. Tried: {searched}")


def _resolve_annotated_json_entry(entry, root, data_dir, keyframe_dir=None):
    raw = entry.strip().replace("\\", "/")
    rel = Path(raw)
    candidates = []

    if keyframe_dir:
        keyframe_base = Path(keyframe_dir)
        if not keyframe_base.is_absolute():
            keyframe_base = root / keyframe_base
        candidates.append(keyframe_base / rel.with_suffix(".json"))
        candidates.append(keyframe_base / f"{rel.stem}.json")

    candidates.append(data_dir / rel.with_suffix(".json"))
    candidates.append(root / rel.with_suffix(".json"))

    seen = set()
    for candidate in candidates:
        key = str(candidate)
        if key in seen:
            continue
        seen.add(key)
        if candidate.is_file():
            return candidate

    searched = ", ".join(str(path) for path in candidates)
    raise FileNotFoundError(f"Could not find annotated keyframe JSON for '{entry}'. Tried: {searched}")


def _is_summary_json(path):
    return Path(path).name.startswith("keyframe_summary")


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
        list_name = dataset_config.get("train_list", "train.txt") if split == "train" else dataset_config.get("test_list", "test.txt")
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
        self.keyframe_dir = dataset_config.get("keyframe_dir", None)
        self.data_dirs = _as_list(dataset_config.get("data_dirs", None))
        self.split_ratios = _as_ratio_pair(dataset_config.get("split_ratios", [0.8, 0.2]))
        self.split_seed = int(dataset_config.get("split_seed", config.get("seed", 1)))

        missing_ratio_keys = set(self.condition_gaps) - set(self.condition_ratios)
        if missing_ratio_keys:
            raise ValueError(f"Missing ratios for conditions: {sorted(missing_ratio_keys)}")
        if self.seq_len != self.num_middle + 2:
            raise ValueError("Express4D expects seq_len == num_middle + 2")
        if self.window_stride <= 0:
            raise ValueError("window_stride must be positive")
        if not self.root.is_dir():
            raise FileNotFoundError(f"Express4D root not found: {self.root}")

        entries = self._resolve_entries(split)
        self.sequences = []
        self.sequence_keyframes = []
        self.samples = []
        for entry in entries:
            path = self._resolve_sequence_path(entry)
            if path.suffix.lower() == ".json":
                data, keyframes = _load_annotated_json(path, self.clamp, self.clamp_min, self.clamp_max)
            else:
                data = load_blendshape_file(path, self.clamp, self.clamp_min, self.clamp_max)
                keyframe_path = _resolve_keyframe_json(
                    entry,
                    path,
                    self.root,
                    self.data_dir,
                    keyframe_dir=self.keyframe_dir,
                )
                keyframe_payload = json.loads(keyframe_path.read_text(encoding="utf-8"))
                keyframes = _load_keyframe_indices(keyframe_payload, keyframe_path, len(data))
            sequence_id = len(self.sequences)
            self.sequences.append({"name": path.stem, "path": path, "data": data})
            self.sequence_keyframes.append(keyframes)
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

    def _resolve_sequence_path(self, entry):
        if isinstance(entry, Path):
            return entry
        try:
            return _resolve_annotated_json_entry(
                entry,
                self.root,
                self.data_dir,
                keyframe_dir=self.keyframe_dir,
            )
        except FileNotFoundError:
            return _resolve_entry(entry, self.root, self.data_dir, self.use_npy_first)

    def _resolve_entries(self, split):
        if self.list_path.is_file():
            return _read_list(self.list_path)

        json_paths = self._scan_annotated_jsons()
        if not json_paths:
            raise FileNotFoundError(
                f"No split file found at {self.list_path}, and no annotated JSON files found under {self.root}"
            )

        indices = np.arange(len(json_paths), dtype=np.int64)
        rng = np.random.default_rng(self.split_seed)
        rng.shuffle(indices)
        train_count = int(len(indices) * self.split_ratios[0])
        if len(indices) < 2:
            raise ValueError("Need at least two annotated JSON sequences to build train/test splits")
        if train_count <= 0 or train_count >= len(indices):
            raise ValueError(
                f"split_ratios={self.split_ratios.tolist()} produce an empty train or test split"
            )

        selected = indices[:train_count] if split == "train" else indices[train_count:]
        return [json_paths[int(index)] for index in selected]

    def _scan_annotated_jsons(self):
        candidate_dirs = []
        for data_dir in self.data_dirs:
            candidate = self.root / str(data_dir)
            if candidate.is_dir():
                candidate_dirs.append(candidate)

        if not candidate_dirs:
            if self.data_dir.is_dir():
                candidate_dirs.append(self.data_dir)
            elif (self.root / "express4d").is_dir():
                express4d_dir = self.root / "express4d"
                candidate_dirs.append(express4d_dir)
            else:
                candidate_dirs.append(self.root)

        seen = set()
        json_paths = []
        for candidate_dir in candidate_dirs:
            for path in sorted(candidate_dir.rglob("*.json")):
                if _is_summary_json(path):
                    continue
                key = str(path.resolve())
                if key in seen:
                    continue
                seen.add(key)
                json_paths.append(path)
        return json_paths

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
