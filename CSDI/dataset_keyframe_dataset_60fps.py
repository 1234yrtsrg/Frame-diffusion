from collections import Counter
from pathlib import Path
import json
import math

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Sampler


def _repo_root():
    return Path(__file__).resolve().parents[1]


def resolve_dataset_root(root):
    root_path = Path(root)
    if root_path.is_absolute():
        return root_path
    return _repo_root() / root_path


def _normalize_split_name(split):
    if split == "val":
        return "test"
    if split not in ("train", "test"):
        raise ValueError(f"split must be 'train' or 'test', got {split!r}")
    return split


def _as_ratio_pair(split_ratios):
    ratios = np.asarray(split_ratios, dtype=np.float64)
    if ratios.shape != (2,):
        raise ValueError(f"split_ratios must contain train/test ratios, got {split_ratios}")
    if np.any(ratios < 0) or ratios.sum() <= 0:
        raise ValueError(f"split_ratios must be non-negative and non-zero, got {split_ratios}")
    return ratios / ratios.sum()


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


def _read_split_list(path):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Split file not found: {path}")
    entries = []
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        item = line.strip()
        if not item or item.startswith("#"):
            continue
        entries.append(item)
    if not entries:
        raise ValueError(f"Split file is empty: {path}")
    return entries


def _entry_to_json_path(entry, dataset_dir):
    raw = str(entry).strip().replace("\\", "/")
    rel = Path(raw)
    candidates = []
    if rel.is_absolute():
        candidates.append(rel)
        if rel.suffix.lower() != ".json":
            candidates.append(rel.with_suffix(".json"))
    else:
        candidates.append(dataset_dir / rel)
        if rel.suffix.lower() != ".json":
            candidates.append(dataset_dir / rel.with_suffix(".json"))

    for candidate in candidates:
        if candidate.is_file():
            return candidate
    searched = ", ".join(str(path) for path in candidates)
    raise FileNotFoundError(f"Could not find JSON for split entry {entry!r}. Tried: {searched}")


def _json_split_entry(path, dataset_dir):
    rel = Path(path).relative_to(dataset_dir)
    return rel.as_posix()


def _load_keyframe_indices(payload, path, total_frames):
    raw_keyframes = payload.get("keyframe_indices", payload.get("keyframes"))
    if raw_keyframes is None:
        raise ValueError(f"Missing keyframe_indices in keyframe JSON: {path}")

    keyframes = []
    for item in raw_keyframes:
        if isinstance(item, dict):
            item = item.get("frame_index", item.get("index"))
        if item is None:
            continue
        keyframe = int(item)
        if 0 <= keyframe < total_frames:
            keyframes.append(keyframe)
    return sorted(set(keyframes))


def _load_keyframe_json(path, clamp=True, clamp_min=0.0, clamp_max=1.0):
    path = Path(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    frames = payload.get("frames")
    if not frames:
        raise ValueError(f"Missing frames in keyframe JSON: {path}")

    data = []
    for frame in frames:
        blendshapes = frame.get("blendshapes")
        if blendshapes is None:
            raise ValueError(f"Missing blendshapes in frame for {path}")
        data.append(blendshapes)

    data = np.asarray(data, dtype=np.float32)
    if data.ndim != 2:
        raise ValueError(f"{path} must load as 2D array, got shape {data.shape}")

    if data.shape[1] != 52:
        raise ValueError(f"{path} must contain 52 blendshape values per frame, got {data.shape}")

    data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32, copy=False)
    if clamp:
        data = np.clip(data, clamp_min, clamp_max)

    keyframes = _load_keyframe_indices(payload, path, len(data))

    dataset_name = str(payload.get("dataset", path.parent.name))
    video_id = str(payload.get("video_id", path.stem))
    return {
        "dataset_name": dataset_name,
        "video_id": video_id,
        "sequence_name": f"{dataset_name}/{video_id}",
        "path": path,
        "data": data,
        "keyframes": np.asarray(keyframes, dtype=np.int64),
        "num_frames": int(len(data)),
    }


def sample_nearest_sequence_with_keyframes(data, start_idx, end_idx, keyframes, seq_len=12):
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


class KeyframeDataset60fps(Dataset):
    """Train directly from annotated 60fps keyframe JSONs with sliding windows."""

    def __init__(self, config, split="train"):
        dataset_config = config["dataset"]
        self.root = resolve_dataset_root(dataset_config["root"])
        self.split = _normalize_split_name(split)
        self.seq_len = int(dataset_config.get("seq_len", 12))
        self.num_middle = int(dataset_config.get("num_middle", self.seq_len - 2))
        self.fps = float(dataset_config.get("fps", 60))
        self.window_stride = int(dataset_config.get("window_stride", 5))
        self.condition_gaps = _normalize_int_mapping(
            dataset_config.get("condition_gaps"),
            {1: 240, 2: 120, 3: 24, 4: 12},
        )
        self.condition_ratios = _normalize_float_mapping(
            dataset_config.get("condition_ratios"),
            {1: 0.4, 2: 0.1, 3: 0.4, 4: 0.1},
        )
        self.split_ratios = _as_ratio_pair(dataset_config.get("split_ratios", [0.8, 0.2]))
        self.split_seed = int(dataset_config.get("split_seed", config.get("seed", 1)))
        self.split_files = dict(dataset_config.get("split_files", {}))
        self.write_split_files = bool(dataset_config.get("write_split_files", True))
        self.random_split_data_dirs = set(_as_list(dataset_config.get("random_split_data_dirs", ["dfew"])))
        self.clamp = bool(dataset_config.get("clamp", True))
        self.clamp_min = float(dataset_config.get("clamp_min", 0.0))
        self.clamp_max = float(dataset_config.get("clamp_max", 1.0))

        data_dirs = dataset_config.get("data_dirs")
        if data_dirs is None:
            single_data_dir = dataset_config.get("data_dir")
            data_dirs = [single_data_dir] if single_data_dir is not None else ["dfew", "express4d"]
        self.data_dirs = _as_list(data_dirs)
        if not self.data_dirs:
            raise ValueError("At least one data directory must be provided")

        if self.seq_len != self.num_middle + 2:
            raise ValueError("KeyframeDataset60fps expects seq_len == num_middle + 2")
        if self.window_stride <= 0:
            raise ValueError("window_stride must be positive")
        if not self.root.is_dir():
            raise FileNotFoundError(f"Keyframe dataset root not found: {self.root}")

        self.sequences = self._load_split_sequences()
        self.samples = self._build_samples()

        if not self.samples:
            raise ValueError(
                f"No windows could be built from {self.root} with gaps={self.condition_gaps} "
                f"and seq_len={self.seq_len}"
            )

        self.condition_counts = Counter(sample[-1] for sample in self.samples)
        self.condition_indices = {}
        for index, sample in enumerate(self.samples):
            condition = sample[-1]
            self.condition_indices.setdefault(condition, []).append(index)
        self.dataset_counts = Counter(sequence["dataset_name"] for sequence in self.sequences)

    def _load_split_sequences(self):
        sequences = []
        for data_dir in self.data_dirs:
            dir_path = self.root / data_dir
            if not dir_path.is_dir():
                raise FileNotFoundError(f"Keyframe dataset directory not found: {dir_path}")

            for path in self._split_json_paths(data_dir, dir_path):
                sequence = _load_keyframe_json(
                    path,
                    clamp=self.clamp,
                    clamp_min=self.clamp_min,
                    clamp_max=self.clamp_max,
                )
                sequences.append(sequence)

        if not sequences:
            raise ValueError(f"No keyframe JSON sequences found under {self.root / self.data_dirs[0]}")
        return sequences

    def _all_json_paths(self, dataset_dir):
        return [
            path
            for path in sorted(dataset_dir.rglob("*.json"))
            if not path.name.startswith("keyframe_summary")
        ]

    def _split_file_paths(self, data_dir):
        configured = self.split_files.get(data_dir, {})
        train_name = configured.get("train", f"{data_dir}_train.txt")
        test_name = configured.get("test", f"{data_dir}_test.txt")
        return {
            "train": self.root / train_name,
            "test": self.root / test_name,
        }

    def _split_json_paths(self, data_dir, dataset_dir):
        split_paths = self._split_file_paths(data_dir)
        split_file = split_paths[self.split]
        if split_file.is_file():
            return [_entry_to_json_path(entry, dataset_dir) for entry in _read_split_list(split_file)]

        all_paths = self._all_json_paths(dataset_dir)
        if not all_paths:
            raise ValueError(f"No keyframe JSON sequences found under {dataset_dir}")

        if data_dir not in self.random_split_data_dirs:
            raise FileNotFoundError(
                f"Missing split file for {data_dir}: {split_file}. "
                f"Expected explicit {data_dir}_train.txt/{data_dir}_test.txt."
            )

        split_map = self._build_random_split_paths(data_dir, dataset_dir, all_paths, split_paths)
        return split_map[self.split]

    def _build_random_split_paths(self, data_dir, dataset_dir, all_paths, split_paths):
        indices = np.arange(len(all_paths), dtype=np.int64)
        rng = np.random.default_rng(self.split_seed)
        rng.shuffle(indices)

        train_count = int(len(indices) * self.split_ratios[0])
        if len(indices) < 2:
            raise ValueError(f"Need at least two {data_dir} sequences to build train/test splits")
        if train_count <= 0 or train_count >= len(indices):
            raise ValueError(
                f"split_ratios={self.split_ratios.tolist()} produce an empty train or test split"
            )

        split_map = {
            "train": [all_paths[int(index)] for index in indices[:train_count]],
            "test": [all_paths[int(index)] for index in indices[train_count:]],
        }

        if self.write_split_files:
            for split_name, paths in split_map.items():
                split_file = split_paths[split_name]
                if not split_file.exists():
                    lines = [_json_split_entry(path, dataset_dir) for path in paths]
                    split_file.write_text("\n".join(lines) + "\n", encoding="utf-8")

        return split_map

    def _build_samples(self):
        samples = []
        for sequence_id, sequence in enumerate(self.sequences):
            total_frames = int(sequence["data"].shape[0])
            for condition, gap in sorted(self.condition_gaps.items()):
                if total_frames <= gap:
                    continue
                for start_idx in range(0, total_frames - gap, self.window_stride):
                    samples.append((sequence_id, start_idx, start_idx + gap, gap, condition))
        return samples

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
        seq, sample_positions, keyframe_slots = sample_nearest_sequence_with_keyframes(
            sequence["data"], start_idx, end_idx, sequence["keyframes"], self.seq_len
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
            "duration": np.float32(gap / self.fps),
            "keyframe_mask": keyframe_mask,
            "sample_positions": sample_positions.astype(np.float32),
            "start": seq[0],
            "end": seq[-1],
            "middle": seq[1:-1],
            "sequence_name": sequence["sequence_name"],
            "dataset_name": sequence["dataset_name"],
            "source_path": str(sequence["path"]),
            "start_idx": np.int64(start_idx),
            "end_idx": np.int64(end_idx),
            "gap": np.int64(gap),
        }

    def __len__(self):
        return len(self.samples)


class BalancedDeterministicConditionSampler(Sampler):
    def __init__(self, dataset, base_condition=1, seed=1):
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
    train_dataset = KeyframeDataset60fps(config, split="train")
    test_dataset = KeyframeDataset60fps(config, split="test")

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
