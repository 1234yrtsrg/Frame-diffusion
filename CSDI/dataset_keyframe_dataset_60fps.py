from collections import Counter
from pathlib import Path
import hashlib
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


def _normalize_source_ratios(mapping, default):
    if mapping is None:
        mapping = default
    values = {str(key).strip().lower(): float(value) for key, value in mapping.items()}
    total = sum(values.values())
    if total <= 0 or any(value < 0 for value in values.values()):
        raise ValueError("data_source_ratios must be non-negative and sum to a positive value")
    return {key: value / total for key, value in values.items()}


def _stable_hash_int(seed, *parts):
    payload = "\x1f".join([str(int(seed)), *(str(part) for part in parts)])
    return int.from_bytes(hashlib.sha256(payload.encode("utf-8")).digest()[:8], "big")


def _deterministic_quotas(total, ratios):
    """Largest-remainder allocation with stable key ordering for ties."""
    ordered = sorted(ratios)
    normalized_total = sum(float(ratios[key]) for key in ordered)
    if total < 0 or normalized_total <= 0:
        raise ValueError("quota total and ratios must be valid")
    exact = {key: total * float(ratios[key]) / normalized_total for key in ordered}
    quotas = {key: int(math.floor(exact[key])) for key in ordered}
    remaining = int(total - sum(quotas.values()))
    remainders = sorted(ordered, key=lambda key: (-(exact[key] - quotas[key]), key))
    for key in remainders[:remaining]:
        quotas[key] += 1
    assert sum(quotas.values()) == total
    return quotas


MASK_MODES = ("none", "partial", "all")


def _normalize_mask_ratios(mapping):
    defaults = {
        1: {"none": 0.40, "partial": 0.30, "all": 0.30},
        2: {"none": 0.30, "partial": 0.35, "all": 0.35},
        3: {"none": 0.00, "partial": 0.50, "all": 0.50},
        4: {"none": 0.00, "partial": 0.50, "all": 0.50},
    }
    mapping = defaults if mapping is None else mapping
    result = {}
    for raw_condition, raw_ratios in mapping.items():
        condition = int(raw_condition)
        unknown = set(raw_ratios) - set(MASK_MODES)
        if unknown:
            raise ValueError(f"Unknown mask modes for condition {condition}: {sorted(unknown)}")
        values = {mode: float(raw_ratios.get(mode, 0.0)) for mode in MASK_MODES}
        if any(value < 0 for value in values.values()) or sum(values.values()) <= 0:
            raise ValueError(f"Invalid mask ratios for condition {condition}: {values}")
        result[condition] = values
    return result


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


def _farthest_temporal_subset(candidates, anchors, count):
    candidates = sorted(set(int(value) for value in candidates))
    selected = set(int(value) for value in anchors)
    chosen = []
    while candidates and len(chosen) < count:
        candidate = min(
            candidates,
            key=lambda value: (-min(abs(value - anchor) for anchor in selected), value),
        )
        candidates.remove(candidate)
        selected.add(candidate)
        chosen.append(candidate)
    return chosen


def _uniform_positions_with_mandatory(start_idx, end_idx, seq_len, mandatory):
    """Minimum squared-error match to a uniform grid while forcing real positions."""
    mandatory = set(int(value) for value in mandatory)
    ideal = np.linspace(start_idx, end_idx, seq_len, dtype=np.float64)
    positions = np.arange(start_idx, end_idx + 1, dtype=np.int64)
    costs = np.full(seq_len + 1, np.inf, dtype=np.float64)
    costs[0] = 0.0
    took = np.zeros((len(positions), seq_len + 1), dtype=bool)
    for row, position in enumerate(positions):
        take_costs = np.full_like(costs, np.inf)
        take_costs[1:] = costs[:-1] + (float(position) - ideal) ** 2
        if int(position) in mandatory:
            next_costs = take_costs
            took[row] = np.isfinite(take_costs)
        else:
            # Prefer taking the earlier real frame on exact-cost ties.
            took[row] = take_costs <= costs
            next_costs = np.minimum(take_costs, costs)
        costs = next_costs
    if not np.isfinite(costs[seq_len]):
        raise ValueError("Could not place all mandatory frames on the uniform sample grid")
    selected = []
    count = seq_len
    for row in range(len(positions) - 1, -1, -1):
        if count > 0 and took[row, count]:
            selected.append(int(positions[row]))
            count -= 1
    if count != 0:
        raise AssertionError("Uniform frame-selection backtracking failed")
    return np.asarray(selected[::-1], dtype=np.int64)


def sample_nearest_sequence_with_keyframes(
    data,
    start_idx,
    end_idx,
    keyframes,
    seq_len=12,
    overflow_strategy="farthest_temporal_coverage",
):
    """Choose ``seq_len`` unique real frames while preserving selected keyframes."""
    data = np.asarray(data)
    start_idx = int(start_idx)
    end_idx = int(end_idx)
    seq_len = int(seq_len)
    if data.ndim < 1:
        raise ValueError("data must have a frame dimension")
    if not (0 <= start_idx < end_idx < len(data)):
        raise ValueError(f"Invalid window [{start_idx}, {end_idx}] for {len(data)} frames")
    if seq_len < 2 or end_idx - start_idx + 1 < seq_len:
        raise ValueError(
            f"Window [{start_idx}, {end_idx}] cannot provide {seq_len} unique real frames"
        )
    if overflow_strategy != "farthest_temporal_coverage":
        raise ValueError(f"Unsupported keyframe overflow strategy: {overflow_strategy}")

    internal_keyframes = sorted(
        set(int(value) for value in keyframes if start_idx < int(value) < end_idx)
    )
    capacity = seq_len - 2
    if len(internal_keyframes) <= capacity:
        retained_keyframes = internal_keyframes
    else:
        retained_keyframes = sorted(
            _farthest_temporal_subset(
                internal_keyframes,
                anchors=(start_idx, end_idx),
                count=capacity,
            )
        )

    retained_set = set(retained_keyframes)
    mandatory = {start_idx, end_idx, *retained_set}
    sample_positions = _uniform_positions_with_mandatory(
        start_idx, end_idx, seq_len, mandatory
    )
    keyframe_slots = np.asarray(
        [1.0 if int(position) in retained_set else 0.0 for position in sample_positions],
        dtype=np.float32,
    )
    sampled = data[sample_positions].astype(np.float32, copy=False)

    assert len(sample_positions) == seq_len
    assert sample_positions[0] == start_idx and sample_positions[-1] == end_idx
    assert np.all((sample_positions >= start_idx) & (sample_positions <= end_idx))
    assert np.all(np.diff(sample_positions) > 0)
    assert np.array_equal(sampled, data[sample_positions].astype(np.float32, copy=False))
    assert retained_set.issubset(set(sample_positions.tolist()))
    return sampled, sample_positions, keyframe_slots


def sample_timepoints(sample_positions, start_idx, end_idx):
    sample_positions = np.asarray(sample_positions, dtype=np.float32)
    start_idx = int(start_idx)
    end_idx = int(end_idx)
    if end_idx <= start_idx:
        raise ValueError("end_idx must be greater than start_idx")
    if sample_positions.ndim != 1 or len(sample_positions) < 2:
        raise ValueError("sample_positions must be a one-dimensional sequence")
    if sample_positions[0] != start_idx or sample_positions[-1] != end_idx:
        raise ValueError("sample_positions must start at start_idx and end at end_idx")
    if not np.all(np.diff(sample_positions) > 0):
        raise ValueError("sample_positions must be strictly increasing")
    timepoints = np.float32(11.0) * (sample_positions - np.float32(start_idx)) / np.float32(
        end_idx - start_idx
    )
    timepoints = timepoints.astype(np.float32, copy=False)
    timepoints[0] = np.float32(0.0)
    timepoints[-1] = np.float32(11.0)
    assert np.all(np.diff(timepoints) > 0)
    return timepoints


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
        self.data_source_ratios = _normalize_source_ratios(
            dataset_config.get("data_source_ratios"),
            {"dfew": 0.8739, "express4d": 0.1261},
        )
        self.mask_seed = int(dataset_config.get("mask_seed", 1))
        self.mask_ratios = _normalize_mask_ratios(dataset_config.get("mask_ratios"))
        self.partial_visible_ratio = float(dataset_config.get("partial_visible_ratio", 0.5))
        self.keyframe_overflow_strategy = str(
            dataset_config.get("keyframe_overflow_strategy", "farthest_temporal_coverage")
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
        if not 0.0 < self.partial_visible_ratio < 1.0:
            raise ValueError("partial_visible_ratio must be strictly between 0 and 1")
        if set(self.condition_gaps) - set(self.mask_ratios):
            raise ValueError("mask_ratios must define every configured condition")
        if set(self.data_dirs) != set(self.data_source_ratios):
            raise ValueError(
                "data_source_ratios keys must exactly match data_dirs after lowercase normalization"
            )
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
        self.condition_source_indices = {}
        for index, sample in enumerate(self.samples):
            condition = sample[-1]
            self.condition_indices.setdefault(condition, []).append(index)
            sequence = self.sequences[sample[0]]
            key = (condition, sequence["data_source"])
            self.condition_source_indices.setdefault(key, []).append(index)
        self.dataset_counts = Counter(sequence["dataset_name"] for sequence in self.sequences)
        self.data_source_counts = Counter(sequence["data_source"] for sequence in self.sequences)
        self.mask_modes = self._assign_mask_modes()

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
                sequence["data_source"] = str(data_dir).strip().lower()
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

    def stable_sample_id(self, index):
        sequence_id, start_idx, end_idx, _, condition = self.samples[index]
        sequence = self.sequences[sequence_id]
        return "|".join(
            (
                sequence["data_source"],
                sequence["sequence_name"],
                str(int(start_idx)),
                str(int(end_idx)),
                str(int(condition)),
            )
        )

    def sample_sampling_metadata(self, index):
        sequence_id, _, _, _, condition = self.samples[index]
        return int(condition), self.sequences[sequence_id]["data_source"], self.mask_modes[index]

    def _internal_keyframe_count(self, index):
        sequence_id, start_idx, end_idx, _, _ = self.samples[index]
        keyframes = self.sequences[sequence_id]["keyframes"]
        return min(
            self.seq_len - 2,
            len(set(int(value) for value in keyframes if start_idx < int(value) < end_idx)),
        )

    def _assign_mask_modes(self):
        modes = [None] * len(self.samples)
        grouped = {}
        for index, sample in enumerate(self.samples):
            condition = int(sample[-1])
            source = self.sequences[sample[0]]["data_source"]
            keyframe_count = self._internal_keyframe_count(index)
            if keyframe_count == 0:
                modes[index] = "none"
                continue
            legal = ("none", "all") if keyframe_count == 1 else MASK_MODES
            grouped.setdefault((condition, source, legal), []).append(index)

        for (condition, source, legal), indices in sorted(grouped.items()):
            ratios = {mode: self.mask_ratios[condition][mode] for mode in legal}
            if sum(ratios.values()) <= 0:
                raise ValueError(
                    f"No positive legal mask ratio for condition={condition}, source={source}, modes={legal}"
                )
            quotas = _deterministic_quotas(len(indices), ratios)
            ordered = sorted(
                indices,
                key=lambda index: (
                    _stable_hash_int(self.mask_seed, "mask-mode", self.stable_sample_id(index)),
                    self.stable_sample_id(index),
                ),
            )
            offset = 0
            for mode in legal:
                next_offset = offset + quotas[mode]
                for index in ordered[offset:next_offset]:
                    modes[index] = mode
                offset = next_offset

        assert all(mode in MASK_MODES for mode in modes)
        return modes

    def _visible_keyframe_slots(self, index, keyframe_slots, sample_positions):
        mode = self.mask_modes[index]
        slots = np.flatnonzero(np.asarray(keyframe_slots) > 0)
        if mode == "none":
            return np.empty((0,), dtype=np.int64)
        if mode == "all":
            return slots.astype(np.int64, copy=False)
        if len(slots) < 2:
            raise AssertionError("partial mode requires at least two internal keyframes")
        visible_count = int(math.floor(len(slots) * self.partial_visible_ratio + 0.5))
        visible_count = min(len(slots) - 1, max(1, visible_count))
        sample_id = self.stable_sample_id(index)
        ranked = sorted(
            slots.tolist(),
            key=lambda slot: (
                _stable_hash_int(
                    self.mask_seed,
                    "partial-visible",
                    sample_id,
                    int(sample_positions[slot]),
                ),
                int(sample_positions[slot]),
            ),
        )
        return np.asarray(sorted(ranked[:visible_count]), dtype=np.int64)

    def __getitem__(self, index):
        sequence_id, start_idx, end_idx, gap, condition = self.samples[index]
        sequence = self.sequences[sequence_id]
        seq, sample_positions, keyframe_slots = sample_nearest_sequence_with_keyframes(
            sequence["data"],
            start_idx,
            end_idx,
            sequence["keyframes"],
            self.seq_len,
            overflow_strategy=self.keyframe_overflow_strategy,
        )

        observed_mask = np.zeros_like(seq, dtype=np.float32)
        observed_mask[0] = 1.0
        observed_mask[-1] = 1.0

        keyframe_mask = np.zeros_like(seq, dtype=np.float32)
        keyframe_mask[keyframe_slots.astype(bool)] = 1.0
        visible_keyframe_mask = np.zeros_like(seq, dtype=np.float32)
        visible_slots = self._visible_keyframe_slots(index, keyframe_slots, sample_positions)
        visible_keyframe_mask[visible_slots] = 1.0
        observed_mask[visible_slots] = 1.0

        target_mask = 1.0 - observed_mask
        timepoints = sample_timepoints(sample_positions, start_idx, end_idx)

        return {
            "observed_data": seq,
            "data": seq,
            "observed_mask": observed_mask,
            "gt_mask": target_mask,
            "target_mask": target_mask,
            "timepoints": timepoints,
            "condition": np.float32(condition),
            "duration": np.float32(gap / self.fps),
            "keyframe_mask": keyframe_mask,
            "visible_keyframe_mask": visible_keyframe_mask,
            "sample_positions": sample_positions,
            "mask_mode": self.mask_modes[index],
            "data_source": sequence["data_source"],
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
    def __init__(self, dataset, base_condition=1, seed=1, data_source_ratios=None):
        self.dataset = dataset
        self.base_condition = int(base_condition)
        self.seed = int(seed)
        self.epoch = 0
        self.target_counts = dataset.target_epoch_counts(base_condition=self.base_condition)
        self.data_source_ratios = _normalize_source_ratios(
            data_source_ratios,
            getattr(dataset, "data_source_ratios", {"dfew": 0.8739, "express4d": 0.1261}),
        )
        self.target_source_counts = {
            condition: _deterministic_quotas(target_count, self.data_source_ratios)
            for condition, target_count in sorted(self.target_counts.items())
        }
        self.last_epoch_stats = None

        missing_conditions = sorted(set(dataset.condition_indices) - set(self.target_counts))
        if missing_conditions:
            raise ValueError(f"Missing target counts for conditions: {missing_conditions}")
        for condition, source_targets in self.target_source_counts.items():
            for source, target_count in source_targets.items():
                if target_count > 0 and not dataset.condition_source_indices.get((condition, source)):
                    raise ValueError(
                        f"No samples for required condition={condition}, data_source={source}"
                    )

    def __len__(self):
        return int(sum(self.target_counts.values()))

    def set_epoch(self, epoch):
        self.epoch = int(epoch)

    def _pick_without_replacement(self, indices, target_count, rng):
        if target_count <= 0:
            return np.empty((0,), dtype=np.int64), 0
        if len(indices) == 0:
            raise ValueError("Cannot sample from an empty condition/data-source bucket")

        permuted = np.array(indices, dtype=np.int64)
        rng.shuffle(permuted)
        if target_count <= len(permuted):
            return permuted[:target_count], 0

        repeats = int(math.ceil(target_count / len(permuted)))
        tiled = np.tile(permuted, repeats)
        return tiled[:target_count], int(target_count - len(permuted))

    def _build_epoch(self, epoch):
        rng = np.random.default_rng(self.seed + int(epoch))
        epoch_indices = []
        repeat_counts = {}
        for condition, source_targets in sorted(self.target_source_counts.items()):
            repeat_counts[condition] = {}
            for source, target_count in sorted(source_targets.items()):
                indices = self.dataset.condition_source_indices.get((condition, source), [])
                selected, repeated = self._pick_without_replacement(indices, target_count, rng)
                epoch_indices.extend(selected.tolist())
                repeat_counts[condition][source] = repeated

        epoch_indices = np.asarray(epoch_indices, dtype=np.int64)
        rng.shuffle(epoch_indices)
        stats = self._summarize(epoch_indices, repeat_counts, epoch)
        return epoch_indices.tolist(), stats

    def _summarize(self, epoch_indices, repeat_counts, epoch):
        condition_counts = Counter()
        source_counts = {condition: Counter() for condition in self.target_counts}
        mask_counts = Counter()
        joint_counts = {}
        for raw_index in epoch_indices:
            index = int(raw_index)
            condition, source, mode = self.dataset.sample_sampling_metadata(index)
            condition_counts[condition] += 1
            source_counts[condition][source] += 1
            mask_counts[mode] += 1
            joint_counts.setdefault(condition, {}).setdefault(source, Counter())[mode] += 1

        actual_source_ratios = {}
        for condition, counts in source_counts.items():
            total = sum(counts.values())
            actual_source_ratios[condition] = {
                source: (float(counts[source]) / total if total else 0.0)
                for source in sorted(self.data_source_ratios)
            }
        return {
            "epoch": int(epoch),
            "total_samples": int(len(epoch_indices)),
            "target_condition_counts": {
                int(key): int(value) for key, value in sorted(self.target_counts.items())
            },
            "target_condition_source_counts": {
                int(condition): {source: int(count) for source, count in sorted(values.items())}
                for condition, values in sorted(self.target_source_counts.items())
            },
            "actual_condition_counts": {
                int(key): int(value) for key, value in sorted(condition_counts.items())
            },
            "actual_condition_source_counts": {
                int(condition): {
                    source: int(counts[source]) for source in sorted(self.data_source_ratios)
                }
                for condition, counts in sorted(source_counts.items())
            },
            "actual_condition_source_ratios": actual_source_ratios,
            "mask_mode_counts": {mode: int(mask_counts[mode]) for mode in MASK_MODES},
            "condition_source_mask_mode_counts": {
                int(condition): {
                    source: {
                        mode: int(joint_counts.get(condition, {}).get(source, {}).get(mode, 0))
                        for mode in MASK_MODES
                    }
                    for source in sorted(self.data_source_ratios)
                }
                for condition in sorted(self.target_counts)
            },
            "repeat_counts": {
                int(condition): {source: int(count) for source, count in sorted(values.items())}
                for condition, values in sorted(repeat_counts.items())
            },
            "total_repeats": int(
                sum(sum(values.values()) for values in repeat_counts.values())
            ),
        }

    def epoch_statistics(self, epoch=0):
        _, stats = self._build_epoch(epoch)
        return stats

    def __iter__(self):
        epoch_indices, self.last_epoch_stats = self._build_epoch(self.epoch)
        self.epoch += 1
        return iter(epoch_indices)


def get_dataloader(config, seed=1, batch_size=16, num_workers=0):
    train_dataset = KeyframeDataset60fps(config, split="train")
    test_dataset = KeyframeDataset60fps(config, split="test")

    sampler = BalancedDeterministicConditionSampler(
        train_dataset,
        base_condition=int(config["dataset"].get("balance_base_condition", 1)),
        seed=seed,
        data_source_ratios=config["dataset"].get("data_source_ratios"),
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
