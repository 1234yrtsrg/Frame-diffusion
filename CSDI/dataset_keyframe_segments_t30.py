from pathlib import Path
import warnings

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


SPLIT_NAMES = ("train", "valid", "test")


def _repo_root():
    return Path(__file__).resolve().parents[1]


def resolve_dataset_path(path):
    path = Path(path)
    if path.is_absolute():
        return path
    return _repo_root() / path


def _normalise_split_name(split):
    if split == "val":
        return "valid"
    if split not in SPLIT_NAMES:
        raise ValueError(f"split must be one of {SPLIT_NAMES}, got {split!r}")
    return split


def _as_ratios(split_ratios):
    ratios = np.asarray(split_ratios, dtype=np.float64)
    if ratios.shape != (3,):
        raise ValueError(f"split_ratios must contain train/valid/test ratios, got {split_ratios}")
    if np.any(ratios < 0) or ratios.sum() <= 0:
        raise ValueError(f"split_ratios must be non-negative and non-zero, got {split_ratios}")
    return ratios / ratios.sum()


class KeyframeSegmentsT30Store:
    """Shared in-memory NPZ backing store.

    The NPZ is compressed, so reading windows decompresses the full array into
    memory. Build one store and share it between split datasets in one process.
    """

    def __init__(self, config):
        dataset_config = config["dataset"]
        root = dataset_config.get("root", "dataset")
        npz_file = dataset_config.get("npz_file", "keyframe_segments_T30.npz")
        self.path = resolve_dataset_path(Path(root) / npz_file)
        if not self.path.is_file():
            raise FileNotFoundError(f"Keyframe segment NPZ not found: {self.path}")

        with np.load(self.path, allow_pickle=False) as data:
            self.windows = data["windows"].astype(np.float32, copy=False)
            self.keyframe_mask = data["keyframe_mask"] if "keyframe_mask" in data.files else None
            self.source_ids = data["source_ids"].astype(np.int64, copy=False)
            self.dataset_ids = data["dataset_ids"].astype(np.int64, copy=False)
            self.original_gap_frames = data["original_gap_frames"].astype(np.int64, copy=False)
            self.durations_sec = data["durations_sec"].astype(np.float32, copy=False)
            self.sources = data["sources"]
            self.source_paths = data["source_paths"]
            self.datasets = data["datasets"]

        self.seq_len = int(dataset_config.get("seq_len", 30))
        self.num_features = int(dataset_config.get("num_features", 52))
        self.num_middle = int(dataset_config.get("num_middle", self.seq_len - 2))
        if self.windows.ndim != 3 or self.windows.shape[1:] != (self.seq_len, self.num_features):
            raise ValueError(
                f"windows must have shape [N,{self.seq_len},{self.num_features}], got {self.windows.shape}"
            )
        if self.num_middle != self.seq_len - 2:
            raise ValueError("keyframe segment training expects num_middle == seq_len - 2")
        n = self.windows.shape[0]
        for name, array in (
            ("source_ids", self.source_ids),
            ("dataset_ids", self.dataset_ids),
            ("original_gap_frames", self.original_gap_frames),
            ("durations_sec", self.durations_sec),
        ):
            if len(array) != n:
                raise ValueError(f"{name} length {len(array)} does not match windows length {n}")

        if bool(dataset_config.get("clamp", False)):
            clamp_min = float(dataset_config.get("clamp_min", 0.0))
            clamp_max = float(dataset_config.get("clamp_max", 1.0))
            np.clip(self.windows, clamp_min, clamp_max, out=self.windows)

        self._split_cache = {}

    def split_indices(self, split, split_seed, split_ratios):
        split = _normalise_split_name(split)
        ratios = tuple(_as_ratios(split_ratios).tolist())
        cache_key = (int(split_seed), ratios)
        if cache_key not in self._split_cache:
            self._split_cache[cache_key] = self._build_split_indices(int(split_seed), ratios)
        return self._split_cache[cache_key][split]

    def _build_split_indices(self, split_seed, ratios):
        unique_sources = np.unique(self.source_ids)
        if len(unique_sources) < 3:
            raise ValueError("Need at least three unique source_ids to build train/valid/test splits")

        rng = np.random.default_rng(split_seed)
        shuffled_sources = unique_sources.copy()
        rng.shuffle(shuffled_sources)

        train_count = int(len(shuffled_sources) * ratios[0])
        valid_count = int(len(shuffled_sources) * ratios[1])
        if train_count <= 0 or valid_count <= 0 or train_count + valid_count >= len(shuffled_sources):
            raise ValueError(
                f"split_ratios={ratios} produce empty split for {len(shuffled_sources)} sources"
            )

        train_sources = shuffled_sources[:train_count]
        valid_sources = shuffled_sources[train_count : train_count + valid_count]
        test_sources = shuffled_sources[train_count + valid_count :]

        split_sources = {
            "train": train_sources,
            "valid": valid_sources,
            "test": test_sources,
        }
        split_indices = {}
        for name, source_ids in split_sources.items():
            mask = np.isin(self.source_ids, source_ids)
            split_indices[name] = np.flatnonzero(mask).astype(np.int64, copy=False)
        return split_indices


class KeyframeSegmentsT30Dataset(Dataset):
    def __init__(self, config, split="train", store=None):
        self.config = config
        self.dataset_config = config["dataset"]
        self.split = _normalise_split_name(split)
        self.store = store or KeyframeSegmentsT30Store(config)

        split_seed = int(self.dataset_config.get("split_seed", config.get("seed", 1)))
        split_ratios = self.dataset_config.get("split_ratios", [0.8, 0.1, 0.1])
        self.indices = self.store.split_indices(self.split, split_seed, split_ratios)
        if len(self.indices) == 0:
            raise ValueError(f"No samples assigned to split {self.split!r}")

        self.seq_len = self.store.seq_len
        self.num_features = self.store.num_features
        self.num_middle = self.store.num_middle

        self.observed_mask = np.zeros((self.seq_len, self.num_features), dtype=np.float32)
        self.observed_mask[0] = 1.0
        self.observed_mask[-1] = 1.0
        self.target_mask = np.zeros((self.seq_len, self.num_features), dtype=np.float32)
        self.target_mask[1:-1] = 1.0
        self.timepoints = np.arange(self.seq_len, dtype=np.float32)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        sample_index = int(self.indices[index])
        window = self.store.windows[sample_index]
        source_id = int(self.store.source_ids[sample_index])
        dataset_id = int(self.store.dataset_ids[sample_index])

        sample = {
            "observed_data": window,
            "data": window,
            "observed_mask": self.observed_mask,
            "gt_mask": self.target_mask,
            "target_mask": self.target_mask,
            "timepoints": self.timepoints,
            "duration": np.float32(self.store.durations_sec[sample_index]),
            "original_gap_frames": np.int64(self.store.original_gap_frames[sample_index]),
            "condition": window[[0, -1]],
            "target": window[1:-1],
            "start": window[0],
            "end": window[-1],
            "middle": window[1:-1],
            "sample_index": np.int64(sample_index),
            "source_id": np.int64(source_id),
            "dataset_id": np.int64(dataset_id),
            "source_name": str(self.store.sources[source_id]),
            "source_path": str(self.store.source_paths[source_id]),
            "dataset_name": str(self.store.datasets[dataset_id]),
        }
        if self.store.keyframe_mask is not None:
            sample["keyframe_mask"] = self.store.keyframe_mask[sample_index]
        return sample


def get_dataloader(config, seed=1, batch_size=16, num_workers=0):
    if num_workers > 0:
        warnings.warn(
            "keyframe_segments_T30.npz decompresses windows into about 1.8GB. "
            "num_workers > 0 may duplicate that memory, especially on Windows.",
            RuntimeWarning,
        )

    store = KeyframeSegmentsT30Store(config)
    train_dataset = KeyframeSegmentsT30Dataset(config, split="train", store=store)
    valid_dataset = KeyframeSegmentsT30Dataset(config, split="valid", store=store)
    test_dataset = KeyframeSegmentsT30Dataset(config, split="test", store=store)

    generator = torch.Generator()
    generator.manual_seed(seed)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        drop_last=False,
        generator=generator,
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=batch_size,
        shuffle=False,
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

    print(f"Loaded {store.path}")
    print(f"windows: {store.windows.shape} {store.windows.dtype}")
    for split_name, dataset in (
        ("train", train_dataset),
        ("valid", valid_dataset),
        ("test", test_dataset),
    ):
        unique_sources = np.unique(store.source_ids[dataset.indices])
        print(f"{split_name}: {len(dataset)} samples from {len(unique_sources)} source_ids")

    return train_loader, valid_loader, test_loader
