from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


EXPRESS4D_61_NAMES = [
    "EyeBlinkLeft",
    "EyeLookDownLeft",
    "EyeLookInLeft",
    "EyeLookOutLeft",
    "EyeLookUpLeft",
    "EyeSquintLeft",
    "EyeWideLeft",
    "EyeBlinkRight",
    "EyeLookDownRight",
    "EyeLookInRight",
    "EyeLookOutRight",
    "EyeLookUpRight",
    "EyeSquintRight",
    "EyeWideRight",
    "JawForward",
    "JawRight",
    "JawLeft",
    "JawOpen",
    "MouthClose",
    "MouthFunnel",
    "MouthPucker",
    "MouthRight",
    "MouthLeft",
    "MouthSmileLeft",
    "MouthSmileRight",
    "MouthFrownLeft",
    "MouthFrownRight",
    "MouthDimpleLeft",
    "MouthDimpleRight",
    "MouthStretchLeft",
    "MouthStretchRight",
    "MouthRollLower",
    "MouthRollUpper",
    "MouthShrugLower",
    "MouthShrugUpper",
    "MouthPressLeft",
    "MouthPressRight",
    "MouthLowerDownLeft",
    "MouthLowerDownRight",
    "MouthUpperUpLeft",
    "MouthUpperUpRight",
    "BrowDownLeft",
    "BrowDownRight",
    "BrowInnerUp",
    "BrowOuterUpLeft",
    "BrowOuterUpRight",
    "CheekPuff",
    "CheekSquintLeft",
    "CheekSquintRight",
    "NoseSneerLeft",
    "NoseSneerRight",
    "TongueOut",
    "HeadYaw",
    "HeadPitch",
    "HeadRoll",
    "LeftEyeYaw",
    "LeftEyePitch",
    "LeftEyeRoll",
    "RightEyeYaw",
    "RightEyePitch",
    "RightEyeRoll",
]


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


def normalize_name(name):
    return str(name).strip().lower()


EXPRESS4D_INDEX = {
    normalize_name(name): index for index, name in enumerate(EXPRESS4D_61_NAMES)
}
ARKIT_FROM_EXPRESS4D = [EXPRESS4D_INDEX[normalize_name(name)] for name in ARKIT_52_NAMES]


def _repo_root():
    return Path(__file__).resolve().parents[1]


def resolve_dataset_root(root):
    root_path = Path(root)
    if root_path.is_absolute():
        return root_path
    return _repo_root() / root_path


def _resolve_entry(entry, root, data_dir, use_npy_first=True):
    raw = entry.strip()
    if not raw:
        raise ValueError("empty dataset list entry")
    raw = raw.replace("\\", "/")
    rel = Path(raw)
    if rel.is_absolute():
        base = rel
    elif len(rel.parts) > 0 and rel.parts[0].lower() == data_dir.name.lower():
        base = root / rel
    elif rel.parent == Path("."):
        base = data_dir / rel
    else:
        base = root / rel

    suffix = base.suffix.lower()
    candidates = []
    if suffix == ".npy":
        candidates.append(base)
    elif suffix == ".csv":
        if use_npy_first:
            candidates.append(base.with_suffix(".npy"))
        candidates.append(base)
    elif suffix:
        candidates.append(base)
    else:
        if use_npy_first:
            candidates.extend([base.with_suffix(".npy"), base.with_suffix(".csv")])
        else:
            candidates.extend([base.with_suffix(".csv"), base.with_suffix(".npy")])

    for candidate in candidates:
        if candidate.is_file():
            return candidate
    searched = ", ".join(str(path) for path in candidates)
    raise FileNotFoundError(f"Could not find data file for '{entry}'. Tried: {searched}")


def _read_list(list_path):
    if not list_path.is_file():
        raise FileNotFoundError(f"Express4D split file not found: {list_path}")
    entries = []
    for line in list_path.read_text(encoding="utf-8-sig").splitlines():
        item = line.strip()
        if not item or item.startswith("#"):
            continue
        entries.append(item)
    if not entries:
        raise ValueError(f"Express4D split file is empty: {list_path}")
    return entries


def _as_t_by_61(array, path):
    array = np.asarray(array)
    if array.ndim != 2:
        raise ValueError(f"{path} must be 2D, got shape {array.shape}")
    if array.shape[1] == 61:
        out = array
    elif array.shape[0] == 61:
        out = array.T
    elif array.shape[1] > 61:
        out = array[:, :61]
    else:
        raise ValueError(f"{path} must contain 61 Express4D columns, got shape {array.shape}")
    if out.shape[1] != 61:
        raise AssertionError(f"{path} conversion failed, expected [T,61], got {out.shape}")
    return out


def _read_npy(path):
    data = np.load(path)
    data = _as_t_by_61(data, path)
    return data[:, ARKIT_FROM_EXPRESS4D]


def _read_csv(path):
    import pandas as pd

    header_df = pd.read_csv(path)
    normalized_cols = {normalize_name(col): col for col in header_df.columns}
    has_named_columns = all(normalize_name(name) in normalized_cols for name in ARKIT_52_NAMES)
    if has_named_columns:
        columns = [normalized_cols[normalize_name(name)] for name in ARKIT_52_NAMES]
        data = header_df[columns].apply(pd.to_numeric, errors="coerce").to_numpy()
        return data

    raw_df = pd.read_csv(path, header=None)
    if raw_df.shape[1] < 61:
        raise ValueError(f"{path} must contain at least 61 columns, got {raw_df.shape[1]}")
    data_61 = raw_df.iloc[:, :61].apply(pd.to_numeric, errors="coerce").to_numpy()
    return data_61[:, ARKIT_FROM_EXPRESS4D]


def load_blendshape_file(path, clamp=True, clamp_min=0.0, clamp_max=1.0):
    path = Path(path)
    if path.suffix.lower() == ".npy":
        data = _read_npy(path)
    elif path.suffix.lower() == ".csv":
        data = _read_csv(path)
    else:
        raise ValueError(f"Unsupported Express4D file extension: {path.suffix}")
    if data.size == 0:
        raise ValueError(f"Express4D file is empty: {path}")
    data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    if clamp:
        data = np.clip(data, clamp_min, clamp_max)
    if data.ndim != 2 or data.shape[1] != 52:
        raise AssertionError(f"Expected Express4D ARKit data shape [T,52], got {data.shape} from {path}")
    return data


def load_vector_52(path, clamp=True, clamp_min=0.0, clamp_max=1.0):
    import pandas as pd

    data = np.load(path) if Path(path).suffix.lower() == ".npy" else np.asarray(pd.read_csv(path, header=None))
    data = np.asarray(data)
    if data.ndim == 2 and 1 in data.shape:
        data = data.reshape(-1)
    if data.ndim != 1:
        raise ValueError(f"{path} must contain a [52] or [61] vector, got {data.shape}")
    if data.shape[0] == 61:
        data = data[ARKIT_FROM_EXPRESS4D]
    if data.shape[0] != 52:
        raise ValueError(f"{path} must contain 52 ARKit values, got {data.shape[0]}")
    data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    if clamp:
        data = np.clip(data, clamp_min, clamp_max)
    return data


def sample_linear_sequence(data, start_idx, end_idx, seq_len=12):
    positions = np.linspace(start_idx, end_idx, seq_len, dtype=np.float32)
    left = np.floor(positions).astype(np.int64)
    right = np.ceil(positions).astype(np.int64)
    weight = (positions - left).reshape(-1, 1)
    sampled = (1.0 - weight) * data[left] + weight * data[right]
    return sampled.astype(np.float32)


class Express4D_Dataset(Dataset):
    def __init__(self, config, split="train"):
        dataset_config = config["dataset"]
        self.root = resolve_dataset_root(dataset_config["root"])
        self.data_dir = self.root / dataset_config.get("data_dir", "data")
        list_name = dataset_config["train_list"] if split == "train" else dataset_config["test_list"]
        self.list_path = self.root / list_name
        self.fps = float(dataset_config.get("fps", 60))
        self.seq_len = int(dataset_config.get("seq_len", 12))
        self.num_middle = int(dataset_config.get("num_middle", self.seq_len - 2))
        self.gaps = [int(gap) for gap in dataset_config.get("gaps", [12, 24, 36, 60, 90, 120, 180, 240])]
        self.use_npy_first = bool(dataset_config.get("use_npy_first", True))
        self.clamp = bool(dataset_config.get("clamp", True))
        self.clamp_min = float(dataset_config.get("clamp_min", 0.0))
        self.clamp_max = float(dataset_config.get("clamp_max", 1.0))

        if self.seq_len != self.num_middle + 2:
            raise ValueError("Express4D expects seq_len == num_middle + 2")
        if not self.root.is_dir():
            raise FileNotFoundError(f"Express4D root not found: {self.root}")
        if not self.data_dir.is_dir():
            raise FileNotFoundError(f"Express4D data directory not found: {self.data_dir}")

        entries = _read_list(self.list_path)
        self.sequences = []
        self.samples = []
        for entry in entries:
            path = _resolve_entry(entry, self.root, self.data_dir, self.use_npy_first)
            data = load_blendshape_file(path, self.clamp, self.clamp_min, self.clamp_max)
            sequence_id = len(self.sequences)
            self.sequences.append({"name": path.stem, "path": path, "data": data})
            total_frames = data.shape[0]
            for gap in self.gaps:
                if total_frames <= gap:
                    continue
                for start_idx in range(0, total_frames - gap):
                    self.samples.append((sequence_id, start_idx, start_idx + gap, gap))
        if not self.samples:
            raise ValueError(
                f"No Express4D samples could be built from {self.list_path}. "
                f"Check sequence lengths and gaps={self.gaps}."
            )

    def __getitem__(self, index):
        sequence_id, start_idx, end_idx, gap = self.samples[index]
        sequence = self.sequences[sequence_id]
        seq = sample_linear_sequence(sequence["data"], start_idx, end_idx, self.seq_len)

        observed_mask = np.zeros_like(seq, dtype=np.float32)
        observed_mask[0] = 1.0
        observed_mask[-1] = 1.0
        target_mask = np.zeros_like(seq, dtype=np.float32)
        target_mask[1:-1] = 1.0

        sample = {
            "observed_data": seq,
            "data": seq,
            "observed_mask": observed_mask,
            "gt_mask": target_mask,
            "target_mask": target_mask,
            "timepoints": np.arange(self.seq_len, dtype=np.float32),
            "duration": np.float32(gap / self.fps),
            "start": seq[0],
            "end": seq[-1],
            "middle": seq[1:-1],
            "sequence_name": sequence["name"],
            "start_idx": np.int64(start_idx),
            "end_idx": np.int64(end_idx),
            "gap": np.int64(gap),
        }
        return sample

    def __len__(self):
        return len(self.samples)


def get_dataloader(config, seed=1, batch_size=16, num_workers=0):
    train_dataset = Express4D_Dataset(config, split="train")
    test_dataset = Express4D_Dataset(config, split="test")
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
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        drop_last=False,
    )
    return train_loader, test_loader
