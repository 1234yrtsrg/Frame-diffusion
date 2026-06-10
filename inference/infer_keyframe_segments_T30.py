import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
CSDI_DIR = REPO_ROOT / "CSDI"
sys.path.insert(0, str(CSDI_DIR))

from infer_keyframes_express4d import DEFAULT_BLENDSHAPE_JSON, load_blendshape_keyframes
from main_model import CSDI_KeyframeSegmentsT30


DEFAULT_CONFIG = "CSDI/config/keyframe_segments_T30.yaml"
DEFAULT_CHECKPOINT = "save/keyframe_segments_T30/checkpoint_step_10000.pth"


def resolve_path(path):
    path = Path(path)
    if path.is_absolute():
        return path
    return REPO_ROOT / path


def set_seed(seed):
    if seed is None:
        return
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


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
    model = CSDI_KeyframeSegmentsT30(
        config,
        device,
        target_dim=config["dataset"].get("num_features", 52),
    ).to(device)
    model.load_state_dict(load_state_dict(checkpoint_path))
    model.eval()
    return model


def normalize_index(index, length, name):
    if not -length <= index < length:
        raise IndexError(f"{name}={index} is out of range for length {length}")
    return index % length


def load_keyframes_json(path, clamp=True, clamp_min=0.0, clamp_max=1.0):
    keyframes, image_paths = load_blendshape_keyframes(path)
    keyframes = np.asarray(keyframes, dtype=np.float32)
    keyframes = np.nan_to_num(keyframes, nan=0.0, posinf=0.0, neginf=0.0)
    if clamp:
        keyframes = np.clip(keyframes, clamp_min, clamp_max)
    return keyframes.astype(np.float32), image_paths


def save_outputs(output_dir, middle, start, end, args, source_meta):
    output_dir = resolve_path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    full_sequence = np.concatenate([start[None], middle, end[None]], axis=0).astype(np.float32)
    middle_path = output_dir / "middle_28.npy"
    full_path = output_dir / "full_sequence_30.npy"
    metadata_path = output_dir / "metadata.json"

    np.save(middle_path, middle.astype(np.float32))
    np.save(full_path, full_sequence)

    metadata = {
        "checkpoint": str(resolve_path(args.checkpoint)),
        "config": str(resolve_path(args.config)),
        "keyframes_json": str(resolve_path(args.keyframes_json)),
        "start_index": args.start_index,
        "end_index": args.end_index,
        "duration": args.duration,
        "num_samples": args.num_samples,
        "seed": args.seed,
        "middle_shape": list(middle.shape),
        "full_sequence_shape": list(full_sequence.shape),
        "middle_path": str(middle_path),
        "full_sequence_path": str(full_path),
        "selected_keyframes": source_meta,
    }
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print(f"saved middle frames: {middle_path} {middle.shape}")
    print(f"saved full sequence: {full_path} {full_sequence.shape}")
    print(f"saved metadata:      {metadata_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Infer 28 middle frames from two endpoint keyframes using the T30 model."
    )
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument(
        "--keyframes_json",
        default=DEFAULT_BLENDSHAPE_JSON,
        help="Path to a JSON file containing one or more blendshape keyframes.",
    )
    parser.add_argument(
        "--start_index",
        type=int,
        default=0,
        help="Start keyframe index inside the JSON list. Negative values are allowed.",
    )
    parser.add_argument(
        "--end_index",
        type=int,
        default=-1,
        help="End keyframe index inside the JSON list. Negative values are allowed.",
    )
    parser.add_argument("--output_dir", default="outputs/keyframe_segments_T30_infer")
    parser.add_argument("--duration", type=float, default=None)
    parser.add_argument("--num_samples", type=int, default=1)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--clip_output",
        action="store_true",
        help="Clip generated frames to [clamp_min, clamp_max] from the config.",
    )
    args = parser.parse_args()

    if args.num_samples != 1:
        raise ValueError("This script writes one final sequence; keep --num_samples 1")

    set_seed(args.seed)

    device = args.device
    if device == "auto":
        device = "cuda:0" if torch.cuda.is_available() else "cpu"

    config = load_config(args.config)
    dataset_config = config["dataset"]
    clamp = bool(dataset_config.get("clamp", True))
    clamp_min = float(dataset_config.get("clamp_min", 0.0))
    clamp_max = float(dataset_config.get("clamp_max", 1.0))

    keyframes, image_paths = load_keyframes_json(
        args.keyframes_json,
        clamp=clamp,
        clamp_min=clamp_min,
        clamp_max=clamp_max,
    )
    start_index = normalize_index(args.start_index, len(keyframes), "start_index")
    end_index = normalize_index(args.end_index, len(keyframes), "end_index")
    if start_index == end_index:
        raise ValueError("start_index and end_index must point to different keyframes")

    start = keyframes[start_index]
    end = keyframes[end_index]
    model = load_model(config, args.checkpoint, device)

    with torch.no_grad():
        middle = model.generate_middle(
            torch.from_numpy(start),
            torch.from_numpy(end),
            args.duration,
            num_samples=args.num_samples,
        )

    middle = middle.detach().cpu().numpy().astype(np.float32)[0]
    if middle.shape != (28, 52):
        raise AssertionError(f"Expected middle shape (28, 52), got {middle.shape}")
    if args.clip_output:
        middle = np.clip(middle, clamp_min, clamp_max).astype(np.float32)

    source_meta = {
        "total_keyframes": int(len(keyframes)),
        "start_image_path": image_paths[start_index]
        if len(image_paths) == len(keyframes)
        else None,
        "end_image_path": image_paths[end_index] if len(image_paths) == len(keyframes) else None,
        "start_index": start_index,
        "end_index": end_index,
    }
    save_outputs(args.output_dir, middle, start, end, args, source_meta)


if __name__ == "__main__":
    main()
