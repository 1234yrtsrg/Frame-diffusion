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
DEFAULT_CHECKPOINT = "save/keyframe_segments_T30/checkpoint_step_50000.pth"


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


def load_keyframes_json(path, clamp=True, clamp_min=0.0, clamp_max=1.0):
    keyframes, image_paths = load_blendshape_keyframes(path)
    keyframes = np.asarray(keyframes, dtype=np.float32)
    keyframes = np.nan_to_num(keyframes, nan=0.0, posinf=0.0, neginf=0.0)
    if clamp:
        keyframes = np.clip(keyframes, clamp_min, clamp_max)
    return keyframes.astype(np.float32), image_paths


def generate_sequence(model, keyframes, duration, clip_output, clamp_min, clamp_max):
    sequence_parts = []
    segment_meta = []

    for segment_index in range(len(keyframes) - 1):
        start = keyframes[segment_index]
        end = keyframes[segment_index + 1]

        with torch.no_grad():
            middle = model.generate_middle(
                torch.from_numpy(start),
                torch.from_numpy(end),
                duration,
                num_samples=1,
            )

        middle = middle.detach().cpu().numpy().astype(np.float32)[0]
        if middle.shape != (28, 52):
            raise AssertionError(f"Expected middle shape (28, 52), got {middle.shape}")
        if clip_output:
            middle = np.clip(middle, clamp_min, clamp_max).astype(np.float32)

        segment = np.concatenate([start[None], middle, end[None]], axis=0)
        sequence_parts.append(segment if segment_index == 0 else segment[1:])
        segment_meta.append(
            {
                "segment_index": segment_index,
                "start_keyframe_index": segment_index,
                "end_keyframe_index": segment_index + 1,
                "output_start_frame": segment_index * 29,
                "output_end_frame": (segment_index + 1) * 29,
            }
        )

    generated_sequence = np.concatenate(sequence_parts, axis=0).astype(np.float32)
    return generated_sequence, segment_meta


def save_outputs(
    output_dir,
    keyframes,
    generated_sequence,
    image_paths,
    segment_meta,
    args,
):
    output_dir = resolve_path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    keyframes_path = output_dir / "keyframes.npy"
    sequence_path = output_dir / "generated_sequence.npy"
    metadata_path = output_dir / "metadata.json"

    for obsolete_name in ("middle_28.npy", "full_sequence_30.npy", "segment_middles.npy"):
        obsolete_path = output_dir / obsolete_name
        if obsolete_path.is_file():
            obsolete_path.unlink()

    np.save(keyframes_path, keyframes)
    np.save(sequence_path, generated_sequence)

    metadata = {
        "checkpoint": str(resolve_path(args.checkpoint)),
        "config": str(resolve_path(args.config)),
        "keyframes_json": str(resolve_path(args.keyframes_json)),
        "duration": args.duration,
        "num_samples": args.num_samples,
        "seed": args.seed,
        "input_keyframes_shape": list(keyframes.shape),
        "generated_sequence_shape": list(generated_sequence.shape),
        "image_paths": image_paths,
        "keyframe_output_positions": [index * 29 for index in range(len(keyframes))],
        "segments": segment_meta,
        "keyframes_path": str(keyframes_path),
        "generated_sequence_path": str(sequence_path),
    }
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print(f"saved keyframes:          {keyframes_path} {keyframes.shape}")
    print(f"saved generated sequence: {sequence_path} {generated_sequence.shape}")
    print(f"saved metadata:           {metadata_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Fill 28 frames between every adjacent keyframe in a blendshape JSON."
    )
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument(
        "--keyframes_json",
        default=DEFAULT_BLENDSHAPE_JSON,
        help="Path to a JSON file containing one or more blendshape keyframes.",
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
    model = load_model(config, args.checkpoint, device)
    generated_sequence, segment_meta = generate_sequence(
        model,
        keyframes,
        duration=args.duration,
        clip_output=args.clip_output,
        clamp_min=clamp_min,
        clamp_max=clamp_max,
    )

    expected_frames = 1 + 29 * (len(keyframes) - 1)
    if generated_sequence.shape != (expected_frames, 52):
        raise AssertionError(
            f"Expected generated shape {(expected_frames, 52)}, "
            f"got {generated_sequence.shape}"
        )

    save_outputs(
        args.output_dir,
        keyframes,
        generated_sequence,
        image_paths,
        segment_meta,
        args,
    )


if __name__ == "__main__":
    main()
