import argparse
import csv
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
CSDI_DIR = REPO_ROOT / "CSDI"
sys.path.insert(0, str(CSDI_DIR))

from infer_keyframes_express4d import (  # noqa: E402
    ARKIT_52_NAMES,
    DEFAULT_BLENDSHAPE_JSON,
    load_blendshape_keyframes,
    load_config,
    resolve_path,
)
from main_model import CSDI_Express4D  # noqa: E402


DEFAULTS = {
    "express4d_condition": {
        "config": "CSDI/config/express4d_condition.yaml",
        "checkpoint": "save/express4d_condition/checkpoint_step_50000.pth",
    },
    "keyframe_dataset_60fps": {
        "config": "CSDI/config/keyframe_dataset_60fps.yaml",
        "checkpoint": "save/keyframe_dataset_60fps/checkpoint_step_50000.pth",
    },
}


def set_seed(seed):
    if seed is None:
        return
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


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


def parse_frame_indices(value, total_frames):
    if value is None:
        return list(range(total_frames))
    indices = [int(item.strip()) for item in value.split(",") if item.strip()]
    if len(indices) < 2:
        raise ValueError("--frame_indices must contain at least two indices")
    for index in indices:
        if index < 0 or index >= total_frames:
            raise ValueError(f"frame index {index} out of range [0, {total_frames - 1}]")
    return indices


def generate_sequence(model, keyframes, condition, num_samples, clip_output, clamp_min, clamp_max):
    if num_samples != 1:
        raise ValueError("This script writes one final sequence; keep --num_samples 1")

    sequence_parts = []
    middle_parts = []
    segment_meta = []

    for segment_index in range(len(keyframes) - 1):
        start_np = keyframes[segment_index].astype(np.float32, copy=False)
        end_np = keyframes[segment_index + 1].astype(np.float32, copy=False)

        start = torch.from_numpy(start_np)
        end = torch.from_numpy(end_np)
        condition_tensor = torch.tensor(float(condition))
        with torch.no_grad():
            middle = model.generate_middle(
                start,
                end,
                duration=None,
                condition=condition_tensor,
                num_samples=num_samples,
            )

        middle_np = middle.detach().cpu().numpy().astype(np.float32)[0]
        if middle_np.shape != (10, 52):
            raise AssertionError(f"Expected middle shape (10, 52), got {middle_np.shape}")
        if clip_output:
            middle_np = np.clip(middle_np, clamp_min, clamp_max).astype(np.float32)

        segment = np.concatenate([start_np[None], middle_np, end_np[None]], axis=0)
        sequence_parts.append(segment if segment_index == 0 else segment[1:])
        middle_parts.append(middle_np)
        segment_meta.append(
            {
                "segment_index": segment_index,
                "start_keyframe_index": segment_index,
                "end_keyframe_index": segment_index + 1,
                "condition": int(condition),
                "generated_middle_frames": 10,
                "frames_written": int(len(segment if segment_index == 0 else segment[1:])),
            }
        )

    return np.concatenate(sequence_parts, axis=0).astype(np.float32), middle_parts, segment_meta


def save_csv(path, sequence):
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["frame"] + ARKIT_52_NAMES)
        for index, row in enumerate(sequence):
            writer.writerow([index] + [float(value) for value in row])


def main():
    parser = argparse.ArgumentParser(
        description="Infer middle blendshape frames from data/blendshapes.json with a condition model."
    )
    parser.add_argument(
        "--model",
        choices=sorted(DEFAULTS),
        default="keyframe_dataset_60fps",
        help="Pick default config/checkpoint pair.",
    )
    parser.add_argument("--config", default=None)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--keyframes_json", default=DEFAULT_BLENDSHAPE_JSON)
    parser.add_argument(
        "--frame_indices",
        default=None,
        help="Optional comma-separated source frame indices, for example 0,3,7. Defaults to all records.",
    )
    parser.add_argument("--condition", type=int, default=3)
    parser.add_argument("--num_samples", type=int, default=1)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output_dir", default="outputs/blendshapes_condition_infer")
    parser.add_argument("--clip_output", action="store_true")
    args = parser.parse_args()

    set_seed(args.seed)

    device = args.device
    if device == "auto":
        device = "cuda:0" if torch.cuda.is_available() else "cpu"

    config_path = args.config or DEFAULTS[args.model]["config"]
    checkpoint_path = args.checkpoint or DEFAULTS[args.model]["checkpoint"]
    config = load_config(config_path)
    clamp_min = float(config.get("dataset", {}).get("clamp_min", 0.0))
    clamp_max = float(config.get("dataset", {}).get("clamp_max", 1.0))

    all_keyframes, image_paths = load_blendshape_keyframes(args.keyframes_json)
    frame_indices = parse_frame_indices(args.frame_indices, len(all_keyframes))
    keyframes = all_keyframes[frame_indices]

    model = load_model(config, checkpoint_path, device)
    generated_sequence, middle_parts, segment_meta = generate_sequence(
        model,
        keyframes,
        condition=args.condition,
        num_samples=args.num_samples,
        clip_output=args.clip_output,
        clamp_min=clamp_min,
        clamp_max=clamp_max,
    )

    output_dir = resolve_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    np.save(output_dir / "keyframes.npy", keyframes)
    np.save(output_dir / "generated_sequence.npy", generated_sequence)
    np.save(output_dir / "segment_middles.npy", np.stack(middle_parts, axis=0))
    save_csv(output_dir / "generated_sequence.csv", generated_sequence)

    selected_image_paths = [
        image_paths[index]
        for index in frame_indices
        if index < len(image_paths)
    ]
    metadata = {
        "model": args.model,
        "checkpoint": str(resolve_path(checkpoint_path)),
        "config": str(resolve_path(config_path)),
        "keyframes_json": str(resolve_path(args.keyframes_json)),
        "condition": int(args.condition),
        "condition_note": "1/2/3/4 correspond to training gaps such as 240/120/24/12 frames.",
        "source_frame_indices": frame_indices,
        "image_paths": selected_image_paths,
        "arkit_52_names": ARKIT_52_NAMES,
        "input_keyframes_shape": list(keyframes.shape),
        "generated_sequence_shape": list(generated_sequence.shape),
        "segments": segment_meta,
    }
    with open(output_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print(f"saved keyframes:          {output_dir / 'keyframes.npy'} {keyframes.shape}")
    print(f"saved generated sequence: {output_dir / 'generated_sequence.npy'} {generated_sequence.shape}")
    print(f"saved generated csv:      {output_dir / 'generated_sequence.csv'}")
    print(f"saved metadata:           {output_dir / 'metadata.json'}")


if __name__ == "__main__":
    main()
