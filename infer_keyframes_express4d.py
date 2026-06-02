import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parent
CSDI_DIR = REPO_ROOT / "CSDI"
sys.path.insert(0, str(CSDI_DIR))


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


DEFAULT_BLENDSHAPE_JSON = "data/blendshapes.json"
TONGUE_OUT_DEFAULT = 0.0


def _resolve_repo_path(path):
    path = Path(path)
    if path.is_absolute():
        return path
    return REPO_ROOT / path


def _get_blendshape_mapping(item, frame_index):
    if isinstance(item, dict):
        if "faces" in item:
            faces = item.get("faces") or []
            if not faces:
                raise ValueError(f"Keyframe {frame_index} has no faces")
            item = faces[0]
        if "blendshapes" in item:
            blendshapes = item["blendshapes"]
            if not isinstance(blendshapes, dict):
                raise ValueError(f"Keyframe {frame_index} blendshapes must be an object")
            return blendshapes
        normalized_keys = {str(key).strip().lower() for key in item.keys()}
        if all(name.lower() in normalized_keys for name in ARKIT_52_NAMES if name != "tongueOut"):
            return item
    return None


def _keyframe_to_vector(item, frame_index):
    if isinstance(item, (list, tuple)):
        vector = np.asarray(item, dtype=np.float32).reshape(-1)
        if vector.shape[0] != len(ARKIT_52_NAMES):
            raise ValueError(
                f"Keyframe {frame_index} must contain {len(ARKIT_52_NAMES)} values, got {vector.shape[0]}"
            )
        return vector

    blendshapes = _get_blendshape_mapping(item, frame_index)
    if blendshapes is None:
        raise ValueError(
            f"Keyframe {frame_index} must be a 52-value list or contain faces[0].blendshapes"
        )

    normalized = {str(key).strip().lower(): value for key, value in blendshapes.items()}
    values = []
    missing = []
    for name in ARKIT_52_NAMES:
        key = name.lower()
        if key in normalized:
            values.append(normalized[key])
        elif name == "tongueOut":
            values.append(TONGUE_OUT_DEFAULT)
        else:
            missing.append(name)
    if missing:
        raise ValueError(f"Keyframe {frame_index} missing blendshape values: {missing}")
    return np.asarray(values, dtype=np.float32)


def _get_keyframe_records(data):
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        for key in ("keyframes", "frames", "blendshapes"):
            records = data.get(key)
            if records is not None:
                if not isinstance(records, list):
                    raise ValueError(f"Top-level '{key}' must be a list")
                return records
        return [data]
    raise ValueError("Blendshape JSON must be a list or object")


def load_blendshape_keyframes(path=DEFAULT_BLENDSHAPE_JSON):
    path = _resolve_repo_path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Blendshape keyframe JSON not found: {path}")

    with open(path, "r", encoding="utf-8-sig") as f:
        data = json.load(f)

    records = _get_keyframe_records(data)
    if len(records) < 2:
        raise ValueError(f"At least two keyframes are required in {path}")

    keyframes = np.stack(
        [_keyframe_to_vector(item, index) for index, item in enumerate(records)],
        axis=0,
    ).astype(np.float32)
    keyframes = np.nan_to_num(keyframes, nan=0.0, posinf=0.0, neginf=0.0)
    keyframes = np.clip(keyframes, 0.0, 1.0).astype(np.float32)

    image_paths = [
        item.get("image_path")
        for item in records
        if isinstance(item, dict) and item.get("image_path")
    ]
    return keyframes, image_paths


BLENDSHAPE_KEYFRAMES, IMAGE_PATHS = load_blendshape_keyframes(DEFAULT_BLENDSHAPE_JSON)


def resolve_path(path):
    path = Path(path)
    if path.is_absolute():
        return path
    return REPO_ROOT / path


def load_config(config_path):
    import yaml

    config_path = resolve_path(config_path)
    if not config_path.is_file():
        raise FileNotFoundError(f"Config not found: {config_path}")
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_model(config, checkpoint_path, device):
    import torch
    from main_model import CSDI_Express4D

    checkpoint_path = resolve_path(checkpoint_path)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    model = CSDI_Express4D(
        config,
        device,
        target_dim=config["dataset"].get("num_features", 52),
    ).to(device)

    state = torch.load(checkpoint_path, map_location=device)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    if all(key.startswith("module.") for key in state.keys()):
        state = {key.removeprefix("module."): value for key, value in state.items()}
    model.load_state_dict(state)
    model.eval()
    return model


def infer_keyframe_sequence(model, keyframes, duration, num_samples):
    import torch

    if keyframes.ndim != 2 or keyframes.shape[1] != 52:
        raise ValueError(f"keyframes must have shape [N,52], got {keyframes.shape}")
    if len(keyframes) < 2:
        raise ValueError("At least two keyframes are required")

    keyframes = np.nan_to_num(keyframes, nan=0.0, posinf=0.0, neginf=0.0)
    keyframes = np.clip(keyframes, 0.0, 1.0).astype(np.float32)

    full_sequence = []
    frame_meta = []
    segment_middles = []

    for segment_index in range(len(keyframes) - 1):
        start_np = keyframes[segment_index]
        end_np = keyframes[segment_index + 1]

        start = torch.from_numpy(start_np)
        end = torch.from_numpy(end_np)
        with torch.no_grad():
            middle = model.generate_middle(start, end, duration, num_samples=num_samples)
        middle_np = middle.detach().cpu().numpy().astype(np.float32)

        if num_samples != 1:
            raise ValueError("This script writes one final sequence; keep --num_samples 1")

        middle_np = middle_np[0]
        segment_middles.append(middle_np)
        segment_full = np.concatenate([start_np[None], middle_np, end_np[None]], axis=0)

        first_frame = 0 if segment_index == 0 else 1
        for local_index, values in enumerate(segment_full[first_frame:], start=first_frame):
            full_sequence.append(values)
            if local_index == 0:
                kind = "keyframe_start"
                keyframe_index = segment_index
            elif local_index == len(segment_full) - 1:
                kind = "keyframe_end"
                keyframe_index = segment_index + 1
            else:
                kind = "generated_middle"
                keyframe_index = None
            frame_meta.append(
                {
                    "frame_index": len(frame_meta),
                    "segment_index": segment_index,
                    "segment_local_index": local_index,
                    "kind": kind,
                    "keyframe_index": keyframe_index,
                }
            )

    return np.stack(full_sequence, axis=0).astype(np.float32), segment_middles, frame_meta


def save_csv(path, sequence):
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["frame"] + ARKIT_52_NAMES)
        for index, row in enumerate(sequence):
            writer.writerow([index] + [float(value) for value in row])


def main():
    parser = argparse.ArgumentParser(description="Infer Express4D sequence from keyframe JSON.")
    parser.add_argument("--config", default="CSDI/config/express4d.yaml")
    parser.add_argument(
        "--checkpoint",
        default="save/express4d_20260528_023203/checkpoint_step_10000.pth",
    )
    parser.add_argument("--duration", "--duraction", dest="duration", type=float, default=3.0)
    parser.add_argument("--num_samples", type=int, default=1)
    parser.add_argument("--keyframes_json", default=DEFAULT_BLENDSHAPE_JSON)
    parser.add_argument("--output_dir", default="outputs/keyframe_infer")
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    import torch

    device = args.device
    if device == "auto":
        device = "cuda:0" if torch.cuda.is_available() else "cpu"

    config = load_config(args.config)
    model = load_model(config, args.checkpoint, device)
    keyframes, image_paths = load_blendshape_keyframes(args.keyframes_json)

    sequence, segment_middles, frame_meta = infer_keyframe_sequence(
        model,
        keyframes,
        duration=args.duration,
        num_samples=args.num_samples,
    )

    output_dir = resolve_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    np.save(output_dir / "keyframes.npy", keyframes)
    np.save(output_dir / "generated_sequence.npy", sequence)
    save_csv(output_dir / "generated_sequence.csv", sequence)

    for index, middle in enumerate(segment_middles):
        np.save(output_dir / f"segment_{index:03d}_{index + 1:03d}_middle.npy", middle)

    metadata = {
        "checkpoint": str(resolve_path(args.checkpoint)),
        "config": str(resolve_path(args.config)),
        "keyframes_json": str(resolve_path(args.keyframes_json)),
        "duration": args.duration,
        "duration_unit": "seconds",
        "duration_usage": "passed to each adjacent keyframe segment",
        "image_paths": image_paths,
        "arkit_52_names": ARKIT_52_NAMES,
        "input_keyframes_shape": list(keyframes.shape),
        "generated_sequence_shape": list(sequence.shape),
        "frame_meta": frame_meta,
    }
    with open(output_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print(f"Saved generated sequence: {output_dir / 'generated_sequence.npy'} {sequence.shape}")
    print(f"Saved generated CSV:      {output_dir / 'generated_sequence.csv'}")
    print(f"Saved metadata:           {output_dir / 'metadata.json'}")


if __name__ == "__main__":
    main()

