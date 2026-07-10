import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
CSDI_DIR = REPO_ROOT / "CSDI"
sys.path.insert(0, str(CSDI_DIR))

from infer_keyframes_express4d import (  # noqa: E402
    ARKIT_52_NAMES,
    DEFAULT_BLENDSHAPE_JSON,
    load_blendshape_keyframes,
    resolve_path,
)


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


def interpolate_segment(start, end, frames_per_segment):
    alpha = np.linspace(0.0, 1.0, frames_per_segment, dtype=np.float32)[:, None]
    return start[None] * (1.0 - alpha) + end[None] * alpha


def generate_sequence(keyframes, frames_per_segment, clip_output):
    if frames_per_segment < 2:
        raise ValueError("--frames_per_segment must be at least 2")

    sequence_parts = []
    middle_parts = []
    segment_meta = []

    for segment_index in range(len(keyframes) - 1):
        start = keyframes[segment_index].astype(np.float32, copy=False)
        end = keyframes[segment_index + 1].astype(np.float32, copy=False)
        segment = interpolate_segment(start, end, frames_per_segment).astype(np.float32)
        if clip_output:
            segment = np.clip(segment, 0.0, 1.0).astype(np.float32)

        middle = segment[1:-1]
        sequence_parts.append(segment if segment_index == 0 else segment[1:])
        middle_parts.append(middle)
        segment_meta.append(
            {
                "segment_index": segment_index,
                "start_keyframe_index": segment_index,
                "end_keyframe_index": segment_index + 1,
                "frames_per_segment": int(frames_per_segment),
                "generated_middle_frames": int(frames_per_segment - 2),
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
        description="Linearly interpolate middle blendshape frames from data/blendshapes.json."
    )
    parser.add_argument("--keyframes_json", default=DEFAULT_BLENDSHAPE_JSON)
    parser.add_argument(
        "--frame_indices",
        default=None,
        help="Optional comma-separated source frame indices, for example 0,3,7. Defaults to all records.",
    )
    parser.add_argument(
        "--frames_per_segment",
        type=int,
        default=12,
        help="Frames per adjacent keyframe segment, including start and end.",
    )
    parser.add_argument("--output_dir", default="outputs/blendshapes_linear")
    parser.add_argument("--clip_output", action="store_true")
    args = parser.parse_args()

    all_keyframes, image_paths = load_blendshape_keyframes(args.keyframes_json)
    frame_indices = parse_frame_indices(args.frame_indices, len(all_keyframes))
    keyframes = all_keyframes[frame_indices]

    generated_sequence, middle_parts, segment_meta = generate_sequence(
        keyframes,
        frames_per_segment=args.frames_per_segment,
        clip_output=args.clip_output,
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
        "method": "linear_interpolation",
        "keyframes_json": str(resolve_path(args.keyframes_json)),
        "source_frame_indices": frame_indices,
        "image_paths": selected_image_paths,
        "arkit_52_names": ARKIT_52_NAMES,
        "frames_per_segment": int(args.frames_per_segment),
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
