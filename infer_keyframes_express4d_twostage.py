import argparse
import json
from pathlib import Path

import numpy as np

from infer_keyframes_express4d import (
    ARKIT_52_NAMES,
    BLENDSHAPE_KEYFRAMES,
    IMAGE_PATHS,
    load_config,
    load_model,
    resolve_path,
    save_csv,
)


def call_model_segment(model, start, end, duration):
    import torch

    start_tensor = torch.from_numpy(np.asarray(start, dtype=np.float32))
    end_tensor = torch.from_numpy(np.asarray(end, dtype=np.float32))
    with torch.no_grad():
        middle = model.generate_middle(start_tensor, end_tensor, duration, num_samples=1)
    middle = middle.detach().cpu().numpy().astype(np.float32)[0]
    return np.concatenate([start_tensor.numpy()[None], middle, end_tensor.numpy()[None]], axis=0)


def resample_sequence(sequence, out_len):
    sequence = np.asarray(sequence, dtype=np.float32)
    if out_len <= 0:
        raise ValueError("out_len must be positive")
    if len(sequence) == out_len:
        return sequence.copy()
    if out_len == 1:
        return sequence[:1].copy()

    source_pos = np.linspace(0.0, 1.0, len(sequence), dtype=np.float32)
    target_pos = np.linspace(0.0, 1.0, out_len, dtype=np.float32)
    columns = [
        np.interp(target_pos, source_pos, sequence[:, feature_index])
        for feature_index in range(sequence.shape[1])
    ]
    return np.stack(columns, axis=1).astype(np.float32)


def default_keyframe_positions(num_keyframes, coarse_frames):
    if num_keyframes < 2:
        raise ValueError("At least two keyframes are required")
    if coarse_frames < num_keyframes:
        raise ValueError("coarse_frames must be >= number of keyframes")

    positions = np.rint(np.linspace(0, coarse_frames - 1, num_keyframes)).astype(np.int64)
    positions[0] = 0
    positions[-1] = coarse_frames - 1
    if len(np.unique(positions)) != len(positions):
        raise ValueError(f"Keyframe positions are not unique: {positions.tolist()}")
    return positions


def build_coarse_sequence(model, keyframes, total_duration, coarse_frames):
    keyframes = np.asarray(keyframes, dtype=np.float32)
    keyframes = np.nan_to_num(keyframes, nan=0.0, posinf=0.0, neginf=0.0)
    keyframes = np.clip(keyframes, 0.0, 1.0).astype(np.float32)

    positions = default_keyframe_positions(len(keyframes), coarse_frames)
    total_intervals = coarse_frames - 1
    coarse_parts = []
    segment_meta = []

    for segment_index in range(len(keyframes) - 1):
        start_pos = int(positions[segment_index])
        end_pos = int(positions[segment_index + 1])
        interval_count = end_pos - start_pos
        segment_duration = float(total_duration * interval_count / total_intervals)

        generated_12 = call_model_segment(
            model,
            keyframes[segment_index],
            keyframes[segment_index + 1],
            segment_duration,
        )
        resampled = resample_sequence(generated_12, interval_count + 1)

        if segment_index > 0:
            resampled = resampled[1:]
        coarse_parts.append(resampled)
        segment_meta.append(
            {
                "segment_index": segment_index,
                "start_keyframe_index": segment_index,
                "end_keyframe_index": segment_index + 1,
                "start_frame": start_pos,
                "end_frame": end_pos,
                "interval_count": interval_count,
                "duration": segment_duration,
                "raw_model_frames": int(generated_12.shape[0]),
                "coarse_frames_written": int(len(resampled)),
            }
        )

    coarse_sequence = np.concatenate(coarse_parts, axis=0).astype(np.float32)
    if coarse_sequence.shape != (coarse_frames, 52):
        raise AssertionError(f"Expected coarse shape {(coarse_frames, 52)}, got {coarse_sequence.shape}")
    return coarse_sequence, positions, segment_meta


def refine_adjacent_sequence(model, coarse_sequence, total_duration):
    coarse_sequence = np.asarray(coarse_sequence, dtype=np.float32)
    interval_duration = float(total_duration / (len(coarse_sequence) - 1))
    refined_frames = []
    refined_meta = []

    for pair_index in range(len(coarse_sequence) - 1):
        segment = call_model_segment(
            model,
            coarse_sequence[pair_index],
            coarse_sequence[pair_index + 1],
            interval_duration,
        )
        first_frame = 0 if pair_index == 0 else 1
        refined_frames.append(segment[first_frame:])
        refined_meta.append(
            {
                "pair_index": pair_index,
                "start_coarse_frame": pair_index,
                "end_coarse_frame": pair_index + 1,
                "duration": interval_duration,
                "generated_middle_frames": 10,
                "frames_written": int(len(segment[first_frame:])),
            }
        )

    refined_sequence = np.concatenate(refined_frames, axis=0).astype(np.float32)
    return refined_sequence, interval_duration, refined_meta


def main():
    parser = argparse.ArgumentParser(
        description="Two-stage Express4D inference: 3 keyframes -> 12 coarse frames -> adjacent 10-frame refinement."
    )
    parser.add_argument("--config", default="CSDI/config/express4d.yaml")
    parser.add_argument(
        "--checkpoint",
        default="save/express4d_20260528_032120/checkpoint_step_50000.pth",
    )
    parser.add_argument("--duration", "--duraction", dest="duration", type=float, default=3.0)
    parser.add_argument("--coarse_frames", type=int, default=12)
    parser.add_argument("--output_dir", default="outputs/keyframe_infer_twostage")
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    import torch

    device = args.device
    if device == "auto":
        device = "cuda:0" if torch.cuda.is_available() else "cpu"

    config = load_config(args.config)
    model = load_model(config, args.checkpoint, device)

    coarse_sequence, keyframe_positions, coarse_segment_meta = build_coarse_sequence(
        model,
        BLENDSHAPE_KEYFRAMES,
        total_duration=args.duration,
        coarse_frames=args.coarse_frames,
    )
    refined_sequence, adjacent_duration, refined_segment_meta = refine_adjacent_sequence(
        model,
        coarse_sequence,
        total_duration=args.duration,
    )

    output_dir = resolve_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    np.save(output_dir / "keyframes.npy", BLENDSHAPE_KEYFRAMES)
    np.save(output_dir / "coarse_12_sequence.npy", coarse_sequence)
    np.save(output_dir / "refined_sequence.npy", refined_sequence)
    save_csv(output_dir / "coarse_12_sequence.csv", coarse_sequence)
    save_csv(output_dir / "refined_sequence.csv", refined_sequence)

    metadata = {
        "checkpoint": str(resolve_path(args.checkpoint)),
        "config": str(resolve_path(args.config)),
        "image_paths": IMAGE_PATHS,
        "arkit_52_names": ARKIT_52_NAMES,
        "total_duration": args.duration,
        "duration_unit": "seconds",
        "input_keyframes_shape": list(BLENDSHAPE_KEYFRAMES.shape),
        "coarse_frames": args.coarse_frames,
        "coarse_sequence_shape": list(coarse_sequence.shape),
        "keyframe_positions_in_coarse_sequence": keyframe_positions.astype(int).tolist(),
        "coarse_stage": coarse_segment_meta,
        "adjacent_refine_duration": adjacent_duration,
        "refined_sequence_shape": list(refined_sequence.shape),
        "refined_stage": refined_segment_meta,
    }
    with open(output_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print(f"Saved coarse sequence:  {output_dir / 'coarse_12_sequence.npy'} {coarse_sequence.shape}")
    print(f"Saved refined sequence: {output_dir / 'refined_sequence.npy'} {refined_sequence.shape}")
    print(f"Adjacent duration:      {adjacent_duration}")
    print(f"Saved metadata:         {output_dir / 'metadata.json'}")


if __name__ == "__main__":
    main()
