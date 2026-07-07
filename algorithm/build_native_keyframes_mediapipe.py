"""Build a native-frame-rate keyframe dataset from MediaPipe-extracted CSVs.

Reads the per-clip ARKit-52 CSVs produced by
``scripts/extract_video_blendshapes_batch.py`` (columns: metadata + the 52
ARKit channels in canonical order) and writes one keyframe JSON per clip,
keeping every frame at the video's native rate (NO downsampling). RDP keyframe
indices are expressed on the native frame grid.

Works for any MediaPipe-extracted dataset (DFEW, RAVDESS, CelebV-HQ, ...).

Usage::

    python scripts/build_native_keyframes_mediapipe.py \
        --csv-dir data/mediapipe_extracted/dfew \
        --output-dir delivery/keyframe_dataset_native/dfew \
        --dataset dfew --native-fps 30 --epsilon 0.6
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from syntheticdata.constants import ARKIT_BLENDSHAPE_NAMES  # noqa: E402
from syntheticdata.keyframe_annotation import pick_keyframes  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--dataset", required=True, type=str)
    parser.add_argument("--native-fps", type=float, default=30.0)
    parser.add_argument("--method", choices=["rdp", "peak"], default="rdp")
    parser.add_argument("--epsilon", type=float, default=0.6)
    parser.add_argument("--max-keyframes", type=int, default=24)
    parser.add_argument("--max-files", type=int, default=0)
    parser.add_argument("--first-face-only", action="store_true", default=True,
                        help="Keep only face_index==0 rows (one track per clip).")
    parser.add_argument("--sweep-only", action="store_true")
    parser.add_argument("--summary-output", type=Path, default=None)
    return parser.parse_args()


def load_clip(csv_path: Path) -> np.ndarray | None:
    """Return blendshapes [T, 52] in canonical ARKit order, or None."""
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            return None
        missing = [n for n in ARKIT_BLENDSHAPE_NAMES if n not in reader.fieldnames]
        if missing:
            return None
        frames: list[list[float]] = []
        for row in reader:
            # one face track per clip
            if row.get("face_index") not in (None, "", "0"):
                continue
            try:
                frames.append([float(row[name]) for name in ARKIT_BLENDSHAPE_NAMES])
            except (TypeError, ValueError):
                continue
    if not frames:
        return None
    return np.asarray(frames, dtype=np.float32)


def main() -> None:
    args = parse_args()
    csv_paths = sorted(args.csv_dir.glob("*.csv"))
    if args.max_files > 0:
        csv_paths = csv_paths[: args.max_files]
    print(f"[native-kf] {len(csv_paths)} CSVs in {args.csv_dir}")

    if args.sweep_only:
        seqs = []
        for p in csv_paths[:200]:
            s = load_clip(p)
            if s is not None and s.shape[0] >= 4:
                seqs.append(s)
        frames = np.array([s.shape[0] for s in seqs])
        print(f"[sweep] clips={len(seqs)} mean_frames={frames.mean():.0f} "
              f"min={frames.min()} max={frames.max()}")
        for eps in [0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0]:
            ks = np.array([
                pick_keyframes(s, method="rdp", epsilon=eps,
                               max_keyframes=args.max_keyframes).indices.size
                for s in seqs
            ])
            print(f"[sweep] eps={eps:.2f}  mean_kf={ks.mean():5.1f}  "
                  f"median={int(np.median(ks))}  p90={int(np.percentile(ks,90))}  "
                  f"per_sec={ks.mean()/(frames.mean()/args.native_fps):.2f}/s")
        return

    out_dir = args.output_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    videos = kf_total = anchor_only = skipped = 0
    histogram: dict[str, int] = {}

    for index, csv_path in enumerate(csv_paths, start=1):
        blendshapes = load_clip(csv_path)
        if blendshapes is None or blendshapes.shape[0] < 2:
            skipped += 1
            continue
        num_frames = int(blendshapes.shape[0])
        annotation = pick_keyframes(
            blendshapes, method=args.method,
            epsilon=args.epsilon, max_keyframes=args.max_keyframes,
        )
        video_id = csv_path.stem
        frames_out = [
            {
                "frame_index": i,
                "timestamp_ms": int(round(i / args.native_fps * 1000)),
                "blendshapes": [round(float(v), 6) for v in blendshapes[i]],
            }
            for i in range(num_frames)
        ]
        document = {
            "schema_version": "kf-native-1",
            "video_id": video_id,
            "dataset": args.dataset,
            "source_representation": "arkit52_mediapipe",
            "native_fps": args.native_fps,
            "target_fps": args.native_fps,
            "num_frames": num_frames,
            "duration_sec": round(num_frames / args.native_fps, 3),
            "blendshape_names": list(ARKIT_BLENDSHAPE_NAMES),
            "frames": frames_out,
            "keyframe_indices": annotation.as_list(),
            "keyframe_annotation": {
                "method": annotation.method,
                "selector": args.method,
                "epsilon": float(args.epsilon),
                "max_keyframes": int(args.max_keyframes),
                "num_keyframes": int(annotation.indices.size),
                "num_interior": int(annotation.peak_indices.size),
                "max_error": float(annotation.max_error),
            },
        }
        (out_dir / f"{video_id}.json").write_text(
            json.dumps(document, ensure_ascii=False), encoding="utf-8"
        )
        videos += 1
        nk = int(annotation.indices.size)
        kf_total += nk
        if annotation.peak_indices.size == 0:
            anchor_only += 1
        key = str(nk) if nk <= 10 else "10+"
        histogram[key] = histogram.get(key, 0) + 1
        if index % 1000 == 0:
            print(f"[native-kf] {index}/{len(csv_paths)}  videos={videos}")

    summary = {
        "csv_dir": str(args.csv_dir),
        "output_dir": str(out_dir),
        "dataset": args.dataset,
        "native_fps": args.native_fps,
        "method": args.method,
        "epsilon": float(args.epsilon),
        "videos": videos,
        "skipped": skipped,
        "keyframes_total": kf_total,
        "avg_keyframes": round(kf_total / videos, 2) if videos else 0,
        "with_interior_keyframes": videos - anchor_only,
        "anchor_only": anchor_only,
        "histogram": histogram,
    }
    print("=== SUMMARY ===")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    out = args.summary_output or (out_dir.parent / f"keyframe_summary_{args.dataset}_native.json")
    out.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[native-kf] wrote summary to {out}")


if __name__ == "__main__":
    main()
