"""Build a native-frame-rate keyframe dataset from raw Express4D CSVs.

Unlike the existing ``data/json_dataset/express4d`` (which was resampled down to
target_fps=10), this reads the ORIGINAL ~60 fps Express4D Live Link CSVs and
keeps every frame. RDP keyframe indices are therefore expressed on the native
frame grid, so no temporal information is lost. Per the meeting decision: keep
full frame rate, let multi-scale sampling happen later at training time.

Input  : a zip OR a directory of raw Express4D CSVs (Timecode, BlendshapeCount,
         51 blendshapes + TongueOut, 9 pose columns) -- e.g.
         Express4D_hf/express4d.zip or data/Express4D_reaction_dense_v1/data.
Output : one JSON per clip under ``--output-dir`` with the same schema the
         keyframe algorithm expects (frames[].blendshapes in ARKit-52 order,
         plus keyframe_indices / keyframe_annotation).

Usage::

    python scripts/build_native_keyframes_express4d.py \
        --zip C:/data/vla_external/Express4D_hf/express4d.zip \
        --output-dir delivery/keyframe_dataset_native/express4d \
        --epsilon 0.6
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import sys
import zipfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from syntheticdata.constants import ARKIT_BLENDSHAPE_NAMES  # noqa: E402
from syntheticdata.express4d_like import ARKIT_TO_EXPRESS4D  # noqa: E402
from syntheticdata.keyframe_annotation import pick_keyframes  # noqa: E402

# Express4D pose columns we carry through (head + per-eye yaw/pitch/roll).
POSE_COLUMNS = [
    "HeadYaw", "HeadPitch", "HeadRoll",
    "LeftEyeYaw", "LeftEyePitch", "LeftEyeRoll",
    "RightEyeYaw", "RightEyePitch", "RightEyeRoll",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--zip", type=Path, default=None, help="Raw Express4D CSV zip.")
    parser.add_argument("--csv-dir", type=Path, default=None,
                        help="Directory of raw Express4D CSVs (alternative to --zip).")
    parser.add_argument("--member-prefix", default="data/", help="CSV path prefix inside the zip.")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--dataset", default="express4d", help="dataset id written into each JSON.")
    parser.add_argument("--native-fps", type=float, default=60.0)
    parser.add_argument("--method", choices=["rdp", "peak"], default="rdp")
    parser.add_argument("--epsilon", type=float, default=0.6)
    parser.add_argument("--max-keyframes", type=int, default=24,
                        help="Higher than the 10fps default: native 60fps clips are longer.")
    parser.add_argument("--max-files", type=int, default=0, help="0 = all.")
    parser.add_argument("--sweep-only", action="store_true",
                        help="Only sweep epsilon on a sample and print stats; write nothing.")
    parser.add_argument("--summary-output", type=Path, default=None)
    args = parser.parse_args()
    if not args.zip and not args.csv_dir:
        parser.error("provide either --zip or --csv-dir")
    return args


def _arkit_indices_in_express4d(header: list[str]) -> list[int]:
    """Map each canonical ARKit-52 channel to its column index in a raw CSV."""
    col = {name: i for i, name in enumerate(header)}
    indices: list[int] = []
    for arkit_name in ARKIT_BLENDSHAPE_NAMES:
        express4d_name = ARKIT_TO_EXPRESS4D.get(arkit_name)
        if express4d_name is None or express4d_name not in col:
            indices.append(-1)            # missing channel -> filled with 0.0
        else:
            indices.append(col[express4d_name])
    return indices


def load_clip(raw_bytes: bytes) -> tuple[np.ndarray, np.ndarray, list[str]] | None:
    """Return (blendshapes [T,52], pose [T,9], pose_names) from one raw CSV."""
    text = raw_bytes.decode("utf-8-sig", errors="replace")
    reader = csv.reader(io.StringIO(text))
    rows = list(reader)
    if len(rows) < 2:
        return None
    header = rows[0]
    arkit_cols = _arkit_indices_in_express4d(header)
    pose_cols = [header.index(p) if p in header else -1 for p in POSE_COLUMNS]

    bs_frames: list[list[float]] = []
    pose_frames: list[list[float]] = []
    for row in rows[1:]:
        if len(row) < len(header):
            continue
        bs = [float(row[i]) if i >= 0 else 0.0 for i in arkit_cols]
        pose = [float(row[i]) if i >= 0 else 0.0 for i in pose_cols]
        bs_frames.append(bs)
        pose_frames.append(pose)
    if not bs_frames:
        return None
    return (
        np.asarray(bs_frames, dtype=np.float32),
        np.asarray(pose_frames, dtype=np.float32),
        POSE_COLUMNS,
    )


def iter_sources(args) -> list[tuple[str, bytes]]:
    """Return [(video_id, raw_csv_bytes)] from either a zip or a directory."""
    items: list[tuple[str, bytes]] = []
    if args.zip:
        with zipfile.ZipFile(args.zip.resolve()) as zf:
            members = sorted(
                n for n in zf.namelist()
                if n.startswith(args.member_prefix) and n.lower().endswith(".csv")
            )
            if args.max_files > 0:
                members = members[: args.max_files]
            for name in members:
                items.append((Path(name).stem, zf.read(name)))
    else:
        paths = sorted(args.csv_dir.resolve().glob("*.csv"))
        if args.max_files > 0:
            paths = paths[: args.max_files]
        for path in paths:
            items.append((path.stem, path.read_bytes()))
    return items


def main() -> None:
    args = parse_args()
    source_name = str(args.zip or args.csv_dir)
    sources = iter_sources(args)
    print(f"[native-kf] {len(sources)} raw CSVs from {source_name}")

    # --- epsilon sweep mode: load a sample, print keyframe stats, exit ---
    if args.sweep_only:
        seqs = []
        for _vid, raw in sources[:150]:
            clip = load_clip(raw)
            if clip and clip[0].shape[0] >= 4:
                seqs.append(clip[0])
        frames = np.array([s.shape[0] for s in seqs])
        print(f"[sweep] clips={len(seqs)} mean_frames={frames.mean():.0f} "
              f"min={frames.min()} max={frames.max()}")
        for eps in [0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0, 1.2]:
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

    videos = kf_total = anchor_only = 0
    histogram: dict[str, int] = {}
    for index, (video_id, raw) in enumerate(sources, start=1):
        clip = load_clip(raw)
        if clip is None:
            continue
        blendshapes, pose, pose_names = clip
        num_frames = int(blendshapes.shape[0])
        annotation = pick_keyframes(
            blendshapes, method=args.method,
            epsilon=args.epsilon, max_keyframes=args.max_keyframes,
        )

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
            "source_representation": "arkit52",
            "native_fps": args.native_fps,
            "target_fps": args.native_fps,      # no downsampling
            "num_frames": num_frames,
            "duration_sec": round(num_frames / args.native_fps, 3),
            "blendshape_names": list(ARKIT_BLENDSHAPE_NAMES),
            "pose_names": pose_names,
            "head_eye_pose": [[round(float(v), 6) for v in pose[i]] for i in range(num_frames)],
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
        if index % 200 == 0:
            print(f"[native-kf] {index}/{len(sources)}  videos={videos}")

    summary = {
        "source": source_name,
        "dataset": args.dataset,
        "output_dir": str(out_dir),
        "native_fps": args.native_fps,
        "method": args.method,
        "epsilon": float(args.epsilon),
        "videos": videos,
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
