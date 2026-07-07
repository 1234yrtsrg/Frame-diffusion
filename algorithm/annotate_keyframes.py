"""Standalone keyframe-annotation runner (no project dependencies).

Reads per-clip JSON files that contain a ``frames`` list of ARKit-52
blendshape vectors, runs the RDP curve-simplification keyframe selector from
``keyframe_annotation.py``, and writes ``keyframe_indices`` back into each JSON
plus an aggregate summary.

Expected per-clip JSON shape (only these fields are required)::

    {
      "video_id": "MySlate_223_iPhone_cal",
      "num_frames": 43,
      "frames": [
        {"frame_index": 0, "blendshapes": [<52 floats>]},
        ...
      ]
    }

Usage::

    python annotate_keyframes.py --root <folder_with_json> --epsilon 0.4

The script recurses into sub-directories, so a layout like
``express4d/0000/clip.json`` works directly.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from keyframe_annotation import pick_keyframes


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path, help="Folder of per-clip JSON files (recurses).")
    parser.add_argument("--method", choices=["rdp", "peak"], default="rdp")
    parser.add_argument("--epsilon", type=float, default=0.4, help="RDP tolerance (L2 linear-recon error).")
    parser.add_argument("--max-keyframes", type=int, default=16)
    parser.add_argument("--summary-output", type=Path, default=None)
    parser.add_argument("--no-overwrite", action="store_true", help="Skip clips that already have keyframe_indices.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    json_files = sorted(args.root.rglob("*.json"))
    # Ignore any summary file we may have written previously.
    json_files = [p for p in json_files if not p.name.startswith("keyframe_summary")]
    print(f"[annotate] found {len(json_files)} JSON files under {args.root}")

    videos = 0
    kf_total = 0
    interior_total = 0
    anchor_only = 0
    histogram: dict[str, int] = {}
    errors: list[dict] = []

    for index, path in enumerate(json_files, start=1):
        try:
            document = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            errors.append({"path": str(path), "error": str(exc)})
            continue

        if args.no_overwrite and document.get("keyframe_indices") is not None:
            continue

        frames = document.get("frames") or []
        if len(frames) < 1:
            errors.append({"path": str(path), "error": "no frames"})
            continue

        blendshapes = np.array([f["blendshapes"] for f in frames], dtype=np.float32)
        annotation = pick_keyframes(
            blendshapes,
            method=args.method,
            epsilon=args.epsilon,
            max_keyframes=args.max_keyframes,
        )

        document["keyframe_indices"] = annotation.as_list()
        document["keyframe_annotation"] = {
            "method": annotation.method,
            "selector": args.method,
            "epsilon": float(args.epsilon),
            "max_keyframes": int(args.max_keyframes),
            "num_keyframes": int(annotation.indices.size),
            "num_interior": int(annotation.peak_indices.size),
            "max_error": float(annotation.max_error),
        }
        path.write_text(json.dumps(document, ensure_ascii=False), encoding="utf-8")

        videos += 1
        num_kf = int(annotation.indices.size)
        kf_total += num_kf
        interior_total += int(annotation.peak_indices.size)
        if annotation.peak_indices.size == 0:
            anchor_only += 1
        key = str(num_kf) if num_kf <= 10 else "10+"
        histogram[key] = histogram.get(key, 0) + 1

        if index % 500 == 0:
            print(f"[annotate] {index}/{len(json_files)}")

    summary = {
        "root": str(args.root),
        "method": args.method,
        "epsilon": float(args.epsilon),
        "videos": videos,
        "keyframes_total": kf_total,
        "avg_keyframes": round(kf_total / videos, 2) if videos else 0,
        "with_interior_keyframes": videos - anchor_only,
        "anchor_only": anchor_only,
        "histogram": histogram,
        "errors": errors[:50],
        "num_errors": len(errors),
    }
    print("=== SUMMARY ===")
    print(json.dumps(summary, ensure_ascii=False, indent=2))

    out = args.summary_output or (args.root / "keyframe_summary.json")
    out.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[annotate] wrote summary to {out}")


if __name__ == "__main__":
    main()
