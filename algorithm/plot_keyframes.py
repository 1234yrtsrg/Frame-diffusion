"""Visualize RDP keyframe selection on a single clip JSON.

Plots the per-frame expression energy (L2 norm of the 52 blendshapes) over
time and marks the selected keyframes, plus the piecewise-linear
reconstruction implied by those keyframes. Useful for eyeballing whether the
chosen ``epsilon`` produces sensible, sparse, semantic keyframes.

Usage::

    python plot_keyframes.py --json <clip.json> --out keyframes.png
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from keyframe_annotation import pick_keyframes  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--epsilon", type=float, default=0.4)
    parser.add_argument("--max-keyframes", type=int, default=16)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    document = json.loads(args.json.read_text(encoding="utf-8"))
    frames = document.get("frames") or []
    blendshapes = np.array([f["blendshapes"] for f in frames], dtype=np.float32)

    annotation = pick_keyframes(
        blendshapes, method="rdp", epsilon=args.epsilon, max_keyframes=args.max_keyframes
    )
    keyframes = annotation.indices
    energy = np.linalg.norm(blendshapes, axis=1)
    timeline = np.arange(len(energy))

    # Piecewise-linear reconstruction from keyframes only (per channel), then
    # take its energy so we can overlay it on the dense energy curve.
    recon = np.zeros_like(blendshapes)
    for left, right in zip(keyframes[:-1], keyframes[1:]):
        span = right - left
        for offset in range(span + 1):
            weight = offset / span if span else 0.0
            recon[left + offset] = (
                blendshapes[left] * (1 - weight) + blendshapes[right] * weight
            )
    recon_energy = np.linalg.norm(recon, axis=1)

    fig, ax = plt.subplots(figsize=(11, 4.5))
    ax.plot(timeline, energy, color="#1f77b4", lw=1.8, label="dense expression energy")
    ax.plot(
        timeline,
        recon_energy,
        color="#ff7f0e",
        lw=1.4,
        ls="--",
        label="linear reconstruction from keyframes",
    )
    ax.scatter(
        keyframes,
        energy[keyframes],
        color="#d62728",
        zorder=5,
        s=70,
        label=f"keyframes ({len(keyframes)})",
    )
    for k in keyframes:
        ax.axvline(k, color="#d62728", alpha=0.15, lw=1)

    text = (document.get("text") or "").strip()
    title = f"{document.get('video_id', args.json.stem)}"
    if text:
        title += f"  —  “{text[:70]}”"
    ax.set_title(title, fontsize=11)
    ax.set_xlabel("frame")
    ax.set_ylabel("L2 expression energy")
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(alpha=0.2)
    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=130)
    print(f"wrote {args.out}  keyframes={keyframes.tolist()}  max_error={annotation.max_error:.3f}")


if __name__ == "__main__":
    main()
