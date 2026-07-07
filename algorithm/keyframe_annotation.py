"""Keyframe annotation for ARKit-52 facial-motion sequences.

Two selection strategies are available:

``peak`` (the original)
    Compute a per-frame expression "energy" signal and keep the frames where
    that signal locally peaks. Fast, but it only fires on energy *maxima*, so
    smooth ramps between two held expressions yield no interior keyframe and a
    clip collapses to its two endpoints.

``rdp`` (curve simplification, recommended)
    Treat the dense sequence as a polyline in 52-D blendshape space and keep
    the frames that a piecewise-*linear* reconstruction cannot reproduce within
    a tolerance ``epsilon``. This is the Ramer-Douglas-Peucker algorithm: at
    each step we add the frame whose value is furthest from the straight line
    interpolated between the current keyframes, recursing until every frame is
    within ``epsilon`` of the reconstruction. The selected frames are exactly
    "the frames you cannot drop without the linear in-between exceeding
    ``epsilon`` of reconstruction error" -- i.e. the meeting's "minimum-MSE
    under linear interpolation" criterion, in its classic greedy top-down form.

In both cases the first and last frame are always kept as anchors so the
inbetweening model has fixed boundary conditions.

This mirrors the way text-to-keyframe systems (GPT-Face etc.) describe a
clip: a small set of *salient* expressions ("smile", "frown", "jaw drop")
plus the rest pose at the ends. Inbetweening then has to recover the dense
trajectory between those salient frames.

Public surface:
    - :func:`pick_keyframes`           -- low-level numpy function (method switch)
    - :func:`pick_keyframes_rdp`       -- the RDP curve-simplification selector
    - :func:`annotate_sequence_file`   -- read JSON, update ``keyframe_indices``
    - :func:`annotate_dataset_root`    -- batch over ``data/json_dataset/``
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.signal import find_peaks


# Defaults are tuned for ARKit-52 at target_fps=10. A 4-second clip (≈40
# frames) typically yields 2–5 peaks with these settings; sequences shorter
# than ~3 frames just return the endpoints with no peak detection.
DEFAULT_METHOD = "rdp"           # "rdp" || "peak"
DEFAULT_ENERGY = "l2"            # "l2" || "l1" || "linf"
DEFAULT_MIN_DISTANCE = 3         # ≥0.3 s between peaks at 10 fps
DEFAULT_MIN_PROMINENCE = 0.05    # relative to global energy max (per-clip)
DEFAULT_REL_HEIGHT = 0.10        # required height above clip's own baseline
DEFAULT_MAX_KEYFRAMES = 16       # safety cap; reaction clips with constant
                                 # micro-expressions should not balloon

# RDP tolerance: a frame is kept as a keyframe when its blendshape vector
# deviates from the linear interpolation of its bracketing keyframes by more
# than this L2 distance. ARKit channels are in [0, 1] and a full sequence
# spans 52 of them, so the L2 of the residual accumulates across channels.
# Empirically on Express4D (10 fps, ~48-frame reaction clips) this default
# yields a median of ~8 keyframes per clip (≈1.6 / second) -- sparse, semantic
# keyframes rather than near-every-frame. Tune per dataset/fps: smaller => more
# keyframes / higher fidelity; larger => fewer, sparser keyframes.
DEFAULT_RDP_EPSILON = 0.4


@dataclass(slots=True)
class KeyframeAnnotation:
    """Result of :func:`pick_keyframes`."""

    indices: np.ndarray            # int64, sorted, contains 0 and T-1 if T>=2
    energy: np.ndarray             # float32 [T], per-frame energy
    peak_indices: np.ndarray       # int64, interior keyframes (no endpoints)
    method: str                    # "peak+anchor"|"anchor_only"|"rdp"
    max_error: float = 0.0         # rdp: worst linear-recon L2 error after fit

    def as_list(self) -> list[int]:
        return [int(value) for value in self.indices.tolist()]


def compute_energy(blendshapes: np.ndarray, mode: str = DEFAULT_ENERGY) -> np.ndarray:
    """Compute the per-frame expression energy.

    ARKit blendshapes are non-negative and rest at exactly 0, so the raw
    norm of the 52-vector is already a meaningful "how expressive is this
    frame" signal -- no neutral subtraction needed.
    """
    coefficients = np.asarray(blendshapes, dtype=np.float64)
    if coefficients.ndim != 2:
        raise ValueError(f"blendshapes must be 2-D, got shape {coefficients.shape}")
    if mode == "l1":
        return coefficients.sum(axis=1).astype(np.float32)
    if mode == "l2":
        return np.linalg.norm(coefficients, axis=1).astype(np.float32)
    if mode == "linf":
        return coefficients.max(axis=1).astype(np.float32)
    raise ValueError(f"Unknown energy mode {mode!r}; expected l1|l2|linf")


def _linear_recon_errors(
    sequence: np.ndarray, start: int, end: int
) -> np.ndarray:
    """L2 error of each interior frame vs the line from ``start`` to ``end``.

    Returns an array of length ``end - start - 1`` aligned to frames
    ``start+1 .. end-1``. Endpoints are excluded (their error is 0 by
    construction).
    """
    span = end - start
    if span < 2:
        return np.empty((0,), dtype=np.float64)
    # Interpolation weights for the interior frames only.
    weights = (np.arange(1, span) / span)[:, None]           # [span-1, 1]
    line = sequence[start] + weights * (sequence[end] - sequence[start])
    return np.linalg.norm(sequence[start + 1 : end] - line, axis=1)


def pick_keyframes_rdp(
    blendshapes: np.ndarray,
    *,
    epsilon: float = DEFAULT_RDP_EPSILON,
    max_keyframes: int = DEFAULT_MAX_KEYFRAMES,
    energy_mode: str = DEFAULT_ENERGY,
) -> KeyframeAnnotation:
    """Select keyframes by greedy curve simplification (Ramer-Douglas-Peucker).

    The sequence is a polyline in 52-D blendshape space. Starting from the two
    endpoints we repeatedly find the frame whose blendshape vector is furthest
    (L2) from the linear interpolation of its bracketing keyframes; if that
    distance exceeds ``epsilon`` the frame becomes a keyframe and its two
    sub-spans are recursed into. This keeps exactly the frames a linear
    in-between cannot reproduce within ``epsilon`` -- the meeting's
    "minimum reconstruction error under linear interpolation" criterion.

    ``max_keyframes`` caps the result: once reached, recursion stops even if
    some frames still exceed ``epsilon`` (the largest-error frames are kept
    because RDP always splits on the worst frame first).
    """
    blendshapes = np.asarray(blendshapes, dtype=np.float64)
    total_frames = blendshapes.shape[0]
    if total_frames == 0:
        raise ValueError("Cannot pick keyframes from an empty sequence")

    energy = compute_energy(blendshapes, energy_mode)

    if total_frames <= 2:
        return KeyframeAnnotation(
            indices=np.arange(total_frames, dtype=np.int64),
            energy=energy,
            peak_indices=np.array([], dtype=np.int64),
            method="anchor_only",
            max_error=0.0,
        )

    keep = [False] * total_frames
    keep[0] = True
    keep[total_frames - 1] = True

    # Iterative stack of (start, end) spans to avoid Python recursion limits on
    # long clips. We always process the span whose worst-frame error is largest
    # first so that ``max_keyframes`` keeps the most important frames.
    import heapq

    worst_overall = 0.0

    def span_candidate(start: int, end: int):
        errors = _linear_recon_errors(blendshapes, start, end)
        if errors.size == 0:
            return None
        local = int(errors.argmax())
        max_err = float(errors[local])
        return max_err, start, end, start + 1 + local

    heap: list[tuple] = []
    seed = span_candidate(0, total_frames - 1)
    if seed is not None:
        worst_overall = seed[0]
        # Negate error for a max-heap via Python's min-heap.
        heapq.heappush(heap, (-seed[0], seed[1], seed[2], seed[3]))

    num_kept = 2
    while heap and num_kept < max_keyframes:
        neg_err, start, end, split = heapq.heappop(heap)
        if -neg_err <= epsilon:
            break                       # every remaining span is within tol
        if keep[split]:
            continue
        keep[split] = True
        num_kept += 1
        for left, right in ((start, split), (split, end)):
            cand = span_candidate(left, right)
            if cand is not None:
                heapq.heappush(heap, (-cand[0], cand[1], cand[2], cand[3]))

    indices = np.flatnonzero(keep).astype(np.int64)

    # Report the worst residual error of the final reconstruction.
    residual = 0.0
    kept_indices = indices.tolist()
    for left, right in zip(kept_indices[:-1], kept_indices[1:]):
        errors = _linear_recon_errors(blendshapes, left, right)
        if errors.size:
            residual = max(residual, float(errors.max()))

    interior = indices[(indices != 0) & (indices != total_frames - 1)]
    return KeyframeAnnotation(
        indices=indices,
        energy=energy,
        peak_indices=interior.astype(np.int64),
        method="rdp" if interior.size > 0 else "anchor_only",
        max_error=residual,
    )


def pick_keyframes(
    blendshapes: np.ndarray,
    *,
    method: str = DEFAULT_METHOD,
    epsilon: float = DEFAULT_RDP_EPSILON,
    energy_mode: str = DEFAULT_ENERGY,
    min_distance: int = DEFAULT_MIN_DISTANCE,
    min_prominence: float = DEFAULT_MIN_PROMINENCE,
    rel_height: float = DEFAULT_REL_HEIGHT,
    max_keyframes: int = DEFAULT_MAX_KEYFRAMES,
) -> KeyframeAnnotation:
    """Pick keyframes from a dense blendshape sequence.

    ``method="rdp"`` (default) uses curve simplification by linear-reconstruction
    error; ``method="peak"`` uses the energy-peak detector. All other arguments
    are forwarded to the selected backend.
    """
    if method == "rdp":
        return pick_keyframes_rdp(
            blendshapes,
            epsilon=epsilon,
            max_keyframes=max_keyframes,
            energy_mode=energy_mode,
        )
    if method == "peak":
        return _pick_keyframes_peak(
            blendshapes,
            energy_mode=energy_mode,
            min_distance=min_distance,
            min_prominence=min_prominence,
            rel_height=rel_height,
            max_keyframes=max_keyframes,
        )
    raise ValueError(f"Unknown keyframe method {method!r}; expected rdp|peak")


def _pick_keyframes_peak(
    blendshapes: np.ndarray,
    *,
    energy_mode: str = DEFAULT_ENERGY,
    min_distance: int = DEFAULT_MIN_DISTANCE,
    min_prominence: float = DEFAULT_MIN_PROMINENCE,
    rel_height: float = DEFAULT_REL_HEIGHT,
    max_keyframes: int = DEFAULT_MAX_KEYFRAMES,
) -> KeyframeAnnotation:
    """Pick peak frames from a dense blendshape sequence.

    The first and last frames are always included; any local maxima of the
    energy signal that meet the prominence/distance thresholds are added in
    between. A peak that lands within ``min_distance`` of frame 0 or T-1 is
    dropped to avoid redundant anchors.

    Returns a :class:`KeyframeAnnotation` with sorted, deduplicated indices.
    """
    blendshapes = np.asarray(blendshapes, dtype=np.float64)
    total_frames = blendshapes.shape[0]
    if total_frames == 0:
        raise ValueError("Cannot pick keyframes from an empty sequence")

    energy = compute_energy(blendshapes, energy_mode)

    if total_frames == 1:
        return KeyframeAnnotation(
            indices=np.array([0], dtype=np.int64),
            energy=energy,
            peak_indices=np.array([], dtype=np.int64),
            method="anchor_only",
        )
    if total_frames == 2:
        return KeyframeAnnotation(
            indices=np.array([0, 1], dtype=np.int64),
            energy=energy,
            peak_indices=np.array([], dtype=np.int64),
            method="anchor_only",
        )

    # Prominence is expressed relative to the clip's own energy max so that
    # both very flat clips and very expressive clips use comparable cutoffs.
    energy_max = float(energy.max())
    if energy_max <= 1e-6:
        # All-neutral clip: anchors only.
        return KeyframeAnnotation(
            indices=np.array([0, total_frames - 1], dtype=np.int64),
            energy=energy,
            peak_indices=np.array([], dtype=np.int64),
            method="anchor_only",
        )

    abs_prominence = max(min_prominence * energy_max, 1e-4)
    abs_height = max(rel_height * energy_max, 1e-4)
    peaks, properties = find_peaks(
        energy,
        distance=max(1, min_distance),
        prominence=abs_prominence,
        height=abs_height,
    )

    # Drop peaks too close to either endpoint (the endpoint anchors already
    # capture that frame). ``min_distance`` here doubles as the boundary band.
    band = max(1, min_distance)
    peaks = np.array(
        [index for index in peaks if band <= index < total_frames - band],
        dtype=np.int64,
    )

    if peaks.size > max_keyframes - 2 and "prominences" in properties:
        # Keep the strongest ``max_keyframes - 2`` peaks if we have too many.
        prominences = properties["prominences"]
        kept_mask = np.array(
            [band <= index < total_frames - band for index in properties["peaks"]]
            if "peaks" in properties
            else [True] * len(prominences)
        )
        prominences_kept = prominences[: len(peaks)] if kept_mask.size != len(peaks) else prominences[kept_mask]
        order = np.argsort(-prominences_kept)
        peaks = np.sort(peaks[order[: max(0, max_keyframes - 2)]])

    indices = np.unique(
        np.concatenate(
            ([0], peaks.astype(np.int64), [total_frames - 1])
        )
    ).astype(np.int64)

    return KeyframeAnnotation(
        indices=indices,
        energy=energy,
        peak_indices=peaks,
        method="peak+anchor" if peaks.size > 0 else "anchor_only",
    )


def annotate_sequence_file(
    json_path: Path,
    *,
    method: str = DEFAULT_METHOD,
    epsilon: float = DEFAULT_RDP_EPSILON,
    energy_mode: str = DEFAULT_ENERGY,
    min_distance: int = DEFAULT_MIN_DISTANCE,
    min_prominence: float = DEFAULT_MIN_PROMINENCE,
    rel_height: float = DEFAULT_REL_HEIGHT,
    max_keyframes: int = DEFAULT_MAX_KEYFRAMES,
    overwrite: bool = True,
) -> dict:
    """Annotate a single per-video JSON in place and return a summary entry."""
    json_path = Path(json_path)
    document = json.loads(json_path.read_text(encoding="utf-8"))

    existing = document.get("keyframe_indices")
    if existing is not None and not overwrite:
        return {
            "video_id": document.get("video_id"),
            "num_frames": document.get("num_frames"),
            "num_keyframes": len(existing),
            "method": document.get("keyframe_annotation", {}).get("method", "unknown"),
            "skipped": True,
        }

    frames = document.get("frames", [])
    if not frames:
        return {
            "video_id": document.get("video_id"),
            "num_frames": 0,
            "num_keyframes": 0,
            "method": "empty",
            "skipped": True,
        }

    blendshapes = np.array(
        [frame["blendshapes"] for frame in frames], dtype=np.float32
    )
    annotation = pick_keyframes(
        blendshapes,
        method=method,
        epsilon=epsilon,
        energy_mode=energy_mode,
        min_distance=min_distance,
        min_prominence=min_prominence,
        rel_height=rel_height,
        max_keyframes=max_keyframes,
    )

    document["keyframe_indices"] = annotation.as_list()
    document["keyframe_annotation"] = {
        "method": annotation.method,
        "selector": method,
        "epsilon": float(epsilon),
        "energy_mode": energy_mode,
        "min_distance": int(min_distance),
        "min_prominence": float(min_prominence),
        "rel_height": float(rel_height),
        "max_keyframes": int(max_keyframes),
        "num_peaks": int(annotation.peak_indices.size),
        "max_error": float(annotation.max_error),
    }
    json_path.write_text(json.dumps(document, ensure_ascii=False), encoding="utf-8")

    return {
        "video_id": document.get("video_id"),
        "dataset": document.get("dataset"),
        "num_frames": int(blendshapes.shape[0]),
        "num_keyframes": int(annotation.indices.size),
        "num_peaks": int(annotation.peak_indices.size),
        "method": annotation.method,
        "max_error": float(annotation.max_error),
        "path": str(json_path),
    }


def annotate_dataset_root(
    dataset_root: Path,
    *,
    datasets: list[str] | None = None,
    method: str = DEFAULT_METHOD,
    epsilon: float = DEFAULT_RDP_EPSILON,
    energy_mode: str = DEFAULT_ENERGY,
    min_distance: int = DEFAULT_MIN_DISTANCE,
    min_prominence: float = DEFAULT_MIN_PROMINENCE,
    rel_height: float = DEFAULT_REL_HEIGHT,
    max_keyframes: int = DEFAULT_MAX_KEYFRAMES,
    overwrite: bool = True,
    progress_every: int = 1000,
    on_progress=None,
) -> dict:
    """Annotate every per-video JSON under ``data/json_dataset/<dataset>/``.

    ``datasets`` lets you restrict to e.g. ``["express4d", "mmhead"]``. The
    top-level ``index.json`` is left untouched -- per-video metadata is the
    source of truth for keyframe indices.
    """
    dataset_root = Path(dataset_root)
    if not dataset_root.exists():
        raise FileNotFoundError(f"Dataset root not found: {dataset_root}")

    if datasets is None:
        candidate_dirs = [child for child in sorted(dataset_root.iterdir()) if child.is_dir()]
    else:
        candidate_dirs = [dataset_root / name for name in datasets]

    summary: dict[str, dict] = {}
    total = 0
    processed = 0

    for dataset_dir in candidate_dirs:
        if not dataset_dir.is_dir():
            continue
        json_files = sorted(dataset_dir.rglob("*.json"))
        dataset_name = dataset_dir.name
        bucket = summary.setdefault(
            dataset_name,
            {
                "videos": 0,
                "keyframes_total": 0,
                "peaks_total": 0,
                "anchor_only": 0,
                "peak_plus_anchor": 0,
                "min_keyframes": None,
                "max_keyframes": None,
                "histogram": {},
            },
        )
        for json_path in json_files:
            try:
                entry = annotate_sequence_file(
                    json_path,
                    method=method,
                    epsilon=epsilon,
                    energy_mode=energy_mode,
                    min_distance=min_distance,
                    min_prominence=min_prominence,
                    rel_height=rel_height,
                    max_keyframes=max_keyframes,
                    overwrite=overwrite,
                )
            except (json.JSONDecodeError, KeyError, ValueError) as exc:
                bucket.setdefault("errors", []).append(
                    {"path": str(json_path), "error": str(exc)}
                )
                continue

            bucket["videos"] += 1
            num_kf = int(entry.get("num_keyframes", 0))
            num_pk = int(entry.get("num_peaks", 0))
            bucket["keyframes_total"] += num_kf
            bucket["peaks_total"] += num_pk
            method_name = entry.get("method", "")
            # "peak+anchor" (peak method) and "rdp" (curve simplification) both
            # mean interior keyframes were found; "anchor_only" means endpoints
            # only.
            if method_name in ("peak+anchor", "rdp"):
                bucket["peak_plus_anchor"] += 1
            else:
                bucket["anchor_only"] += 1
            current_min = bucket["min_keyframes"]
            current_max = bucket["max_keyframes"]
            bucket["min_keyframes"] = num_kf if current_min is None else min(current_min, num_kf)
            bucket["max_keyframes"] = num_kf if current_max is None else max(current_max, num_kf)
            histogram = bucket["histogram"]
            key = str(num_kf if num_kf <= 10 else "10+")
            histogram[key] = histogram.get(key, 0) + 1

            total += 1
            processed += 1
            if progress_every and on_progress is not None and processed % progress_every == 0:
                on_progress(processed, dataset_name, json_path)

    return {
        "dataset_root": str(dataset_root),
        "total_videos_annotated": total,
        "datasets": summary,
        "params": {
            "energy_mode": energy_mode,
            "min_distance": int(min_distance),
            "min_prominence": float(min_prominence),
            "rel_height": float(rel_height),
            "max_keyframes": int(max_keyframes),
        },
    }
