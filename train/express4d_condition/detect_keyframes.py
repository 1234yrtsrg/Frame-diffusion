import argparse
import json
from pathlib import Path
import sys

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
CSDI_DIR = REPO_ROOT / "CSDI"
if str(CSDI_DIR) not in sys.path:
    sys.path.insert(0, str(CSDI_DIR))

from dataset_express4d import (  # noqa: E402
    ARKIT_52_NAMES,
    ARKIT_FROM_EXPRESS4D,
    EXPRESS4D_61_NAMES,
    _read_list,
    _resolve_entry,
    normalize_name,
    resolve_dataset_root,
)

try:
    from scipy.signal import find_peaks as scipy_find_peaks
except Exception:  # pragma: no cover
    scipy_find_peaks = None

try:
    from scipy.signal import savgol_filter as scipy_savgol_filter
except Exception:  # pragma: no cover
    scipy_savgol_filter = None


def load_express4d_sequence(path, clamp=True, clamp_min=0.0, clamp_max=1.0):
    """Load Express4D .npy/.csv and return ARKit-52 sequence with shape [T, 52]."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Input file not found: {path}")

    suffix = path.suffix.lower()
    if suffix == ".npy":
        data = _read_express4d_npy(path)
    elif suffix == ".csv":
        data = _read_express4d_csv(path)
    else:
        raise ValueError(f"Unsupported Express4D input format: {suffix}")

    data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    if clamp:
        data = np.clip(data, clamp_min, clamp_max)
    if data.ndim != 2 or data.shape[1] != 52:
        raise ValueError(f"Expected Express4D ARKit data shape [T,52], got {data.shape} from {path}")
    return data


def load_blendshape_sequence(path):
    """Backward-compatible alias; this detector is Express4D-specific."""
    return load_express4d_sequence(path)


def _read_express4d_npy(path):
    data = np.asarray(np.load(path))
    if data.ndim != 2:
        raise ValueError(f"{path} must be 2D, got shape {data.shape}")
    data = _ensure_time_major(data)
    if data.shape[1] == 52:
        return data
    if data.shape[1] >= 61:
        return data[:, :61][:, ARKIT_FROM_EXPRESS4D]
    raise ValueError(f"{path} must contain 52 ARKit or 61 Express4D features, got {data.shape}")


def _read_express4d_csv(path):
    header = _read_csv_header(path)
    normalized = {normalize_name(name): index for index, name in enumerate(header)}

    arkit_columns = _named_columns(normalized, ARKIT_52_NAMES)
    if arkit_columns is not None:
        data = np.genfromtxt(path, delimiter=",", skip_header=1, usecols=arkit_columns, dtype=np.float32, ndmin=2)
        return data

    express4d_columns = _named_columns(normalized, EXPRESS4D_61_NAMES)
    if express4d_columns is not None:
        data_61 = np.genfromtxt(path, delimiter=",", skip_header=1, usecols=express4d_columns, dtype=np.float32, ndmin=2)
        return data_61[:, ARKIT_FROM_EXPRESS4D]

    raw = np.genfromtxt(path, delimiter=",", dtype=np.float32, ndmin=2, filling_values=np.nan)
    if raw.size == 0:
        raise ValueError(f"CSV file has no numeric data: {path}")
    if raw.shape[0] > 0 and np.isnan(raw[0]).all():
        raw = raw[1:]
    raw = _ensure_time_major(raw)
    if raw.shape[1] == 52:
        return raw
    if raw.shape[1] >= 61:
        return raw[:, :61][:, ARKIT_FROM_EXPRESS4D]
    raise ValueError(f"{path} must contain named ARKit/Express4D columns or at least 61 numeric columns")


def _read_csv_header(path):
    lines = Path(path).read_text(encoding="utf-8-sig").splitlines()
    if not lines:
        raise ValueError(f"CSV file is empty: {path}")
    first_line = lines[0]
    return [item.strip() for item in first_line.split(",")]


def _named_columns(normalized_header, names):
    columns = []
    for name in names:
        key = normalize_name(name)
        if key not in normalized_header:
            return None
        columns.append(normalized_header[key])
    return columns


def _ensure_time_major(array):
    if array.shape[0] in (52, 61) and array.shape[1] not in (52, 61):
        return array.T
    return array


def fill_nan_frames(X):
    """Fill NaNs by linear interpolation along time; fall back to zeros."""
    X = np.asarray(X, dtype=np.float32)
    if not np.isnan(X).any():
        return X

    filled = X.copy()
    t = np.arange(len(filled), dtype=np.float32)
    for d in range(filled.shape[1]):
        col = filled[:, d]
        valid = np.isfinite(col)
        if valid.any():
            col[~valid] = np.interp(t[~valid], t[valid], col[valid])
        else:
            col[:] = 0.0
    return filled


def preprocess_sequence(X, smooth=True):
    X = fill_nan_frames(X)
    if smooth:
        X = smooth_sequence(X)
    return X


def smooth_sequence(X):
    """Temporal smoothing with Savitzky-Golay when available, else moving average."""
    X = np.asarray(X, dtype=np.float32)
    T = len(X)
    if T < 3:
        return X

    if scipy_savgol_filter is not None:
        window = min(11, T if (T % 2 == 1) else T - 1)
        if window >= 3:
            polyorder = min(2, window - 1)
            if polyorder < window:
                return scipy_savgol_filter(X, window_length=window, polyorder=polyorder, axis=0, mode="interp")

    window = min(5, T)
    if window < 3:
        return X
    if window % 2 == 0:
        window -= 1
    if window < 3:
        return X

    pad = window // 2
    padded = np.pad(X, ((pad, pad), (0, 0)), mode="edge")
    kernel = np.ones(window, dtype=np.float32) / float(window)
    smoothed = np.stack(
        [np.convolve(padded[:, d], kernel, mode="valid") for d in range(X.shape[1])],
        axis=1,
    )
    return smoothed.astype(np.float32, copy=False)


def compute_motion_score(X, lam=0.5):
    """
    Motion score = velocity + lam * acceleration.

    Velocity and acceleration are computed per frame from consecutive blendshape
    differences. Endpoints use one-sided approximations so the score is defined
    for every frame.
    """
    X = np.asarray(X, dtype=np.float32)
    T = len(X)
    if T == 0:
        return np.zeros((0,), dtype=np.float32)
    if T == 1:
        return np.zeros((1,), dtype=np.float32)

    velocity = np.zeros(T, dtype=np.float32)
    velocity[1:] = np.linalg.norm(X[1:] - X[:-1], axis=1)
    velocity[0] = velocity[1]

    acceleration = np.zeros(T, dtype=np.float32)
    if T >= 3:
        acceleration[1:-1] = np.linalg.norm(X[2:] - 2.0 * X[1:-1] + X[:-2], axis=1)
        acceleration[0] = acceleration[1]
        acceleration[-1] = acceleration[-2]

    return velocity + float(lam) * acceleration


def detect_motion_peaks(score, min_gap=6, prominence_percentile=75):
    """Detect local motion peaks, using scipy if present and a simple fallback otherwise."""
    score = np.asarray(score, dtype=np.float32)
    if len(score) < 3:
        return []

    prominence = float(np.percentile(score, prominence_percentile))
    prominence = max(prominence, 0.0)

    if scipy_find_peaks is not None:
        peaks, _ = scipy_find_peaks(score, distance=min_gap, prominence=prominence)
        return peaks.astype(int).tolist()

    return _fallback_find_peaks(score, min_gap=min_gap, prominence=prominence)


def _fallback_find_peaks(score, min_gap=6, prominence=0.0):
    """Minimal peak detector when scipy is unavailable."""
    score = np.asarray(score, dtype=np.float32)
    candidates = []
    n = len(score)
    for i in range(1, n - 1):
        if score[i] >= score[i - 1] and score[i] > score[i + 1]:
            left = score[max(0, i - min_gap):i]
            right = score[i + 1:min(n, i + min_gap + 1)]
            left_min = float(left.min()) if len(left) else float(score[i])
            right_min = float(right.min()) if len(right) else float(score[i])
            local_prominence = score[i] - max(left_min, right_min)
            if local_prominence >= prominence:
                candidates.append(i)

    candidates.sort(key=lambda idx: score[idx], reverse=True)
    selected = []
    for idx in candidates:
        if all(abs(idx - other) >= min_gap for other in selected):
            selected.append(idx)
    return sorted(selected)


def _segment_linear_interp(X, i, j):
    """Linearly interpolate the interval [i, j] inclusive."""
    if j <= i:
        return X[i : i + 1]
    alpha = np.linspace(0.0, 1.0, j - i + 1, dtype=np.float32)[:, None]
    return (1.0 - alpha) * X[i] + alpha * X[j]


def rdp_keyframes(X, eps=0.05):
    """Recursive RDP-style keyframe extraction based on reconstruction error."""
    X = np.asarray(X, dtype=np.float32)
    if len(X) == 0:
        return []
    if len(X) == 1:
        return [0]

    keyframes = {0, len(X) - 1}

    def recurse(i, j):
        if j <= i + 1:
            keyframes.add(i)
            keyframes.add(j)
            return

        interp = _segment_linear_interp(X, i, j)
        segment = X[i : j + 1]
        errors = np.linalg.norm(segment - interp, axis=1)
        if len(errors) <= 2:
            keyframes.add(i)
            keyframes.add(j)
            return

        inner_errors = errors[1:-1]
        rel_idx = int(np.argmax(inner_errors)) + 1
        max_error = float(inner_errors[rel_idx - 1])
        if max_error > eps:
            k = i + rel_idx
            keyframes.add(k)
            recurse(i, k)
            recurse(k, j)
        else:
            keyframes.add(i)
            keyframes.add(j)

    recurse(0, len(X) - 1)
    return sorted(keyframes)


def merge_keyframes(keyframes, motion_score, min_gap=6):
    """Merge candidate keyframes with distance suppression, keeping endpoints."""
    motion_score = np.asarray(motion_score, dtype=np.float32)
    if len(motion_score) == 0:
        return []
    if len(motion_score) == 1:
        return [0]

    n = len(motion_score)
    endpoints = {0, n - 1}
    candidates = sorted(set(int(k) for k in keyframes if 0 <= int(k) < n))
    others = [k for k in candidates if k not in endpoints]
    others.sort(key=lambda k: (motion_score[k], -k), reverse=True)

    selected = [0, n - 1]
    for k in others:
        if all(abs(k - s) >= min_gap for s in selected):
            selected.append(k)

    return sorted(set(selected))


def detect_blendshape_keyframes(
    X,
    eps=0.05,
    min_gap=6,
    lam=0.5,
    prominence_percentile=75,
    smooth=True,
):
    """
    Detect keyframes from a blendshape sequence.

    Returns:
        keyframes: sorted list of detected indices
        motion_score: per-frame motion score
    """
    X = np.asarray(X, dtype=np.float32)
    if X.ndim != 2:
        raise ValueError(f"X must have shape [T, D], got {X.shape}")
    if len(X) == 0:
        return [], np.zeros((0,), dtype=np.float32)
    if min_gap < 1:
        raise ValueError("min_gap must be at least 1")
    if not (0.0 <= prominence_percentile <= 100.0):
        raise ValueError("prominence_percentile must be in [0, 100]")

    X = preprocess_sequence(X, smooth=smooth)

    motion_score = compute_motion_score(X, lam=lam)
    motion_peaks = detect_motion_peaks(
        motion_score,
        min_gap=min_gap,
        prominence_percentile=prominence_percentile,
    )
    rdp_frames = rdp_keyframes(X, eps=eps)

    merged = merge_keyframes(
        [0, len(X) - 1] + motion_peaks + rdp_frames,
        motion_score,
        min_gap=min_gap,
    )
    return merged, motion_score


def _plot_keyframes(motion_score, keyframes, output_path):
    import matplotlib.pyplot as plt

    motion_score = np.asarray(motion_score, dtype=np.float32)
    keyframes = sorted(set(int(k) for k in keyframes))
    fig, ax = plt.subplots(figsize=(14, 4))
    ax.plot(motion_score, linewidth=1.4, color="steelblue", label="motion_score")
    for idx in keyframes:
        ax.axvline(idx, color="tomato", alpha=0.35, linewidth=1)
    ax.scatter(keyframes, motion_score[keyframes], color="tomato", s=18, zorder=3, label="keyframes")
    ax.set_xlabel("frame")
    ax.set_ylabel("score")
    ax.set_title("Blendshape keyframe detection")
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def detect_express4d_file(
    input_path,
    eps=0.05,
    min_gap=6,
    lam=0.5,
    prominence_percentile=75,
    smooth=True,
    clamp=True,
    clamp_min=0.0,
    clamp_max=1.0,
):
    X = load_express4d_sequence(
        input_path,
        clamp=clamp,
        clamp_min=clamp_min,
        clamp_max=clamp_max,
    )
    if len(X) == 0:
        raise ValueError(f"Input sequence is empty: {input_path}")
    if min_gap < 1:
        raise ValueError("min_gap must be at least 1")
    if not (0.0 <= prominence_percentile <= 100.0):
        raise ValueError("prominence_percentile must be in [0, 100]")

    X_proc = preprocess_sequence(X, smooth=smooth)
    motion_score = compute_motion_score(X_proc, lam=lam)
    motion_peaks = detect_motion_peaks(
        motion_score,
        min_gap=min_gap,
        prominence_percentile=prominence_percentile,
    )
    rdp_frames = rdp_keyframes(X_proc, eps=eps)
    keyframes = merge_keyframes(
        [0, len(X_proc) - 1] + motion_peaks + rdp_frames,
        motion_score,
        min_gap=min_gap,
    )
    payload = {
        "input": str(Path(input_path).resolve()),
        "num_frames": int(len(X)),
        "num_features": int(X.shape[1]),
        "feature_space": "ARKit-52",
        "source_format": "Express4D",
        "keyframes": [int(k) for k in keyframes],
        "motion_peaks": [int(k) for k in motion_peaks],
        "rdp_keyframes": [int(k) for k in rdp_frames],
        "params": {
            "eps": eps,
            "min_gap": min_gap,
            "lam": lam,
            "prominence_percentile": prominence_percentile,
            "smooth": smooth,
            "clamp": clamp,
            "clamp_min": clamp_min,
            "clamp_max": clamp_max,
        },
    }
    return payload, motion_score


def _write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def _resolve_express4d_entries(list_path, root, data_dir, use_npy_first=True):
    root = resolve_dataset_root(root)
    data_dir = root / data_dir
    return [
        _resolve_entry(entry, root, data_dir, use_npy_first=use_npy_first)
        for entry in _read_list(Path(list_path))
    ]


def main():
    parser = argparse.ArgumentParser(description="Detect keyframes in Express4D blendshape sequences")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--input", help="Path to one Express4D .npy or .csv sequence")
    source.add_argument("--list", help="Express4D split list, e.g. dataset/train.txt")
    parser.add_argument("--root", default="dataset", help="Express4D dataset root for --list")
    parser.add_argument("--data_dir", default=".", help="Data directory under --root for --list")
    parser.add_argument("--output", required=True, help="Output keyframes .json")
    parser.add_argument("--output_dir", default=None, help="Optional per-sequence JSON directory for --list")
    parser.add_argument("--eps", type=float, default=0.05)
    parser.add_argument("--min_gap", type=int, default=6)
    parser.add_argument("--lam", type=float, default=0.5)
    parser.add_argument("--prominence_percentile", type=float, default=75)
    parser.add_argument("--no-smooth", dest="smooth", action="store_false")
    parser.set_defaults(smooth=True)
    parser.add_argument("--no-clamp", dest="clamp", action="store_false")
    parser.set_defaults(clamp=True)
    parser.add_argument("--clamp_min", type=float, default=0.0)
    parser.add_argument("--clamp_max", type=float, default=1.0)
    parser.add_argument("--use_csv_first", action="store_true", help="Prefer .csv over .npy when resolving --list entries")
    parser.add_argument("--motion_score_out", default=None, help="Optional .npy path for motion score")
    parser.add_argument("--motion_score_dir", default=None, help="Optional motion-score directory for --list")
    parser.add_argument("--plot", default=None, help="Optional .png path for a score plot")
    parser.add_argument("--plot_dir", default=None, help="Optional plot directory for --list")
    args = parser.parse_args()

    common_kwargs = dict(
        eps=args.eps,
        min_gap=args.min_gap,
        lam=args.lam,
        prominence_percentile=args.prominence_percentile,
        smooth=args.smooth,
        clamp=args.clamp,
        clamp_min=args.clamp_min,
        clamp_max=args.clamp_max,
    )

    if args.input:
        payload, motion_score = detect_express4d_file(args.input, **common_kwargs)
        _write_json(args.output, payload)

        if args.motion_score_out:
            motion_path = Path(args.motion_score_out)
            motion_path.parent.mkdir(parents=True, exist_ok=True)
            np.save(motion_path, motion_score)

        if args.plot:
            plot_path = Path(args.plot)
            plot_path.parent.mkdir(parents=True, exist_ok=True)
            _plot_keyframes(motion_score, payload["keyframes"], plot_path)

        print(json.dumps({"keyframes": payload["keyframes"], "output": str(args.output)}, indent=2))
        return

    entries = _resolve_express4d_entries(
        args.list,
        root=args.root,
        data_dir=args.data_dir,
        use_npy_first=not args.use_csv_first,
    )
    sequence_payloads = []
    output_dir = Path(args.output_dir) if args.output_dir else None
    motion_score_dir = Path(args.motion_score_dir) if args.motion_score_dir else None
    plot_dir = Path(args.plot_dir) if args.plot_dir else None

    for path in entries:
        payload, motion_score = detect_express4d_file(path, **common_kwargs)
        payload["sequence_name"] = Path(path).stem
        sequence_payloads.append(payload)

        if output_dir is not None:
            _write_json(output_dir / f"{Path(path).stem}.json", payload)
        if motion_score_dir is not None:
            motion_score_dir.mkdir(parents=True, exist_ok=True)
            np.save(motion_score_dir / f"{Path(path).stem}_motion_score.npy", motion_score)
        if plot_dir is not None:
            plot_dir.mkdir(parents=True, exist_ok=True)
            _plot_keyframes(motion_score, payload["keyframes"], plot_dir / f"{Path(path).stem}.png")

    aggregate = {
        "list": str(Path(args.list).resolve()),
        "root": str(resolve_dataset_root(args.root)),
        "data_dir": args.data_dir,
        "num_sequences": len(sequence_payloads),
        "params": common_kwargs,
        "sequences": sequence_payloads,
    }
    _write_json(args.output, aggregate)
    print(json.dumps({"num_sequences": len(sequence_payloads), "output": str(args.output)}, indent=2))


if __name__ == "__main__":
    main()
