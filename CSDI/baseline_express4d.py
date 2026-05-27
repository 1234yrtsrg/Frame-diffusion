import numpy as np


def linear_interpolation(start, end, num_middle=10):
    start = np.asarray(start, dtype=np.float32)
    end = np.asarray(end, dtype=np.float32)
    weights = np.linspace(0.0, 1.0, num_middle + 2, dtype=np.float32)[1:-1]
    return ((1.0 - weights[:, None]) * start + weights[:, None] * end).astype(np.float32)


def cubic_interpolation(start, end, num_middle=10):
    start = np.asarray(start, dtype=np.float32)
    end = np.asarray(end, dtype=np.float32)
    x = np.array([0.0, float(num_middle + 1)], dtype=np.float32)
    xi = np.arange(1, num_middle + 1, dtype=np.float32)
    try:
        from scipy.interpolate import CubicSpline

        spline = CubicSpline(x, np.stack([start, end], axis=0), axis=0, bc_type="clamped")
        return spline(xi).astype(np.float32)
    except Exception:
        t = xi / float(num_middle + 1)
        smooth = t * t * (3.0 - 2.0 * t)
        return ((1.0 - smooth[:, None]) * start + smooth[:, None] * end).astype(np.float32)
