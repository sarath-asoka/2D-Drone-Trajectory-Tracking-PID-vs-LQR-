"""
Performance metrics.

Tracking metrics use the Euclidean position error

    e_k = || p_ref,k - p_k ||_2        (at the N+1 state times t_k = k dt)

and the *applied* (saturated) control input u_k, held constant over
[t_k, t_{k+1}):

    RMSE            sqrt( mean_k e_k^2 )
    Max error       max_k e_k
    Mean error      mean_k e_k
    IAE             integral |e| dt          (trapezoidal rule on e_k)
    Control effort  integral ||u||^2 dt = dt * sum_k ||u_k||^2   (exact for ZOH)
    Max input       max_k ||u_k||_2
    Saturation      fraction of steps where any axis sits at the actuator limit

Disturbance metrics compare the disturbed run with a *nominal* run of the
same controller (identical setup, gust switched off):

    deviation  d_k = || p_k(disturbed) - p_k(nominal) ||_2

Because the closed loop is linear while no actuator saturates, d_k is
exactly the response to the gust alone, independent of how large the normal
tracking error happens to be at that point of the trajectory.

    Peak deviation   max_k d_k                    (for t_k >= t_start)
    Recovery time    first time after the gust ends from which d_k stays
                     <= recovery_tolerance until the end of the run, minus
                     t_end (must hold for at least recovery_hold_time).
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np

from config import DisturbanceConfig, MetricsConfig
from simulator import SimulationResult


def trapezoid(y: np.ndarray, dt: float) -> float:
    """Trapezoidal integral of uniformly sampled y (written out explicitly)."""
    if len(y) < 2:
        return 0.0
    return float(dt * (np.sum(y) - 0.5 * (y[0] + y[-1])))


def disturbance_deviation(disturbed: SimulationResult, nominal: SimulationResult) -> np.ndarray:
    """d_k = ||p_disturbed - p_nominal|| at every state time."""
    if disturbed.t.shape != nominal.t.shape:
        raise ValueError("disturbed and nominal runs must share the same time grid")
    return np.linalg.norm(disturbed.states[:, :2] - nominal.states[:, :2], axis=1)


def recovery_time(
    t: np.ndarray, deviation: np.ndarray, end_time: float, tolerance: float, hold_time: float
) -> float:
    """
    Settling-style recovery time after the gust ends.

    Returns:
        t_rec - end_time, where t_rec is the earliest time >= end_time from
        which deviation <= tolerance for the rest of the run. Returns 0.0 if
        the deviation is already within tolerance at end_time, and inf if it
        never settles (or settles less than ``hold_time`` before the end).
    """
    idx_after = np.nonzero(t >= end_time - 1e-9)[0]
    if len(idx_after) == 0:
        return float("inf")
    bad = idx_after[deviation[idx_after] > tolerance]
    if len(bad) == 0:
        return 0.0
    last_bad = int(bad[-1])
    if last_bad + 1 >= len(t):
        return float("inf")
    t_rec = float(t[last_bad + 1])
    if t[-1] - t_rec < hold_time:
        return float("inf")
    return t_rec - end_time


def compute_metrics(
    result: SimulationResult,
    nominal: Optional[SimulationResult],
    dist: DisturbanceConfig,
    mcfg: MetricsConfig,
    dt: float,
    accel_limit: float,
) -> Dict[str, float]:
    """
    Compute every metric for one controller.

    Args:
        result: Run under test (with the gust, if enabled).
        nominal: Same controller without the gust (None if the gust is disabled).
        dist, mcfg: Disturbance and metric settings.
        dt: Timestep [s].
        accel_limit: Per-axis actuator limit [m/s^2].
    """
    e = result.position_error
    u_norm = np.linalg.norm(result.inputs, axis=1)
    at_limit = np.any(np.abs(result.inputs) >= accel_limit - 1e-9, axis=1)

    rec = float("nan")
    peak_dev = float("nan")
    if dist.enabled and nominal is not None:
        dev = disturbance_deviation(result, nominal)
        win = result.t >= dist.start_time - 1e-9
        peak_dev = float(np.max(dev[win]))
        rec = recovery_time(result.t, dev, dist.end_time, mcfg.recovery_tolerance, mcfg.recovery_hold_time)

    return {
        "rmse": float(np.sqrt(np.mean(e**2))),
        "max_error": float(np.max(e)),
        "mean_error": float(np.mean(e)),
        "iae": trapezoid(e, dt),
        "control_effort": float(dt * np.sum(u_norm**2)),
        "max_input": float(np.max(u_norm)),
        "saturation_fraction": float(np.mean(at_limit)),
        "peak_disturbance_deviation": peak_dev,
        "recovery_time": rec,
    }


# (key, label, unit, format)
METRIC_ROWS: List[Tuple[str, str, str, str]] = [
    ("rmse", "Position RMSE", "m", "{:.4f}"),
    ("max_error", "Max position error", "m", "{:.4f}"),
    ("mean_error", "Mean position error", "m", "{:.4f}"),
    ("iae", "IAE", "m·s", "{:.4f}"),
    ("control_effort", "Control effort ∫‖u‖²dt", "m²/s³", "{:.2f}"),
    ("max_input", "Max control input ‖u‖", "m/s²", "{:.3f}"),
    ("saturation_fraction", "Steps at actuator limit", "frac", "{:.3f}"),
    ("peak_disturbance_deviation", "Peak gust deviation", "m", "{:.4f}"),
    ("recovery_time", "Recovery time after gust", "s", "{:.2f}"),
]


def _fmt(value: float, fmt: str) -> str:
    if value is None or np.isnan(value):
        return "n/a"
    if np.isinf(value):
        return "not recovered"
    return fmt.format(value)


def format_comparison_table(
    metrics: Dict[str, Dict[str, float]], title: Optional[str] = None
) -> str:
    """Return a plain-text comparison table (lower is better for every row)."""
    names = list(metrics.keys())
    label_w = max(len(f"{r[1]} [{r[2]}]") for r in METRIC_ROWS) + 3
    col_w = 15
    width = label_w + col_w * len(names)
    lines = []
    if title:
        lines.append(title)
    lines.append("=" * width)
    lines.append("Metric (lower is better)".ljust(label_w) + "".join(n.rjust(col_w) for n in names))
    lines.append("-" * width)
    for key, label, unit, fmt in METRIC_ROWS:
        row = f"{label} [{unit}]".ljust(label_w)
        row += "".join(_fmt(metrics[n].get(key, float("nan")), fmt).rjust(col_w) for n in names)
        lines.append(row)
    lines.append("=" * width)
    return "\n".join(lines)


def metrics_to_csv(metrics: Dict[str, Dict[str, float]]) -> str:
    """CSV text with one row per metric and one column per controller."""
    names = list(metrics.keys())
    out = ["metric,unit," + ",".join(names)]
    for key, _label, unit, _fmt_str in METRIC_ROWS:
        out.append(f"{key},{unit}," + ",".join(repr(float(metrics[n][key])) for n in names))
    return "\n".join(out) + "\n"
