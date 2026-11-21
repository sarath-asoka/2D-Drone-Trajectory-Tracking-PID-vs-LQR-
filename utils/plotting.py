"""
Matplotlib plots and optional animation.

Every time-domain plot shades the disturbance interval so the gust is
easy to locate. All figures are saved as PNG in the results directory.
"""

from __future__ import annotations

import os
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation

from config import DisturbanceConfig, MetricsConfig, OutputConfig
from simulator import SimulationResult

# Consistent, colour-blind-friendly styling across every figure.
STYLE: Dict[str, Dict[str, object]] = {
    "Reference": {"color": "#222222", "linestyle": "--", "linewidth": 1.6},
    "PID": {"color": "#D55E00", "linestyle": "-", "linewidth": 1.8},
    "LQR": {"color": "#0072B2", "linestyle": "-", "linewidth": 1.8},
}
DIST_COLOR = "#E69F00"


def _style(name: str) -> Dict[str, object]:
    return STYLE.get(name, {"linewidth": 1.6})


def _shade_disturbance(ax: plt.Axes, dist: DisturbanceConfig, label: bool = True) -> None:
    """Shade the gust interval on a time axis."""
    if dist.enabled:
        ax.axvspan(
            dist.start_time,
            dist.end_time,
            color=DIST_COLOR,
            alpha=0.22,
            label=f"Wind gust {tuple(dist.acceleration)} m/s²" if label else None,
        )


def _finish(fig: plt.Figure, out: OutputConfig, filename: str) -> str:
    path = os.path.join(out.results_dir, filename)
    fig.tight_layout()
    if out.save_plots:
        fig.savefig(path, dpi=out.plot_dpi)
    return path


def plot_xy(results: List[SimulationResult], out: OutputConfig, prefix: str, dist: DisturbanceConfig) -> str:
    """1. XY trajectory: reference, PID, LQR (gust segment highlighted)."""
    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    ref = results[0].refs
    ax.plot(ref[:, 0], ref[:, 1], label="Reference", **_style("Reference"))
    for r in results:
        ax.plot(r.states[:, 0], r.states[:, 1], label=r.name, **_style(r.name))
        if dist.enabled:
            m = (r.t >= dist.start_time) & (r.t <= dist.end_time)
            ax.plot(r.states[m, 0], r.states[m, 1], color=DIST_COLOR, linewidth=4, alpha=0.5, zorder=1)
    ax.plot(*results[0].states[0, :2], "ko", markersize=6, label="Start")
    if dist.enabled:
        ax.plot([], [], color=DIST_COLOR, linewidth=4, alpha=0.5, label="Path during gust")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_title("XY trajectory tracking")
    ax.set_aspect("equal", adjustable="datalim")
    ax.grid(True, alpha=0.35)
    ax.legend(loc="best", fontsize=9)
    return _finish(fig, out, f"{prefix}_xy_trajectory.png")


def plot_position_error(results: List[SimulationResult], out: OutputConfig, prefix: str, dist: DisturbanceConfig) -> str:
    """2. Euclidean position error vs time."""
    fig, ax = plt.subplots(figsize=(10, 4.5))
    _shade_disturbance(ax, dist)
    for r in results:
        ax.plot(r.t, r.position_error, label=r.name, **_style(r.name))
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("‖p_ref − p‖ [m]")
    ax.set_title("Position error vs time")
    ax.set_xlim(results[0].t[0], results[0].t[-1])
    ax.set_ylim(bottom=0)
    ax.grid(True, alpha=0.35)
    ax.legend(loc="upper right", fontsize=9)
    return _finish(fig, out, f"{prefix}_position_error.png")


def plot_tracking(results: List[SimulationResult], out: OutputConfig, prefix: str, dist: DisturbanceConfig) -> str:
    """3. x(t) and y(t) tracking vs time."""
    fig, axes = plt.subplots(2, 1, figsize=(10, 6.5), sharex=True)
    for i, (ax, lab) in enumerate(zip(axes, ["x [m]", "y [m]"])):
        _shade_disturbance(ax, dist, label=(i == 0))
        ax.plot(results[0].t, results[0].refs[:, i], label="Reference", **_style("Reference"))
        for r in results:
            ax.plot(r.t, r.states[:, i], label=r.name, **_style(r.name))
        ax.set_ylabel(lab)
        ax.grid(True, alpha=0.35)
    axes[0].set_title("Position tracking vs time")
    axes[0].legend(loc="upper right", fontsize=9, ncol=4)
    axes[-1].set_xlabel("Time [s]")
    axes[-1].set_xlim(results[0].t[0], results[0].t[-1])
    return _finish(fig, out, f"{prefix}_xy_tracking_vs_time.png")


def plot_control(
    results: List[SimulationResult], out: OutputConfig, prefix: str, dist: DisturbanceConfig, accel_limit: float
) -> str:
    """4. Applied control inputs a_x(t), a_y(t) (zero-order-hold steps)."""
    fig, axes = plt.subplots(2, 1, figsize=(10, 6.5), sharex=True)
    for i, (ax, lab) in enumerate(zip(axes, ["a_x [m/s²]", "a_y [m/s²]"])):
        _shade_disturbance(ax, dist, label=(i == 0))
        for r in results:
            ax.step(r.t_inputs, r.inputs[:, i], where="post", label=r.name, **_style(r.name))
        ax.axhline(accel_limit, color="gray", linestyle=":", linewidth=1.2)
        ax.axhline(-accel_limit, color="gray", linestyle=":", linewidth=1.2,
                   label="Actuator limit" if i == 0 else None)
        ax.set_ylabel(lab)
        ax.grid(True, alpha=0.35)
    axes[0].set_title("Applied control input vs time")
    axes[0].legend(loc="upper right", fontsize=9, ncol=4)
    axes[-1].set_xlabel("Time [s]")
    axes[-1].set_xlim(results[0].t[0], results[0].t[-1])
    return _finish(fig, out, f"{prefix}_control_inputs.png")


def plot_disturbance_response(
    results: List[SimulationResult],
    nominals: Dict[str, SimulationResult],
    metrics: Dict[str, Dict[str, float]],
    out: OutputConfig,
    prefix: str,
    dist: DisturbanceConfig,
    mcfg: MetricsConfig,
) -> Optional[str]:
    """
    5. Disturbance response, zoomed around the gust.

    Top: deviation from the undisturbed run, ||p_disturbed - p_nominal||,
         with the recovery tolerance and each controller's recovery instant.
    Bottom: the extra control each controller spends because of the gust,
         per axis (u_disturbed - u_nominal), against the wind acceleration.
         Perfect rejection would mean Δu = -wind during the gust.
    """
    if not dist.enabled:
        return None
    t0 = max(dist.start_time - mcfg.plot_window_before, 0.0)
    t1 = dist.end_time + mcfg.response_window_after

    fig, axes = plt.subplots(3, 1, figsize=(10, 8.5), sharex=True)
    ax = axes[0]
    _shade_disturbance(ax, dist)
    for r in results:
        nom = nominals[r.name]
        dev = np.linalg.norm(r.states[:, :2] - nom.states[:, :2], axis=1)
        m = (r.t >= t0) & (r.t <= t1)
        ax.plot(r.t[m], dev[m], label=r.name, **_style(r.name))
        rec = metrics[r.name]["recovery_time"]
        if np.isfinite(rec):
            ax.axvline(dist.end_time + rec, color=_style(r.name).get("color", "k"),
                       linestyle="-.", linewidth=1.1, label=f"{r.name} recovered (+{rec:.2f} s)")
    ax.axhline(mcfg.recovery_tolerance, color="gray", linestyle=":", linewidth=1.2,
               label=f"Recovery tolerance {mcfg.recovery_tolerance} m")
    ax.set_ylabel("‖p − p_nominal‖ [m]")
    ax.set_title("Disturbance response: deviation from the undisturbed run")
    ax.set_ylim(bottom=0)
    ax.grid(True, alpha=0.35)
    ax.legend(loc="upper right", fontsize=8)

    for i, (ax, lab) in enumerate(zip(axes[1:], ["Δa_x [m/s²]", "Δa_y [m/s²]"])):
        _shade_disturbance(ax, dist, label=False)
        for r in results:
            nom = nominals[r.name]
            m = (r.t_inputs >= t0) & (r.t_inputs <= t1)
            du = r.inputs[m, i] - nom.inputs[m, i]
            ax.step(r.t_inputs[m], du, where="post", label=r.name, **_style(r.name))
        tt = r.t_inputs[(r.t_inputs >= t0) & (r.t_inputs <= t1)]
        w_neg = np.where((tt >= dist.start_time) & (tt < dist.end_time), -dist.acceleration[i], 0.0)
        ax.step(tt, w_neg, where="post", color=DIST_COLOR, linestyle="--", linewidth=1.3,
                label="−wind (ideal cancellation)")
        ax.axhline(0, color="black", linewidth=0.6)
        ax.set_ylabel(lab)
        ax.grid(True, alpha=0.35)
        if i == 0:
            ax.legend(loc="upper right", fontsize=8)
    axes[-1].set_xlabel("Time [s]")
    axes[-1].set_xlim(t0, t1)
    return _finish(fig, out, f"{prefix}_disturbance_response.png")


def generate_all_plots(
    results: List[SimulationResult],
    nominals: Dict[str, SimulationResult],
    metrics: Dict[str, Dict[str, float]],
    out: OutputConfig,
    prefix: str,
    dist: DisturbanceConfig,
    mcfg: MetricsConfig,
    accel_limit: float,
) -> List[str]:
    """Create and save all five figures; returns the saved file paths."""
    os.makedirs(out.results_dir, exist_ok=True)
    paths = [
        plot_xy(results, out, prefix, dist),
        plot_position_error(results, out, prefix, dist),
        plot_tracking(results, out, prefix, dist),
        plot_control(results, out, prefix, dist, accel_limit),
    ]
    p = plot_disturbance_response(results, nominals, metrics, out, prefix, dist, mcfg)
    if p:
        paths.append(p)
    return paths


def save_animation(
    results: List[SimulationResult], out: OutputConfig, prefix: str, dist: DisturbanceConfig
) -> str:
    """
    Animated GIF of the reference point and both drones moving along the path.
    Uses Pillow (already a Matplotlib dependency), so no extra installs.
    """
    os.makedirs(out.results_dir, exist_ok=True)
    t = results[0].t
    dt = t[1] - t[0]
    # Simulation samples to advance per animation frame.
    stride = max(1, int(round(out.animation_speedup / (out.animation_fps * dt))))
    frames = np.arange(0, len(t), stride)

    fig, ax = plt.subplots(figsize=(6.5, 6))
    ref = results[0].refs
    ax.plot(ref[:, 0], ref[:, 1], color="#999999", linestyle="--", linewidth=1, label="Reference path")
    allx = np.concatenate([ref[:, 0]] + [r.states[:, 0] for r in results])
    ally = np.concatenate([ref[:, 1]] + [r.states[:, 1] for r in results])
    pad = 0.8
    ax.set_xlim(allx.min() - pad, allx.max() + pad)
    ax.set_ylim(ally.min() - pad, ally.max() + pad)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")

    ref_pt, = ax.plot([], [], "o", color="#222222", markersize=9, label="Reference point")
    trails, dots = [], []
    for r in results:
        c = _style(r.name).get("color", None)
        tr, = ax.plot([], [], color=c, linewidth=1.4, alpha=0.8)
        dt_, = ax.plot([], [], "o", color=c, markersize=8, label=f"{r.name} drone")
        trails.append(tr)
        dots.append(dt_)
    title = ax.set_title("")
    ax.legend(loc="upper right", fontsize=8)
    trail_len = int(3.0 / dt)

    def update(k: int):
        ref_pt.set_data([ref[k, 0]], [ref[k, 1]])
        for r, tr, d in zip(results, trails, dots):
            s = max(0, k - trail_len)
            tr.set_data(r.states[s:k + 1, 0], r.states[s:k + 1, 1])
            d.set_data([r.states[k, 0]], [r.states[k, 1]])
        gust = dist.enabled and dist.start_time <= t[k] < dist.end_time
        title.set_text(f"t = {t[k]:5.2f} s" + ("   WIND GUST ACTIVE" if gust else ""))
        title.set_color("#B35900" if gust else "black")
        return [ref_pt, *trails, *dots, title]

    anim = animation.FuncAnimation(fig, update, frames=frames, interval=1000 / out.animation_fps, blit=False)
    path = os.path.join(out.results_dir, f"{prefix}_animation.gif")
    anim.save(path, writer=animation.PillowWriter(fps=out.animation_fps))
    plt.close(fig)
    return path
