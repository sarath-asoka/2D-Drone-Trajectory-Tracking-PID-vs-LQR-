"""
2D Drone Trajectory Tracking — PID vs LQR

Usage
-----
    python main.py                              # circle, with wind gust
    python main.py --trajectory figure_eight
    python main.py --trajectory waypoints --no-disturbance
    python main.py --trajectory all --animate   # every trajectory + GIFs
    python main.py --show                       # also open the figures

Both controllers are run by the *same* Simulator instance, so they share
initial state, trajectory, duration, timestep, actuator limits and disturbance.
"""

from __future__ import annotations

import argparse
import copy
import os
import sys
from typing import Dict, List

import matplotlib
import numpy as np

from config import Config, default_config

TRAJECTORIES = ["circle", "figure_eight", "waypoints"]


def parse_args(argv: List[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="2D drone trajectory tracking: PID vs LQR")
    p.add_argument("--trajectory", choices=TRAJECTORIES + ["all"], default=None,
                   help="reference trajectory (default: value in config.py)")
    p.add_argument("--duration", type=float, default=None, help="simulation duration [s]")
    p.add_argument("--dt", type=float, default=None, help="timestep [s]")
    p.add_argument("--no-disturbance", action="store_true", help="disable the wind gust")
    p.add_argument("--animate", action="store_true", help="also save a GIF animation")
    p.add_argument("--show", action="store_true", help="show figures interactively")
    p.add_argument("--results-dir", default=None, help="output folder (default: results/)")
    return p.parse_args(argv)


def apply_overrides(cfg: Config, args: argparse.Namespace) -> Config:
    """Return a copy of cfg with the command-line overrides applied."""
    cfg = copy.deepcopy(cfg)
    if args.duration is not None:
        cfg.sim.duration = args.duration
    if args.dt is not None:
        cfg.sim.dt = args.dt
    if args.no_disturbance:
        cfg.disturbance.enabled = False
    if args.animate:
        cfg.output.animate = True
    if args.results_dir is not None:
        cfg.output.results_dir = args.results_dir
    return cfg


def run_experiment(cfg: Config, show: bool = False) -> Dict[str, Dict[str, float]]:
    """Run PID and LQR on the configured scenario, print metrics, save plots."""
    # Imports are deferred so the Matplotlib backend is chosen first.
    import matplotlib.pyplot as plt

    from controllers.lqr import LQRController
    from controllers.pid import PIDController
    from simulator import Simulator
    from utils.metrics import compute_metrics, format_comparison_table, metrics_to_csv
    from utils.plotting import generate_all_plots, save_animation

    sim = Simulator(cfg)
    lqr = LQRController.from_config(cfg.lqr, sim.model.A, sim.model.B)
    pid = PIDController.from_config(cfg.pid, cfg.sim.accel_limit, cfg.sim.dt)

    name = cfg.trajectory.name
    print(f"\n### Trajectory: {name}   (dt = {cfg.sim.dt} s, T = {cfg.sim.duration} s, "
          f"|a_i| <= {cfg.sim.accel_limit} m/s²)")
    print(f"Initial state X0 = {np.round(sim.x0, 3).tolist()}")
    if cfg.disturbance.enabled:
        d = cfg.disturbance
        print(f"Wind gust: a_w = {tuple(d.acceleration)} m/s² for {d.start_time} s <= t < {d.end_time} s")
    else:
        print("Wind gust: disabled")

    with np.printoptions(precision=4, suppress=True):
        print("LQR gain K =\n", lqr.K)
    eig = np.abs(lqr.closed_loop_eigenvalues())
    print(f"DARE residual = {lqr.riccati_residual():.2e},  max |eig(A - BK)| = {eig.max():.6f}")
    print(f"PID gains: Kp = {cfg.pid.kp}, Ki = {cfg.pid.ki}, Kd = {cfg.pid.kd}")

    controllers = [pid, lqr]
    results = [sim.run(c) for c in controllers]
    # Nominal (gust-free) runs of the same controllers, used only to isolate
    # the effect of the gust for the disturbance-rejection metrics.
    nominals = (
        {c.name: sim.run(c, with_disturbance=False) for c in controllers}
        if cfg.disturbance.enabled else {}
    )
    metrics = {
        r.name: compute_metrics(r, nominals.get(r.name), cfg.disturbance, cfg.metrics,
                                cfg.sim.dt, cfg.sim.accel_limit)
        for r in results
    }

    print()
    print(format_comparison_table(metrics, title=f"Comparison — {name}"))

    os.makedirs(cfg.output.results_dir, exist_ok=True)
    csv_path = os.path.join(cfg.output.results_dir, f"{name}_metrics.csv")
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write(metrics_to_csv(metrics))

    paths = generate_all_plots(results, nominals, metrics, cfg.output, name, cfg.disturbance,
                               cfg.metrics, cfg.sim.accel_limit)
    if cfg.output.animate:
        print("Rendering animation (this can take a little while)...")
        paths.append(save_animation(results, cfg.output, name, cfg.disturbance))

    print(f"\nSaved: {csv_path}")
    for pth in paths:
        print(f"Saved: {pth}")

    if show:
        plt.show()
    plt.close("all")
    return metrics


def main(argv: List[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    if not args.show:
        matplotlib.use("Agg")  # headless: just save files

    base = apply_overrides(default_config(), args)
    selected = args.trajectory or base.trajectory.name
    names = TRAJECTORIES if selected == "all" else [selected]

    for traj_name in names:
        cfg = copy.deepcopy(base)
        cfg.trajectory.name = traj_name
        run_experiment(cfg, show=args.show)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
