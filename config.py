"""
Central configuration for the 2D drone PID vs LQR simulation.

Every tunable number in the project lives here so that experiments are
reproducible and both controllers are guaranteed to see *identical*
simulation settings (initial state, trajectory, duration, timestep,
actuator limits and disturbance).

Edit the default values below, or override them from the command line
(see ``python main.py --help``).

Tuning philosophy (to keep the comparison fair)
-----------------------------------------------
Both controllers are tuned to a similar closed-loop bandwidth of roughly
2 rad/s per axis, so that the comparison reflects controller *structure*
(integral action, anti-windup, optimal gain computation) rather than one
controller simply being made more aggressive than the other:

* PID:  Kp = 4.0, Kd = 3.2  ->  error dynamics  s^2 + 3.2 s + 4  (wn = 2, zeta = 0.8)
        Ki = 1.0 adds slow integral action.
* LQR:  Q = diag(16, 16, 1, 1), R = diag(1, 1)  ->  for the continuous-time
        double integrator this gives k_p = sqrt(16/1) = 4 and
        k_v = sqrt(2*4 + 1) = 3.0 (wn = 2, zeta = 0.75). The discrete gain
        computed by the code is very close to this.

Neither tuning was iterated on to win any particular metric.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple


@dataclass
class SimulationConfig:
    """Settings shared by every controller run."""

    dt: float = 0.01
    """Simulation / controller timestep [s]."""

    duration: float = 30.0
    """Total simulated time [s]."""

    accel_limit: float = 5.0
    """Actuator saturation: each axis is limited to |a_i| <= accel_limit [m/s^2]."""

    initial_position_offset: Tuple[float, float] = (1.0, -1.0)
    """Initial position = reference position at t=0 plus this offset [m]."""

    initial_velocity: Tuple[float, float] = (0.0, 0.0)
    """Initial velocity [m/s] (the drone starts at rest by default)."""


@dataclass
class TrajectoryConfig:
    """Reference trajectory selection and parameters."""

    name: str = "circle"
    """One of: 'circle', 'figure_eight', 'waypoints'."""

    # Circle
    circle_center: Tuple[float, float] = (0.0, 0.0)
    circle_radius: float = 5.0          # [m]
    circle_period: float = 20.0         # time for one revolution [s]

    # Figure-eight (1:2 Lissajous curve)
    fig8_center: Tuple[float, float] = (0.0, 0.0)
    fig8_amplitude_x: float = 6.0       # [m]
    fig8_amplitude_y: float = 3.0       # [m]
    fig8_period: float = 24.0           # time for one full figure-eight [s]

    # Waypoint path (piecewise-linear at constant speed)
    waypoints: List[Tuple[float, float]] = field(
        default_factory=lambda: [(0.0, 0.0), (8.0, 0.0), (8.0, 6.0), (0.0, 6.0), (0.0, 0.0)]
    )
    waypoint_speed: float = 1.5         # [m/s]
    waypoint_loop: bool = True          # restart the path after the last waypoint


@dataclass
class PIDConfig:
    """Gains for the two independent (x and y) PID loops: (x_axis, y_axis)."""

    kp: Tuple[float, float] = (4.0, 4.0)          # [1/s^2]
    ki: Tuple[float, float] = (1.0, 1.0)          # [1/s^3]
    kd: Tuple[float, float] = (3.2, 3.2)          # [1/s]

    integral_limit: float = 2.0
    """Hard clamp on the accumulated integral of position error [m*s]."""

    output_limit: Optional[float] = None
    """PID output limit [m/s^2]. ``None`` means 'use SimulationConfig.accel_limit'."""


@dataclass
class LQRConfig:
    """
    Weights of the discrete LQR cost

        J = sum_k ( e_k^T Q e_k + u_k^T R u_k ),   e_k = X_k - X_ref,k

    Q (4x4, state order [x, y, vx, vy]) penalises tracking error: a larger
    entry makes the controller work harder to drive that state error to zero.
    R (2x2, input order [ax, ay]) penalises control effort: a larger entry
    makes the controller use less acceleration (slower, gentler response).
    Only the ratio Q/R matters for the resulting gain.
    """

    q_diag: Tuple[float, float, float, float] = (16.0, 16.0, 1.0, 1.0)
    r_diag: Tuple[float, float] = (1.0, 1.0)


@dataclass
class DisturbanceConfig:
    """Constant external acceleration ('wind gust') over a time window."""

    enabled: bool = True
    start_time: float = 10.0                         # [s]
    end_time: float = 12.0                           # [s]
    acceleration: Tuple[float, float] = (2.0, -1.0)  # [m/s^2]


@dataclass
class MetricsConfig:
    """
    Parameters for the disturbance metrics.

    Disturbance rejection is measured against a *nominal* run: the same
    controller, same everything, but with the gust switched off. The
    disturbance deviation d(t) = ||p_disturbed(t) - p_nominal(t)|| then
    contains only the effect of the gust (not the normal tracking error,
    which varies along the trajectory).
    """

    recovery_tolerance: float = 0.05
    """Recovered once d(t) <= tolerance [m] ..."""

    recovery_hold_time: float = 0.5
    """... and stays there until the end of the run, for at least this long [s]."""

    plot_window_before: float = 2.0
    """Time shown before the gust in the disturbance-response plot [s]."""

    response_window_after: float = 8.0
    """Time shown after the gust in the disturbance-response plot [s]."""


@dataclass
class OutputConfig:
    """Where and how results are written."""

    results_dir: str = "results"
    save_plots: bool = True
    animate: bool = False            # GIF animation is optional (slower)
    animation_fps: int = 20
    animation_speedup: float = 2.0   # 2.0 -> animation plays at 2x real time
    plot_dpi: int = 140


@dataclass
class Config:
    """Top-level configuration bundle."""

    sim: SimulationConfig = field(default_factory=SimulationConfig)
    trajectory: TrajectoryConfig = field(default_factory=TrajectoryConfig)
    pid: PIDConfig = field(default_factory=PIDConfig)
    lqr: LQRConfig = field(default_factory=LQRConfig)
    disturbance: DisturbanceConfig = field(default_factory=DisturbanceConfig)
    metrics: MetricsConfig = field(default_factory=MetricsConfig)
    output: OutputConfig = field(default_factory=OutputConfig)


def default_config() -> Config:
    """Return a fresh configuration object with the default values."""
    return Config()
