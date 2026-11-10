"""
Reference trajectories.

Every trajectory returns the full reference state

    X_ref(t) = [x_ref, y_ref, vx_ref, vy_ref]

with velocities computed *analytically* (the exact time derivative of the
position), so the reference is self-consistent and both controllers receive
exactly the same information.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Sequence, Tuple

import numpy as np

from config import TrajectoryConfig


class Trajectory(ABC):
    """Abstract base class for time-parameterised reference trajectories."""

    name: str = "trajectory"

    @abstractmethod
    def reference(self, t: float) -> np.ndarray:
        """Return X_ref(t) = [x, y, vx, vy] at time t [s]."""

    def sample(self, times: np.ndarray) -> np.ndarray:
        """Evaluate the reference on an array of times. Returns shape (N, 4)."""
        return np.array([self.reference(float(t)) for t in times])


class CircleTrajectory(Trajectory):
    """
    Circle traversed counter-clockwise at constant angular rate w = 2*pi/T:

        x = cx + r cos(w t)        vx = -r w sin(w t)
        y = cy + r sin(w t)        vy =  r w cos(w t)
    """

    name = "circle"

    def __init__(self, center: Tuple[float, float], radius: float, period: float) -> None:
        if radius <= 0 or period <= 0:
            raise ValueError("radius and period must be positive")
        self.cx, self.cy = map(float, center)
        self.r = float(radius)
        self.w = 2.0 * np.pi / float(period)

    def reference(self, t: float) -> np.ndarray:
        th = self.w * t
        return np.array(
            [
                self.cx + self.r * np.cos(th),
                self.cy + self.r * np.sin(th),
                -self.r * self.w * np.sin(th),
                self.r * self.w * np.cos(th),
            ]
        )


class FigureEightTrajectory(Trajectory):
    """
    Figure-eight as a 1:2 Lissajous curve, w = 2*pi/T:

        x = cx + Ax sin(w t)         vx = Ax w cos(w t)
        y = cy + Ay sin(2 w t)       vy = 2 Ay w cos(2 w t)
    """

    name = "figure_eight"

    def __init__(
        self,
        center: Tuple[float, float],
        amplitude_x: float,
        amplitude_y: float,
        period: float,
    ) -> None:
        if period <= 0:
            raise ValueError("period must be positive")
        self.cx, self.cy = map(float, center)
        self.ax = float(amplitude_x)
        self.ay = float(amplitude_y)
        self.w = 2.0 * np.pi / float(period)

    def reference(self, t: float) -> np.ndarray:
        th = self.w * t
        return np.array(
            [
                self.cx + self.ax * np.sin(th),
                self.cy + self.ay * np.sin(2.0 * th),
                self.ax * self.w * np.cos(th),
                2.0 * self.ay * self.w * np.cos(2.0 * th),
            ]
        )


class WaypointTrajectory(Trajectory):
    """
    Piecewise-linear path through waypoints at constant speed v.

    On segment i (from P_i to P_{i+1}, length L_i, unit direction d_i):

        p(t) = P_i + v (t - t_i) d_i,     v_ref = v d_i

    where t_i is the time the segment starts. Velocity is therefore
    discontinuous at corners — a deliberately demanding test case.
    If ``loop`` is False the reference stops (zero velocity) at the last
    waypoint.
    """

    name = "waypoints"

    def __init__(self, waypoints: Sequence[Tuple[float, float]], speed: float, loop: bool = True) -> None:
        pts = np.asarray(waypoints, dtype=float)
        if pts.ndim != 2 or pts.shape[1] != 2 or len(pts) < 2:
            raise ValueError("need at least two 2D waypoints")
        if speed <= 0:
            raise ValueError("speed must be positive")

        # Remove zero-length segments (repeated consecutive points).
        keep = [0]
        for i in range(1, len(pts)):
            if np.linalg.norm(pts[i] - pts[keep[-1]]) > 1e-12:
                keep.append(i)
        pts = pts[keep]
        if len(pts) < 2:
            raise ValueError("waypoints must not all coincide")

        self.points = pts
        self.speed = float(speed)
        self.loop = bool(loop)

        seg = np.diff(pts, axis=0)
        self.seg_lengths = np.linalg.norm(seg, axis=1)
        self.seg_dirs = seg / self.seg_lengths[:, None]
        # Cumulative arc length at the start of every segment.
        self.cum_lengths = np.concatenate([[0.0], np.cumsum(self.seg_lengths)])
        self.total_length = float(self.cum_lengths[-1])

    def reference(self, t: float) -> np.ndarray:
        s = self.speed * max(t, 0.0)  # travelled arc length
        if self.loop:
            s = s % self.total_length
        elif s >= self.total_length:
            end = self.points[-1]
            return np.array([end[0], end[1], 0.0, 0.0])

        # Segment index i such that cum_lengths[i] <= s < cum_lengths[i+1].
        i = int(np.searchsorted(self.cum_lengths, s, side="right") - 1)
        i = min(max(i, 0), len(self.seg_lengths) - 1)
        pos = self.points[i] + (s - self.cum_lengths[i]) * self.seg_dirs[i]
        vel = self.speed * self.seg_dirs[i]
        return np.array([pos[0], pos[1], vel[0], vel[1]])


def make_trajectory(cfg: TrajectoryConfig) -> Trajectory:
    """Factory: build the trajectory selected in the configuration."""
    name = cfg.name.lower()
    if name == "circle":
        return CircleTrajectory(cfg.circle_center, cfg.circle_radius, cfg.circle_period)
    if name in ("figure_eight", "figure8", "figure-eight"):
        return FigureEightTrajectory(
            cfg.fig8_center, cfg.fig8_amplitude_x, cfg.fig8_amplitude_y, cfg.fig8_period
        )
    if name in ("waypoints", "waypoint"):
        return WaypointTrajectory(cfg.waypoints, cfg.waypoint_speed, cfg.waypoint_loop)
    raise ValueError(f"unknown trajectory '{cfg.name}' (use circle, figure_eight or waypoints)")
