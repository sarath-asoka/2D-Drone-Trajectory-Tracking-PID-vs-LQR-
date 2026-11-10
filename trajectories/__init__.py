"""Reference trajectory generators."""

from trajectories.trajectories import (
    CircleTrajectory,
    FigureEightTrajectory,
    Trajectory,
    WaypointTrajectory,
    make_trajectory,
)

__all__ = [
    "Trajectory",
    "CircleTrajectory",
    "FigureEightTrajectory",
    "WaypointTrajectory",
    "make_trajectory",
]
