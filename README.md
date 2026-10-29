# 2D Drone Trajectory Tracking — PID vs LQR

A Python simulation of a simplified 2D drone (double-integrator model) that compares a **PID** controller and a **discrete LQR** controller for trajectory tracking and wind-gust rejection.

> 🚧 Work in progress. Full documentation, equations and results will be added as the project is built.

## Planned features

- 2D double-integrator drone model with exact discretisation
- Circle, figure-eight and waypoint reference trajectories
- Per-axis PID with output limits and anti-windup
- Discrete LQR computed from the Riccati equation
- Fair side-by-side comparison under identical conditions
- Wind-gust disturbance test and recovery-time measurement
- Performance metrics (RMSE, IAE, control effort, ...) and Matplotlib plots

## Requirements

```bash
pip install -r requirements.txt
```

Python 3.8+, NumPy, SciPy, Matplotlib.
