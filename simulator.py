"""
Closed-loop simulator.

The same :class:`Simulator` instance (and therefore the same model,
trajectory, initial state, duration, timestep, actuator limit and
disturbance) is used for every controller. The loop is, for k = 0..N-1:

    t_k      = k dt
    X_ref,k  = trajectory(t_k)
    u_cmd,k  = controller(t_k, X_k, X_ref,k)
    u_k      = clip(u_cmd,k, -a_max, +a_max)        (actuator saturation)
    w_k      = disturbance(t_k)                      (unknown to controller)
    X_{k+1}  = A X_k + B (u_k + w_k)                 (exact ZOH step)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from config import Config
from controllers.base import Controller
from models.drone_2d import Drone2D, WindDisturbance
from trajectories.trajectories import Trajectory, make_trajectory


@dataclass
class SimulationResult:
    """Time histories of one closed-loop run.

    Shapes: ``t``, ``states``, ``refs`` have N+1 samples (state times);
    ``inputs``, ``inputs_cmd``, ``disturbance`` have N samples (one per step,
    held constant from t_k to t_{k+1}).
    """

    name: str
    t: np.ndarray            # (N+1,)
    states: np.ndarray       # (N+1, 4)
    refs: np.ndarray         # (N+1, 4)
    inputs: np.ndarray       # (N, 2)  applied (saturated) acceleration
    inputs_cmd: np.ndarray   # (N, 2)  controller command before saturation
    disturbance: np.ndarray  # (N, 2)

    @property
    def t_inputs(self) -> np.ndarray:
        """Times at which each input sample starts being applied."""
        return self.t[:-1]

    @property
    def position_error(self) -> np.ndarray:
        """Euclidean position error ||p_ref - p|| at every state time."""
        return np.linalg.norm(self.refs[:, :2] - self.states[:, :2], axis=1)


class Simulator:
    """Runs any :class:`Controller` under one fixed, shared experimental setup."""

    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg
        self.dt = cfg.sim.dt
        self.n_steps = int(round(cfg.sim.duration / cfg.sim.dt))
        self.model = Drone2D(self.dt)
        self.trajectory: Trajectory = make_trajectory(cfg.trajectory)
        d = cfg.disturbance
        self.disturbance = WindDisturbance(d.enabled, d.start_time, d.end_time, d.acceleration)
        self.accel_limit = float(cfg.sim.accel_limit)
        self.x0 = self.initial_state()

    def initial_state(self) -> np.ndarray:
        """Reference start position + configured offset, with configured velocity."""
        ref0 = self.trajectory.reference(0.0)
        off = np.asarray(self.cfg.sim.initial_position_offset, dtype=float)
        vel = np.asarray(self.cfg.sim.initial_velocity, dtype=float)
        return np.array([ref0[0] + off[0], ref0[1] + off[1], vel[0], vel[1]])

    def saturate(self, u: np.ndarray) -> np.ndarray:
        """Per-axis actuator limit, identical for every controller."""
        return np.clip(u, -self.accel_limit, self.accel_limit)

    def run(self, controller: Controller, with_disturbance: bool = True) -> SimulationResult:
        """
        Simulate one controller from the shared initial state.

        Args:
            controller: Controller under test.
            with_disturbance: If False, the gust is suppressed. Used to build
                the nominal run against which disturbance rejection is measured.
        """
        n = self.n_steps
        t = np.arange(n + 1) * self.dt
        states = np.zeros((n + 1, 4))
        refs = np.zeros((n + 1, 4))
        inputs = np.zeros((n, 2))
        inputs_cmd = np.zeros((n, 2))
        dist = np.zeros((n, 2))

        controller.reset()
        x = self.x0.copy()
        states[0] = x
        for k in range(n):
            tk = t[k]
            x_ref = self.trajectory.reference(tk)
            u_cmd = np.asarray(controller.compute(tk, x, x_ref), dtype=float)
            u = self.saturate(u_cmd)
            w = self.disturbance.at(tk) if with_disturbance else np.zeros(2)
            x = self.model.step(x, u, w)

            refs[k] = x_ref
            inputs_cmd[k] = u_cmd
            inputs[k] = u
            dist[k] = w
            states[k + 1] = x
        refs[n] = self.trajectory.reference(t[n])

        return SimulationResult(controller.name, t, states, refs, inputs, inputs_cmd, dist)
