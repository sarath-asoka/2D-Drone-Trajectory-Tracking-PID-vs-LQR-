"""
2D double-integrator drone model.

State  X = [x, y, vx, vy]^T      (position [m], velocity [m/s])
Input  U = [ax, ay]^T            (commanded acceleration [m/s^2])

Continuous-time dynamics
------------------------
    dx/dt  = vx
    dy/dt  = vy
    dvx/dt = ax + wx
    dvy/dt = ay + wy

where W = [wx, wy] is an external disturbance acceleration (e.g. wind).
In state-space form:

    dX/dt = A_c X + B_c (U + W)

        [0 0 1 0]          [0 0]
    A_c=[0 0 0 1]     B_c= [0 0]
        [0 0 0 0]          [1 0]
        [0 0 0 0]          [0 1]

Exact zero-order-hold (ZOH) discretisation
------------------------------------------
If U and W are held constant over each step of length dt, the exact
discrete model is X[k+1] = A X[k] + B (U[k] + W[k]) with

    A = e^{A_c dt},       B = ( integral_0^dt e^{A_c s} ds ) B_c

Because A_c is nilpotent (A_c^2 = 0) the matrix exponential series stops
after two terms, giving the familiar closed form

        [1 0 dt 0 ]          [dt^2/2   0    ]
    A = [0 1 0  dt]      B = [  0    dt^2/2 ]
        [0 0 1  0 ]          [  dt     0    ]
        [0 0 0  1 ]          [  0      dt   ]

This is *exact* (not an Euler approximation) for piecewise-constant
inputs, which is what a digital controller produces.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
from scipy.linalg import expm


class Drone2D:
    """Discrete-time 2D double-integrator drone (exact ZOH discretisation)."""

    n_states: int = 4
    n_inputs: int = 2

    def __init__(self, dt: float) -> None:
        """
        Args:
            dt: Sampling / integration timestep [s]. Must be positive.
        """
        if dt <= 0.0:
            raise ValueError("dt must be positive")
        self.dt = float(dt)
        self.A_c, self.B_c = self.continuous_matrices()
        self.A, self.B = self.discretize_zoh(self.dt)

    # ------------------------------------------------------------------ #
    # Model matrices
    # ------------------------------------------------------------------ #
    @staticmethod
    def continuous_matrices() -> Tuple[np.ndarray, np.ndarray]:
        """Return the continuous-time state-space matrices (A_c, B_c)."""
        A_c = np.array(
            [
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
                [0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0],
            ]
        )
        B_c = np.array(
            [
                [0.0, 0.0],
                [0.0, 0.0],
                [1.0, 0.0],
                [0.0, 1.0],
            ]
        )
        return A_c, B_c

    @staticmethod
    def discretize_zoh(dt: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        Exact ZOH discretisation written out explicitly (closed form).

        Args:
            dt: Timestep [s].

        Returns:
            (A, B) discrete-time matrices.
        """
        A = np.array(
            [
                [1.0, 0.0, dt, 0.0],
                [0.0, 1.0, 0.0, dt],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ]
        )
        B = np.array(
            [
                [0.5 * dt**2, 0.0],
                [0.0, 0.5 * dt**2],
                [dt, 0.0],
                [0.0, dt],
            ]
        )
        return A, B

    @staticmethod
    def discretize_expm(A_c: np.ndarray, B_c: np.ndarray, dt: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generic ZOH discretisation via the augmented matrix exponential

            expm( [[A_c, B_c], [0, 0]] * dt ) = [[A, B], [0, I]]

        Used only as an independent cross-check of :meth:`discretize_zoh`.
        """
        n, m = B_c.shape
        M = np.zeros((n + m, n + m))
        M[:n, :n] = A_c
        M[:n, n:] = B_c
        Md = expm(M * dt)
        return Md[:n, :n], Md[:n, n:]

    # ------------------------------------------------------------------ #
    # Dynamics
    # ------------------------------------------------------------------ #
    def derivative(self, x: np.ndarray, u: np.ndarray, w: np.ndarray) -> np.ndarray:
        """Continuous-time derivative dX/dt = A_c X + B_c (U + W)."""
        return self.A_c @ x + self.B_c @ (u + w)

    def step(self, x: np.ndarray, u: np.ndarray, w: np.ndarray) -> np.ndarray:
        """
        Advance the state by one timestep.

        The disturbance W enters through the same input matrix as the
        control (it is an external acceleration), so

            X[k+1] = A X[k] + B (U[k] + W[k]).

        Args:
            x: Current state [x, y, vx, vy].
            u: Applied (already saturated) acceleration [ax, ay].
            w: Disturbance acceleration [wx, wy].

        Returns:
            Next state.
        """
        return self.A @ x + self.B @ (u + w)


class WindDisturbance:
    """
    Constant external acceleration applied on the interval [t_start, t_end).

    The disturbance is unknown to both controllers; it only acts on the
    plant. Sampling it at the start of each step and holding it for the step
    matches the ZOH model exactly when the window edges lie on the time grid.
    """

    _EPS = 1e-9  # guards against floating-point round-off in k*dt

    def __init__(
        self,
        enabled: bool,
        start_time: float,
        end_time: float,
        acceleration: Tuple[float, float],
    ) -> None:
        if end_time < start_time:
            raise ValueError("disturbance end_time must be >= start_time")
        self.enabled = bool(enabled)
        self.start_time = float(start_time)
        self.end_time = float(end_time)
        self.acceleration = np.asarray(acceleration, dtype=float)

    def is_active(self, t: float) -> bool:
        """True if the gust is acting at time t."""
        return (
            self.enabled
            and (self.start_time - self._EPS) <= t < (self.end_time - self._EPS)
        )

    def at(self, t: float) -> np.ndarray:
        """Disturbance acceleration [wx, wy] at time t."""
        return self.acceleration.copy() if self.is_active(t) else np.zeros(2)
