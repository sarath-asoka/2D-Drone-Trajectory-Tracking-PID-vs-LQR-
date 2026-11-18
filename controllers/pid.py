"""
PID trajectory-tracking controller with independent x and y loops.

For each axis i in {x, y}:

    error          e_i(t)    = p_ref,i(t) - p_i(t)
    derivative     de_i/dt   = v_ref,i(t) - v_i(t)
    integral       I_i(t)    = integral_0^t e_i(tau) dtau

    a_i = Kp e_i + Ki I_i + Kd de_i/dt,   then saturated to [-a_max, a_max]

Why the derivative uses velocities
----------------------------------
Since e = p_ref - p, its exact time derivative is v_ref - v. Using the
measured/reference velocities is therefore the *true* derivative of the
error — no noisy finite differencing and no "derivative kick" when the
reference jumps (e.g. at a waypoint corner). LQR receives the same velocity
information, so this keeps the comparison fair.

Anti-windup (conditional integration / clamping)
------------------------------------------------
When the actuator saturates, the integrator would otherwise keep growing
("windup") and cause a large overshoot when the error finally changes sign.
We stop integrating whenever (a) the output is saturated AND (b) the error
would push the integral further in the saturated direction. Additionally the
integral is hard-clamped to +/- integral_limit.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

from controllers.base import Controller


class PIDAxis:
    """A single-axis discrete PID controller with output limits and anti-windup."""

    def __init__(
        self,
        kp: float,
        ki: float,
        kd: float,
        output_limit: float,
        integral_limit: float,
        dt: float,
    ) -> None:
        """
        Args:
            kp, ki, kd: Proportional, integral and derivative gains (>= 0).
            output_limit: Symmetric output saturation |u| <= output_limit.
            integral_limit: Symmetric clamp on the error integral.
            dt: Controller sampling time [s].
        """
        if min(kp, ki, kd) < 0:
            raise ValueError("PID gains must be non-negative")
        if output_limit <= 0 or integral_limit <= 0 or dt <= 0:
            raise ValueError("limits and dt must be positive")
        self.kp, self.ki, self.kd = float(kp), float(ki), float(kd)
        self.output_limit = float(output_limit)
        self.integral_limit = float(integral_limit)
        self.dt = float(dt)
        self.reset()

    def reset(self) -> None:
        """Clear the integrator and logged terms."""
        self.integral = 0.0
        self.last_error = 0.0
        self.last_error_rate = 0.0
        self.last_terms: Tuple[float, float, float] = (0.0, 0.0, 0.0)
        self.saturated = False

    def update(self, error: float, error_rate: float) -> float:
        """
        Compute one control output.

        Args:
            error: e = reference - measurement (position) [m].
            error_rate: de/dt = reference rate - measured rate (velocity) [m/s].

        Returns:
            Saturated control output (acceleration) [m/s^2].
        """
        p_term = self.kp * error
        d_term = self.kd * error_rate

        # Candidate integral (forward-Euler integration, hard-clamped).
        candidate = float(np.clip(self.integral + error * self.dt,
                                  -self.integral_limit, self.integral_limit))
        u_candidate = p_term + self.ki * candidate + d_term

        # Anti-windup: refuse the integral update if the output is saturated
        # and the error has the same sign as the (too large) output, i.e.
        # integrating would push the output even deeper into saturation.
        saturated = abs(u_candidate) > self.output_limit
        winding_up = saturated and np.sign(error) == np.sign(u_candidate)
        if not winding_up:
            self.integral = candidate

        i_term = self.ki * self.integral
        u_unsat = p_term + i_term + d_term
        u = float(np.clip(u_unsat, -self.output_limit, self.output_limit))

        # Logging / introspection.
        self.saturated = abs(u_unsat) > self.output_limit
        self.last_error = error
        self.last_error_rate = error_rate
        self.last_terms = (p_term, i_term, d_term)
        return u


class PIDController(Controller):
    """Two independent PID loops: one for x, one for y."""

    name = "PID"

    def __init__(
        self,
        kp: Tuple[float, float],
        ki: Tuple[float, float],
        kd: Tuple[float, float],
        output_limit: float,
        integral_limit: float,
        dt: float,
    ) -> None:
        self.axes = [
            PIDAxis(kp[i], ki[i], kd[i], output_limit, integral_limit, dt) for i in range(2)
        ]

    @classmethod
    def from_config(cls, pid_cfg, accel_limit: float, dt: float) -> "PIDController":
        """Build from a :class:`config.PIDConfig`; output limit defaults to accel_limit."""
        limit: Optional[float] = pid_cfg.output_limit
        return cls(
            kp=pid_cfg.kp,
            ki=pid_cfg.ki,
            kd=pid_cfg.kd,
            output_limit=accel_limit if limit is None else limit,
            integral_limit=pid_cfg.integral_limit,
            dt=dt,
        )

    def reset(self) -> None:
        for axis in self.axes:
            axis.reset()

    def compute(self, t: float, x: np.ndarray, x_ref: np.ndarray) -> np.ndarray:
        # Position errors (reference - measured) and their exact derivatives.
        e_pos = x_ref[0:2] - x[0:2]
        e_vel = x_ref[2:4] - x[2:4]
        return np.array([self.axes[i].update(e_pos[i], e_vel[i]) for i in range(2)])
