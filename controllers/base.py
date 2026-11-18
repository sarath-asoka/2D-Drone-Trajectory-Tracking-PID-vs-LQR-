"""Common controller interface used by the simulator."""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class Controller(ABC):
    """
    A trajectory-tracking controller.

    Every controller receives exactly the same information each step:
    the time t, the measured state X = [x, y, vx, vy] and the reference
    state X_ref = [x_ref, y_ref, vx_ref, vy_ref]. Neither controller is told
    about the disturbance.
    """

    name: str = "controller"

    @abstractmethod
    def compute(self, t: float, x: np.ndarray, x_ref: np.ndarray) -> np.ndarray:
        """Return the commanded acceleration [ax, ay] (before plant saturation)."""

    def reset(self) -> None:
        """Clear any internal state (e.g. integrators). Default: nothing to clear."""
