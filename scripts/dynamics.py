"""Dynamic system models and discretization utilities."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Tuple
import numpy as np
from scipy.linalg import expm


@dataclass
class LinearTimeInvariantSystem:
    """Compact container for continuous-time LTI models."""
    A: np.ndarray
    B: np.ndarray
    C: np.ndarray
    D: np.ndarray | None = None

    def discretize(self, ts: float) -> "LinearTimeInvariantSystem":
        """Return a zero-order-hold discretized copy of the system."""
        Ad, Bd = zoh_discretize(self.A, self.B, ts)
        Cd = self.C.copy()
        if self.D is None:
            Dd = np.zeros((self.C.shape[0], self.B.shape[1]))
        else:
            Dd = self.D.copy()
        return LinearTimeInvariantSystem(Ad, Bd, Cd, Dd)


def zoh_discretize(A: np.ndarray, B: np.ndarray, ts: float) -> Tuple[np.ndarray, np.ndarray]:
    """Discretize the continuous-time pair (A, B) with a zero-order hold."""
    n = A.shape[0]
    m = B.shape[1]
    M = np.zeros((n + m, n + m))
    M[:n, :n] = A
    M[:n, n:] = B
    Md = expm(M * ts)
    Ad = Md[:n, :n]
    Bd = Md[:n, n:]
    return Ad, Bd


def mass_spring_damper(m: float = 1.0, c: float = 0.4, k: float = 4.0) -> LinearTimeInvariantSystem:
    """Construct the continuous mass-spring-damper model."""
    A = np.array([[0.0, 1.0],
                  [-k / m, -c / m]])
    B = np.array([[0.0],
                  [1.0 / m]])
    C = np.array([[1.0, 0.0]])
    return LinearTimeInvariantSystem(A=A, B=B, C=C)


def canonical_double_integrator() -> LinearTimeInvariantSystem:
    """Return the canonical double integrator system useful for pole-placement examples."""
    A = np.array([[0.0, 1.0],
                  [0.0, 0.0]])
    B = np.array([[0.0],
                  [1.0]])
    C = np.array([[1.0, 0.0]])
    return LinearTimeInvariantSystem(A, B, C)


def first_order_plant(a: float = -0.5, b: float = 1.0) -> LinearTimeInvariantSystem:
    """Return a basic first-order system useful for adaptive control demos."""
    A = np.array([[a]])
    B = np.array([[b]])
    C = np.array([[1.0]])
    return LinearTimeInvariantSystem(A, B, C)
