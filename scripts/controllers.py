"""Controller implementations used throughout the course."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from scipy.linalg import solve_discrete_are


@dataclass
class PIDController:
    kp: float
    ki: float
    kd: float
    ts: float
    u_min: Optional[float] = None
    u_max: Optional[float] = None
    _e1: float = field(default=0.0, init=False)
    _e2: float = field(default=0.0, init=False)
    _u1: float = field(default=0.0, init=False)

    def step(self, reference: float, measurement: float) -> float:
        error = reference - measurement
        du = self.kp * (error - self._e1)
        du += self.ki * self.ts * error
        du += self.kd * (error - 2 * self._e1 + self._e2) / self.ts
        u = self._u1 + du
        if self.u_min is not None or self.u_max is not None:
            u = np.clip(u,
                        self.u_min if self.u_min is not None else -np.inf,
                        self.u_max if self.u_max is not None else np.inf)
        self._e2 = self._e1
        self._e1 = error
        self._u1 = u
        return float(u)


@dataclass
class DiscreteStateFeedback:
    K: np.ndarray
    Nbar: Optional[float] = None

    def step(self, xhat: np.ndarray, reference: float = 0.0) -> float:
        u = -float(self.K @ xhat)
        if self.Nbar is not None:
            u += self.Nbar * reference
        return u


def discrete_lqr(Ad: np.ndarray, Bd: np.ndarray, Q: np.ndarray, R: np.ndarray) -> np.ndarray:
    """Solve the discrete LQR problem and return the gain matrix."""
    P = solve_discrete_are(Ad, Bd, Q, R)
    K = np.linalg.solve(Bd.T @ P @ Bd + R, Bd.T @ P @ Ad)
    return K


def discrete_nbar(Ad: np.ndarray, Bd: np.ndarray, C: np.ndarray, K: np.ndarray) -> float:
    """Compute the reference prefilter gain for step tracking."""
    n = Ad.shape[0]
    aug = np.block([
        [np.eye(n) - Ad, -Bd],
        [C, np.zeros((C.shape[0], Bd.shape[1]))],
    ])
    rhs = np.zeros((n + C.shape[0], 1))
    rhs[n:, :] = 1.0
    sol = np.linalg.solve(aug, rhs)
    x_ss = sol[:n]
    u_ss = sol[n:]
    nbar = float(u_ss + K @ x_ss)
    return nbar


@dataclass
class KalmanFilter:
    Ad: np.ndarray
    Bd: np.ndarray
    C: np.ndarray
    Qn: np.ndarray
    Rn: np.ndarray
    xhat: np.ndarray
    P: np.ndarray

    def predict(self, u: float) -> None:
        self.xhat = self.Ad @ self.xhat + self.Bd * u
        self.P = self.Ad @ self.P @ self.Ad.T + self.Qn

    def update(self, y: float) -> float:
        S = self.C @ self.P @ self.C.T + self.Rn
        Kf = self.P @ self.C.T @ np.linalg.inv(S)
        innovation = y - float(self.C @ self.xhat)
        self.xhat = self.xhat + Kf * innovation
        self.P = (np.eye(self.P.shape[0]) - Kf @ self.C) @ self.P
        return float(innovation)


@dataclass
class MRACController:
    a_m: float = 0.95
    b_m: float = 0.05
    gamma: float = 0.5
    theta: np.ndarray = field(default_factory=lambda: np.zeros((2, 1)))
    y_model: float = 0.0

    def step(self, reference: float, measurement: float) -> float:
        error = measurement - self.y_model
        phi = np.array([[measurement], [reference]])
        self.theta -= self.gamma * phi * error
        control = float(self.theta.T @ phi)
        self.y_model = self.a_m * self.y_model + self.b_m * reference
        return control
