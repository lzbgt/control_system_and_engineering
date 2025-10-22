"""Simulation utilities for discrete-time control experiments."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Tuple
import numpy as np


@dataclass
class SimulationResult:
    time: np.ndarray
    states: np.ndarray
    outputs: np.ndarray
    control: np.ndarray
    reference: np.ndarray
    metadata: Dict[str, Any]

    def performance_metrics(self) -> Dict[str, float]:
        if "ts" not in self.metadata:
            raise ValueError("Simulation metadata must contain 'ts'.")
        ts = self.metadata["ts"]
        steady_samples = max(int(1.0 / ts), 1)
        error = self.reference - self.outputs
        steady_error = float(np.mean(error[-steady_samples:]))
        overshoot = 0.0
        final_ref = self.reference[-1]
        if abs(final_ref) > 1e-9:
            overshoot = max(0.0, (np.max(self.outputs) - final_ref) / abs(final_ref))
        return {"steady_state_error": steady_error, "percent_overshoot": overshoot * 100.0}


def simulate_discrete_system(
    Ad: np.ndarray,
    Bd: np.ndarray,
    C: np.ndarray,
    controller_step: Callable[[float, float, float], float],
    ts: float,
    steps: int,
    reference: Callable[[int], float],
    process_noise_std: float = 0.0,
    measurement_noise_std: float = 0.0,
    quantization: float | None = None,
    actuator_limits: Tuple[float, float] | None = None,
) -> SimulationResult:
    """Simulate a discrete-time system with a user-provided controller callback."""
    n = Ad.shape[0]
    x = np.zeros((n, 1))
    time = np.arange(steps) * ts
    states = np.zeros((steps, n))
    outputs = np.zeros(steps)
    control = np.zeros(steps)
    ref = np.zeros(steps)

    for k in range(steps):
        r = reference(k)
        ref[k] = r
        y_true = float(C @ x) + np.random.randn() * process_noise_std
        y_meas = y_true + np.random.randn() * measurement_noise_std
        if quantization is not None:
            y_meas = np.round(y_meas / quantization) * quantization
        u = controller_step(r, y_meas, time[k])
        if actuator_limits is not None:
            u = float(np.clip(u, actuator_limits[0], actuator_limits[1]))
        control[k] = u
        outputs[k] = y_true
        states[k] = x.flatten()
        x = Ad @ x + Bd * u

    metadata = {"ts": ts}
    return SimulationResult(
        time=time,
        states=states,
        outputs=outputs,
        control=control,
        reference=ref,
        metadata=metadata,
    )
