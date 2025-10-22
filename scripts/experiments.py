"""CLI for running canonical experiments aligned with course modules."""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Tuple

import numpy as np

from .controllers import (
    DiscreteStateFeedback,
    KalmanFilter,
    MRACController,
    PIDController,
    discrete_lqr,
    discrete_nbar,
)
from .dynamics import LinearTimeInvariantSystem, mass_spring_damper
from .simulation import SimulationResult, simulate_discrete_system
from .visualization import plot_response


@dataclass
class Experiment:
    description: str
    runner: Callable[..., SimulationResult]


def _msd_pid(*, return_logs: bool = False) -> SimulationResult:
    system = mass_spring_damper()
    ts = 0.005
    dsys = system.discretize(ts)
    pid = PIDController(kp=180.0, ki=400.0, kd=1.0, ts=ts, u_min=-7.0, u_max=7.0)

    if return_logs:
        # No additional logs are recorded for the PID baseline.
        pass

    def controller(reference: float, measurement: float, _t: float) -> float:
        return pid.step(reference, measurement)

    result = simulate_discrete_system(
        dsys.A,
        dsys.B,
        dsys.C,
        controller,
        ts=ts,
        steps=4000,
        reference=lambda k: 0.05 if k > 200 else 0.0,
        measurement_noise_std=5e-4,
        quantization=5e-5,
        actuator_limits=(-7.0, 7.0),
    )
    return result


def _msd_lqr(*, return_logs: bool = False) -> SimulationResult:
    system = mass_spring_damper()
    ts = 0.01
    dsys = system.discretize(ts)
    Q = np.diag([200.0, 20.0])
    R = np.array([[0.5]])
    K = discrete_lqr(dsys.A, dsys.B, Q, R)
    nbar = discrete_nbar(dsys.A, dsys.B, dsys.C, K)
    controller = DiscreteStateFeedback(K, Nbar=nbar)

    if return_logs:
        # No auxiliary logs for the pure state-feedback case.
        pass

    state_estimate = np.zeros((2, 1))
    prev_measurement = 0.0

    def ctrl(reference: float, measurement: float, _t: float) -> float:
        nonlocal prev_measurement, state_estimate
        velocity = (measurement - prev_measurement) / ts
        prev_measurement = measurement
        state_estimate[0, 0] = measurement
        state_estimate[1, 0] = velocity
        return controller.step(state_estimate, reference)

    return simulate_discrete_system(
        dsys.A,
        dsys.B,
        dsys.C,
        ctrl,
        ts=ts,
        steps=3200,
        reference=lambda k: 0.1 if k > 100 else 0.0,
        measurement_noise_std=1e-3,
        quantization=1e-4,
        actuator_limits=(-6.0, 6.0),
)


def _msd_kf_lqr(*, return_logs: bool = False) -> SimulationResult:
    system = mass_spring_damper()
    ts = 0.01
    dsys = system.discretize(ts)
    Q = np.diag([150.0, 10.0])
    R = np.array([[0.4]])
    K = discrete_lqr(dsys.A, dsys.B, Q, R)
    nbar = discrete_nbar(dsys.A, dsys.B, dsys.C, K)
    kf = KalmanFilter(
        Ad=dsys.A,
        Bd=dsys.B,
        C=dsys.C,
        Qn=np.diag([1e-4, 1e-3]),
        Rn=np.array([[1e-4]]),
        xhat=np.zeros((2, 1)),
        P=np.eye(2) * 1e-2,
    )
    sf = DiscreteStateFeedback(K, Nbar=nbar)

    last_control = 0.0
    innovation_log: List[Tuple[float, float, float]] = []

    def ctrl(reference: float, measurement: float, _t: float) -> float:
        nonlocal last_control
        kf.predict(last_control)
        if return_logs:
            S = float(kf.C @ kf.P @ kf.C.T + kf.Rn)
        innovation = kf.update(measurement)
        if return_logs:
            innovation_log.append((_t, float(innovation), S))
        last_control = sf.step(kf.xhat, reference)
        return last_control

    result = simulate_discrete_system(
        dsys.A,
        dsys.B,
        dsys.C,
        ctrl,
        ts=ts,
        steps=3600,
        reference=lambda k: 0.08 if k > 150 else 0.0,
        measurement_noise_std=3e-3,
        quantization=2e-4,
        actuator_limits=(-5.0, 5.0),
    )
    if return_logs:
        result.metadata["innovation_log"] = np.asarray(innovation_log)
    return result


def _first_order_mrac(*, return_logs: bool = False) -> SimulationResult:
    from .dynamics import first_order_plant

    system = first_order_plant(a=-0.4, b=1.0)
    ts = 0.02
    dsys = system.discretize(ts)
    mrac = MRACController(a_m=0.97, b_m=0.03, gamma=0.2)

    def ctrl(reference: float, measurement: float, _t: float) -> float:
        return mrac.step(reference, measurement)

    return simulate_discrete_system(
        dsys.A,
        dsys.B,
        dsys.C,
        ctrl,
        ts=ts,
        steps=2500,
        reference=lambda k: 1.0 if k > 50 else 0.0,
        measurement_noise_std=5e-3,
        quantization=None,
        actuator_limits=(-10.0, 10.0),
    )


EXPERIMENTS: Dict[str, Experiment] = {
    "pid": Experiment("Mass-spring-damper with PID", _msd_pid),
    "lqr": Experiment("Mass-spring-damper with LQR", _msd_lqr),
    "kf_lqr": Experiment("LQR with Kalman state estimation", _msd_kf_lqr),
    "mrac": Experiment("Model reference adaptive control", _first_order_mrac),
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Run course simulation experiments")
    parser.add_argument("experiment", choices=EXPERIMENTS.keys(), help="Experiment identifier")
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="Skip plotting (useful for headless environments).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional path to save the response plot as a PNG.",
    )
    args = parser.parse_args()

    np.random.seed(0)
    result = EXPERIMENTS[args.experiment].runner()
    metrics = result.performance_metrics()
    print(f"Experiment: {EXPERIMENTS[args.experiment].description}")
    for key, value in metrics.items():
        print(f"  {key.replace('_', ' ').title()}: {value:.4f}")

    if not args.no_plot:
        plot_response(result, title=EXPERIMENTS[args.experiment].description, save_path=args.output)


if __name__ == "__main__":
    main()
