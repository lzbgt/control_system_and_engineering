"""Tutorial utilities for generating figures referenced in the lecture notes."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
from scipy import signal

from .controllers import (
    PIDController,
    MRACController,
    KalmanFilter,
    discrete_lqr,
    discrete_nbar,
)
from .dynamics import mass_spring_damper
from .experiments import EXPERIMENTS
from .simulation import simulate_discrete_system


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def tutorial_modeling(output_dir: Path) -> None:
    """Generate continuous vs. discrete step response for the MSD plant."""
    system = mass_spring_damper(m=1.0, c=0.4, k=4.0)
    Ts = 0.005
    discrete = system.discretize(Ts)

    # Continuous-time step response using SciPy
    cont_tf = signal.TransferFunction([1.0], [1.0, 0.4, 4.0])
    t_cont, y_cont = signal.step(cont_tf, T=np.linspace(0, 4, 2000))

    # Discrete-time response via simulation harness
    step = lambda k: 0.05 if k > 0 else 0.0
    result = simulate_discrete_system(
        discrete.A,
        discrete.B,
        discrete.C,
        controller_step=lambda r, y, t: 0.0,
        ts=Ts,
        steps=800,
        reference=step,
    )

    plt.figure(figsize=(6, 4))
    plt.plot(t_cont, y_cont, label="Continuous step")
    plt.plot(result.time, result.outputs, label="Discrete step", linestyle="--")
    plt.xlabel("Time [s]")
    plt.ylabel("Position [m]")
    plt.title("Mass–Spring–Damper Step Response")
    plt.grid(True)
    plt.legend()
    _ensure_dir(output_dir)
    plt.savefig(output_dir / "modeling_step.png", dpi=300, bbox_inches="tight")
    plt.close()


def tutorial_time_domain(output_dir: Path) -> None:
    """Generate PID closed-loop response plots for multiple tuning styles."""
    system = mass_spring_damper(m=1.0, c=0.4, k=4.0)
    Ts = 0.005
    dsys = system.discretize(Ts)

    cases = {
        "aggressive": PIDController(kp=200.0, ki=400.0, kd=0.8, ts=Ts, u_min=-7.0, u_max=7.0),
        "balanced": PIDController(kp=120.0, ki=180.0, kd=2.0, ts=Ts, u_min=-7.0, u_max=7.0),
        "conservative": PIDController(kp=80.0, ki=80.0, kd=3.0, ts=Ts, u_min=-7.0, u_max=7.0),
    }

    # Remove legacy figures
    for legacy in ["time_pid_response.png", "time_pid_control.png"]:
        legacy_path = output_dir / legacy
        if legacy_path.exists():
            legacy_path.unlink()

    responses: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    metrics: dict[str, dict[str, float]] = {}

    for label, controller in cases.items():
        controller_step = lambda r, y, t, ctrl=controller: ctrl.step(r, y)  # noqa: E731
        pid_result = simulate_discrete_system(
            dsys.A,
            dsys.B,
            dsys.C,
            controller_step=controller_step,
            ts=Ts,
            steps=4000,
            reference=lambda k: 0.05 if k > 200 else 0.0,
            actuator_limits=(-7.0, 7.0),
        )
        responses[label] = (pid_result.time, pid_result.outputs, pid_result.control)
        metrics[label] = pid_result.performance_metrics()
        print(f"[{label}] steady_state_error={metrics[label]['steady_state_error']:.4e}, "
              f"percent_overshoot={metrics[label]['percent_overshoot']:.2f}")

    _ensure_dir(output_dir)

    plt.figure(figsize=(6.5, 4))
    first_time = next(iter(responses.values()))[0]
    plt.plot(first_time, 0.05 * np.ones_like(first_time), linestyle="--", color="gray", label="Reference")
    for label, (time, output, _) in responses.items():
        plt.plot(time, output, label=label.title())
    plt.xlabel("Time [s]")
    plt.ylabel("Position [m]")
    plt.title("PID Step Responses Under Different Tunings")
    plt.grid(True)
    plt.legend()
    plt.savefig(output_dir / "time_pid_responses.png", dpi=300, bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(6.5, 3))
    for label, (time, _, control_signal) in responses.items():
        plt.plot(time, control_signal, label=label.title())
    plt.xlabel("Time [s]")
    plt.ylabel("Control Input")
    plt.title("PID Control Effort Comparison")
    plt.grid(True)
    plt.legend()
    plt.savefig(output_dir / "time_pid_controls.png", dpi=300, bbox_inches="tight")
    plt.close()


def tutorial_frequency(output_dir: Path) -> None:
    """Generate Bode and Nyquist plots for the MSD plant."""
    system = mass_spring_damper(m=1.0, c=0.4, k=4.0)
    omega = np.logspace(-1, 2, 600)
    A, B, C = system.A, system.B, system.C

    mag = np.zeros_like(omega)
    phase = np.zeros_like(omega)
    plant = mass_spring_damper(m=1.0, c=0.4, k=4.0)
    _ensure_dir(output_dir)

    def plant_freq_response(s: complex) -> complex:
        return 1.0 / (s**2 + 0.4 * s + 4.0)

    # Controllers for overlay
    Kp, Ki, Kd = 40.0, 80.0, 2.0
    lead_gain, lead_z, lead_p = 6.5, 1.3, 7.2

    responses = {
        "Plant only": np.array([plant_freq_response(1j * w) for w in omega]),
        "Lead compensated": np.array([
            plant_freq_response(1j * w) * lead_gain * (1j * w + lead_z) / (1j * w + lead_p)
            for w in omega
        ]),
        "PID compensated": np.array([
            plant_freq_response(1j * w) * (Kp + Ki / (1j * w) + Kd * 1j * w)
            for w in omega
        ]),
    }

    plt.figure(figsize=(6.5, 4))
    for label, data in responses.items():
        plt.semilogx(omega, 20 * np.log10(np.abs(data)), label=label)
    plt.xlabel("Frequency [rad/s]")
    plt.ylabel("Magnitude [dB]")
    plt.title("Bode Magnitude Comparisons")
    plt.grid(True, which="both")
    plt.legend()
    plt.savefig(output_dir / "frequency_bode_overlay.png", dpi=300, bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(6.5, 4))
    for label, data in responses.items():
        plt.semilogx(omega, np.degrees(np.angle(data)), label=label)
    plt.xlabel("Frequency [rad/s]")
    plt.ylabel("Phase [deg]")
    plt.title("Bode Phase Comparisons")
    plt.grid(True, which="both")
    plt.legend()
    plt.savefig(output_dir / "frequency_phase_overlay.png", dpi=300, bbox_inches="tight")
    plt.close()

    # Nyquist overlays
    jw = np.logspace(-2, 2, 800)
    plt.figure(figsize=(5.5, 5.5))
    # recompute for overlay to maintain frequency mapping
    nyquist_curves = {
        "Plant only": np.array([plant_freq_response(1j * w) for w in jw]),
        "Lead compensated": np.array([
            plant_freq_response(1j * w) * lead_gain * (1j * w + lead_z) / (1j * w + lead_p)
            for w in jw
        ]),
        "PID compensated": np.array([
            plant_freq_response(1j * w) * (Kp + Ki / (1j * w) + Kd * 1j * w)
            for w in jw
        ]),
    }

    for label, curve in nyquist_curves.items():
        plt.plot(curve.real, curve.imag, label=label)
        plt.plot(curve.real, -curve.imag, linestyle="--", color="gray", linewidth=0.5)
    plt.scatter([-1], [0], marker="x", color="red", label="-1 point")
    plt.xlabel("Re")
    plt.ylabel("Im")
    plt.title("Nyquist Comparisons")
    plt.axis("equal")
    plt.grid(True)
    plt.legend(loc="best")
    plt.savefig(output_dir / "frequency_nyquist_overlay.png", dpi=300, bbox_inches="tight")
    plt.close()

    # Root locus for L(s) = K / (s(s + 4))
    num = np.array([1.0])
    den = np.array([1.0, 4.0, 0.0])
    zeros = np.roots(num)
    gains = np.logspace(-3, 3, 500)
    pole_trajectories = np.array([np.roots(np.polyadd(den, K * num)) for K in gains])
    # Sort each row so that branches remain continuous as gain varies
    pole_trajectories.sort(axis=1)

    plt.figure(figsize=(6, 5))
    for branch in range(pole_trajectories.shape[1]):
        branch_poles = pole_trajectories[:, branch]
        plt.plot(branch_poles.real, branch_poles.imag, label=f"Branch {branch + 1}")
    plant_poles = np.roots(den)
    plt.scatter(plant_poles.real, plant_poles.imag, marker="x", color="black", label="Open-loop poles")
    if zeros.size > 0:
        plt.scatter(zeros.real, zeros.imag, marker="o", facecolors="none", edgecolors="black", label="Open-loop zeros")
    plt.axhline(0, color="gray", linewidth=0.5)
    plt.axvline(0, color="gray", linewidth=0.5)
    plt.xlabel("Re")
    plt.ylabel("Im")
    plt.title("Root Locus for $K/(s(s+4))$")
    plt.grid(True)
    plt.legend(loc="best")
    plt.savefig(output_dir / "frequency_root_locus.png", dpi=300, bbox_inches="tight")
    plt.close()


def tutorial_modern(output_dir: Path) -> None:
    """Generate LQR and Kalman-filter demonstrations for modern control."""
    system = mass_spring_damper(m=1.0, c=0.4, k=4.0)
    Ts = 0.01
    dsys = system.discretize(Ts)

    scenarios = {
        "tracking": (np.diag([150.0, 10.0]), np.array([[0.4]])),
        "energy": (np.diag([20.0, 2.0]), np.array([[1.2]])),
    }

    responses: dict[str, dict[str, np.ndarray]] = {}

    steps = 3500
    time = np.arange(steps) * Ts
    reference = np.array([0.05 if k > 200 else 0.0 for k in range(steps)])

    for label, (Q, R) in scenarios.items():
        K = discrete_lqr(dsys.A, dsys.B, Q, R)
        Nbar = discrete_nbar(dsys.A, dsys.B, dsys.C, K)
        nbar_scalar = float(Nbar)
        x = np.zeros((dsys.A.shape[0], 1))
        outputs = np.zeros(steps)
        controls = np.zeros(steps)
        for k in range(steps):
            r = reference[k]
            u = -float((K @ x).item()) + nbar_scalar * r
            controls[k] = u
            x = dsys.A @ x + dsys.B * u
            outputs[k] = float((dsys.C @ x).item())
        responses[label] = {
            "output": outputs,
            "control": controls,
        }

    _ensure_dir(output_dir)

    plt.figure(figsize=(6.5, 4))
    plt.plot(time, reference, linestyle="--", color="gray", label="Reference")
    for label, data in responses.items():
        plt.plot(time, data["output"], label=f"LQR ({label})")
    plt.xlabel("Time [s]")
    plt.ylabel("Position [m]")
    plt.title("LQR Step Responses for Different Weightings")
    plt.grid(True)
    plt.legend()
    plt.savefig(output_dir / "modern_lqr_responses.png", dpi=300, bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(6.5, 3))
    for label, data in responses.items():
        plt.plot(time, data["control"], label=f"LQR ({label})")
    plt.xlabel("Time [s]")
    plt.ylabel("Control Input")
    plt.title("LQR Control Effort Comparison")
    plt.grid(True)
    plt.legend()
    plt.savefig(output_dir / "modern_lqr_controls.png", dpi=300, bbox_inches="tight")
    plt.close()

    # Kalman filter demonstration
    Qn = np.diag([1e-4, 1e-3])
    Rn = np.array([[4e-4]])
    K = discrete_lqr(dsys.A, dsys.B, np.diag([150.0, 10.0]), np.array([[0.4]]))
    Nbar = discrete_nbar(dsys.A, dsys.B, dsys.C, K)
    nbar_scalar = float(Nbar)
    kf = KalmanFilter(
        Ad=dsys.A,
        Bd=dsys.B,
        C=dsys.C,
        Qn=Qn,
        Rn=Rn,
        xhat=np.zeros((2, 1)),
        P=np.eye(2) * 1e-2,
    )

    x_true = np.zeros((2, 1))
    outputs = np.zeros(steps)
    estimates = np.zeros(steps)
    controls = np.zeros(steps)

    for k in range(steps):
        r = reference[k]
        measurement = float((dsys.C @ x_true).item() + np.random.normal(scale=np.sqrt(Rn[0, 0])))
        kf.predict(float(controls[k - 1]) if k > 0 else 0.0)
        kf.update(measurement)
        u = -float((K @ kf.xhat).item()) + nbar_scalar * r
        controls[k] = u
        x_true = dsys.A @ x_true + dsys.B * u + np.random.multivariate_normal(np.zeros(2), Qn).reshape(-1, 1)
        outputs[k] = float((dsys.C @ x_true).item())
        estimates[k] = float(kf.xhat[0, 0])

    plt.figure(figsize=(6.5, 4))
    plt.plot(time, reference, linestyle="--", color="gray", label="Reference")
    plt.plot(time, outputs, label="True output")
    plt.plot(time, estimates, label="Kalman estimate", linewidth=1.0)
    plt.xlabel("Time [s]")
    plt.ylabel("Position [m]")
    plt.title("Kalman-Filtered LQR Tracking")
    plt.grid(True)
    plt.legend()
    plt.savefig(output_dir / "modern_kalman_tracking.png", dpi=300, bbox_inches="tight")
    plt.close()


def tutorial_adaptive(output_dir: Path) -> None:
    """Generate MRAC tracking and parameter adaptation plots."""
    Ts = 0.02
    steps = 2500
    reference = np.array([1.0 if k > 50 else 0.0 for k in range(steps)])

    # Plant: y_{k+1} = a y_k + b u_k
    a, b = -0.4, 1.0
    a_m, b_m = 0.97, 0.03
    controller = MRACController(a_m=a_m, b_m=b_m, gamma=0.2)

    y = 0.0
    y_model = 0.0
    outputs = np.zeros(steps)
    model_outputs = np.zeros(steps)
    params = np.zeros((steps, 2))

    for k in range(steps):
        r = reference[k]
        u = controller.step(r, y)
        y_next = a * y + b * u
        y_model = a_m * y_model + b_m * r
        outputs[k] = y_next
        model_outputs[k] = y_model
        params[k] = controller.theta.reshape(-1)
        y = y_next

    _ensure_dir(output_dir)

    time = np.arange(steps) * Ts

    plt.figure(figsize=(6.5, 4))
    plt.plot(time, reference, linestyle="--", color="gray", label="Reference")
    plt.plot(time, outputs, label="Plant output")
    plt.plot(time, model_outputs, label="Reference model")
    plt.xlabel("Time [s]")
    plt.ylabel("Output")
    plt.title("MRAC Tracking Performance")
    plt.grid(True)
    plt.legend()
    plt.savefig(output_dir / "adaptive_tracking.png", dpi=300, bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(6.5, 3.5))
    plt.plot(time, params[:, 0], label=r"$\theta_1$")
    plt.plot(time, params[:, 1], label=r"$\theta_2$")
    plt.xlabel("Time [s]")
    plt.ylabel("Parameter value")
    plt.title("MRAC Parameter Adaptation")
    plt.grid(True)
    plt.legend()
    plt.savefig(output_dir / "adaptive_parameters.png", dpi=300, bbox_inches="tight")
    plt.close()


def tutorial_learning(output_dir: Path) -> None:
    """Generate figures for data-driven identification and PDE adaptive control."""
    _ensure_dir(output_dir)

    # Recursive least squares parameter identification
    Ts = 0.02
    steps = 600
    time = np.arange(steps) * Ts
    true_theta = np.array([[0.7], [0.3]])  # coefficients for ARX model y_{k+1} = a y_k + b u_k
    u = 0.6 * np.sin(0.8 * time) + 0.4 * np.sin(0.2 * time)
    y = np.zeros(steps)
    for k in range(1, steps):
        y[k] = true_theta[0, 0] * y[k - 1] + true_theta[1, 0] * u[k - 1] + 0.02 * np.random.randn()

    theta = np.zeros((2, 1))
    P = np.eye(2) * 100.0
    theta_history = np.zeros((steps, 2))
    for k in range(1, steps):
        phi = np.array([[y[k - 1]], [u[k - 1]]])
        K_gain = P @ phi / (1.0 + phi.T @ P @ phi)
        error = y[k] - float((phi.T @ theta).item())
        theta = theta + K_gain * error
        P = P - K_gain @ phi.T @ P
        theta_history[k] = theta.ravel()

    plt.figure(figsize=(6.5, 3.5))
    plt.plot(time, theta_history[:, 0], label=r"$\hat{a}$")
    plt.plot(time, theta_history[:, 1], label=r"$\hat{b}$")
    plt.axhline(true_theta[0, 0], color="gray", linestyle="--", label=r"$a$ true")
    plt.axhline(true_theta[1, 0], color="gray", linestyle=":", label=r"$b$ true")
    plt.xlabel("Time [s]")
    plt.ylabel("Parameter estimate")
    plt.title("Recursive Least Squares Identification")
    plt.grid(True)
    plt.legend()
    plt.savefig(output_dir / "learning_rls_parameters.png", dpi=300, bbox_inches="tight")
    plt.close()

    # Simple PDE (1D heat equation) adaptive boundary control demonstration
    length = 1.0
    nx = 40
    dx = length / (nx - 1)
    alpha = 0.02  # diffusion coefficient
    dt = 0.25 * dx**2 / alpha
    steps_pde = 400
    x_grid = np.linspace(0, length, nx)

    # Desired steady state is zero temperature
    state = 0.5 * np.sin(np.pi * x_grid)
    adaptation_gain = 2.0
    gamma = 0.05
    theta_hat = 0.0
    energy = np.zeros(steps_pde)

    for k in range(steps_pde):
        energy[k] = np.trapezoid(state**2, x_grid)
        # discretized Laplacian with Dirichlet boundary at x=0 and adaptive Neumann at x=1
        laplacian = np.zeros_like(state)
        laplacian[1:-1] = (state[2:] - 2 * state[1:-1] + state[:-2]) / dx**2
        # boundary control at x=1 uses adaptive gain
        boundary_error = state[-1]
        u = -theta_hat * boundary_error
        # apply Neumann condition via finite-difference (one-sided)
        laplacian[-1] = (state[-2] - state[-1]) / dx**2 + u / dx
        state = state + dt * alpha * laplacian
        theta_hat += gamma * boundary_error * state[-1] * dt

    plt.figure(figsize=(6.5, 3.5))
    plt.plot(np.arange(steps_pde) * dt, energy)
    plt.xlabel("Time [s]")
    plt.ylabel("Energy")
    plt.title("Adaptive Boundary Control for Heat Equation")
    plt.grid(True)
    plt.savefig(output_dir / "learning_pde_energy.png", dpi=300, bbox_inches="tight")
    plt.close()


def tutorial_digital(output_dir: Path) -> None:
    """Generate figures illustrating digital control effects (quantization, delay)."""
    _ensure_dir(output_dir)

    system = mass_spring_damper(m=1.0, c=0.4, k=4.0)
    Ts = 0.01
    dsys = system.discretize(Ts)

    scenarios = {
        "ideal": {"quantization": None, "delay": 0, "noise": 1e-3},
        "quantized": {"quantization": 2e-4, "delay": 0, "noise": 1e-3},
        "delayed": {"quantization": None, "delay": 2, "noise": 1e-3},
    }

    pid = PIDController(kp=180.0, ki=400.0, kd=1.0, ts=Ts, u_min=-7.0, u_max=7.0)

    time = np.arange(3000) * Ts
    reference = np.array([0.05 if k > 200 else 0.0 for k in range(3000)])

    outputs = {}
    controls = {}

    for label, cfg in scenarios.items():
        controller = PIDController(kp=pid.kp, ki=pid.ki, kd=pid.kd, ts=pid.ts, u_min=pid.u_min, u_max=pid.u_max)
        x = np.zeros((dsys.A.shape[0], 1))
        measurements = np.zeros(3000)
        control_seq = np.zeros(3000)
        output_seq = np.zeros(3000)
        delay_buffer = [0.0] * (cfg["delay"] + 1)

        for k in range(3000):
            r = reference[k]
            y_true = float((dsys.C @ x).item()) + np.random.normal(scale=cfg["noise"])
            delay_buffer.append(y_true)
            y_delayed = delay_buffer.pop(0)
            if cfg["quantization"] is not None:
                y_delayed = np.round(y_delayed / cfg["quantization"]) * cfg["quantization"]
            u = controller.step(r, y_delayed)
            u = float(np.clip(u, -7.0, 7.0))
            x = dsys.A @ x + dsys.B * u
            measurements[k] = y_delayed
            control_seq[k] = u
            output_seq[k] = y_true

        outputs[label] = output_seq
        controls[label] = control_seq

    plt.figure(figsize=(6.5, 4))
    plt.plot(time, reference, linestyle="--", color="gray", label="Reference")
    for label, data in outputs.items():
        plt.plot(time, data, label=label.title())
    plt.xlabel("Time [s]")
    plt.ylabel("Position [m]")
    plt.title("Digital Effects on PID Tracking")
    plt.grid(True)
    plt.legend()
    plt.savefig(output_dir / "digital_pid_responses.png", dpi=300, bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(6.5, 3))
    for label, data in controls.items():
        plt.plot(time, data, label=label.title())
    plt.xlabel("Time [s]")
    plt.ylabel("Control Input")
    plt.title("Control Effort Under Quantization/Delay")
    plt.grid(True)
    plt.legend()
    plt.savefig(output_dir / "digital_pid_controls.png", dpi=300, bbox_inches="tight")
    plt.close()

    # Sampling study: compare Ts choices
    sampling_times = [0.02, 0.01, 0.002]
    spectra = {}
    freq = np.fft.rfftfreq(1024, d=Ts)

    for Ts_candidate in sampling_times:
        dsys_cand = system.discretize(Ts_candidate)
        controller = PIDController(kp=pid.kp, ki=pid.ki, kd=pid.kd, ts=Ts_candidate, u_min=pid.u_min, u_max=pid.u_max)

        def step_fn(r: float, y: float, t: float, ctrl=controller) -> float:
            return ctrl.step(r, y)

        sim = simulate_discrete_system(
            dsys_cand.A,
            dsys_cand.B,
            dsys_cand.C,
            controller_step=step_fn,
            ts=Ts_candidate,
            steps=2000,
            reference=lambda k: 0.05 if k > 200 else 0.0,
            measurement_noise_std=0.0,
        )
        spectrum = np.abs(np.fft.rfft(sim.outputs[:1024]))
        spectra[f"Ts={Ts_candidate:.3f}s"] = spectrum / spectrum.max()

    plt.figure(figsize=(6.5, 3.5))
    for label, spec in spectra.items():
        plt.semilogy(freq, spec, label=label)
    plt.xlabel("Frequency [Hz]")
    plt.ylabel("Normalised magnitude")
    plt.title("Sampling Effects on Output Spectrum")
    plt.grid(True, which="both")
    plt.legend()
    plt.savefig(output_dir / "digital_sampling_spectrum.png", dpi=300, bbox_inches="tight")
    plt.close()


def tutorial_adrc(output_dir: Path) -> None:
    """Generate figures for linear ADRC on the mass–spring–damper system."""
    _ensure_dir(output_dir)

    system = mass_spring_damper(m=1.0, c=0.4, k=4.0)
    Ts = 0.005
    dsys = system.discretize(Ts)
    reference = lambda k: 0.05 if k > 200 else 0.0
    steps = 4000
    time = np.arange(steps) * Ts

    scenarios = {
        "fast": {"omega_c": 120.0, "eso_bw": 300.0},
        "medium": {"omega_c": 80.0, "eso_bw": 200.0},
        "slow": {"omega_c": 40.0, "eso_bw": 120.0},
    }

    outputs = {}
    controls = {}

    def ladrc_step_factory(omega_c: float, eso_bw: float) -> Callable[[float, float, float], float]:
        b0 = 1.0
        kp = omega_c**2
        kd = 2 * omega_c
        beta1 = 3 * eso_bw
        beta2 = 3 * eso_bw**2
        beta3 = eso_bw**3
        z = np.zeros(3)
        u_last = 0.0

        def step(ref: float, y: float, t: float) -> float:
            nonlocal z, u_last
            e = z[0] - y
            z1_dot = z[1] - beta1 * e
            z2_dot = z[2] - beta2 * e + b0 * u_last
            z3_dot = -beta3 * e
            z = z + Ts * np.array([z1_dot, z2_dot, z3_dot])
            error = ref - z[0]
            v = kp * error - kd * z[1]
            u = (v - z[2]) / b0
            u = float(np.clip(u, -7.0, 7.0))
            u_last = u
            return u

        return step

    for label, params in scenarios.items():
        controller_step = ladrc_step_factory(params["omega_c"], params["eso_bw"])
        sim = simulate_discrete_system(
            dsys.A,
            dsys.B,
            dsys.C,
            controller_step=controller_step,
            ts=Ts,
            steps=steps,
            reference=reference,
        )
        outputs[label] = sim.outputs
        controls[label] = sim.control
        metrics = sim.performance_metrics()
        print(f"[adrc-{label}] steady_state_error={metrics['steady_state_error']:.3e}, percent_overshoot={metrics['percent_overshoot']:.2f}")

    plt.figure(figsize=(6.5, 4))
    plt.plot(time, [reference(k) for k in range(steps)], linestyle="--", color="gray", label="Reference")
    for label, data in outputs.items():
        plt.plot(time, data, label=label.title())
    plt.xlabel("Time [s]")
    plt.ylabel("Position [m]")
    plt.title("ADRC Step Responses with Different Bandwidths")
    plt.grid(True)
    plt.legend()
    plt.savefig(output_dir / "adrc_responses.png", dpi=300, bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(6.5, 3))
    for label, data in controls.items():
        plt.plot(time, data, label=label.title())
    plt.xlabel("Time [s]")
    plt.ylabel("Control Input")
    plt.title("ADRC Control Effort Comparison")
    plt.grid(True)
    plt.legend()
    plt.savefig(output_dir / "adrc_controls.png", dpi=300, bbox_inches="tight")
    plt.close()

    # Observer estimation error for the medium case
    controller_step = ladrc_step_factory(scenarios["medium"]["omega_c"], scenarios["medium"]["eso_bw"])
    z = np.zeros(3)
    u_last = 0.0
    estimation_error = np.zeros((steps, 2))
    x = np.zeros((dsys.A.shape[0], 1))
    for k in range(steps):
        y = float((dsys.C @ x).item())
        e = z[0] - y
        b0 = 1.0
        beta1 = 3 * scenarios["medium"]["eso_bw"]
        beta2 = 3 * scenarios["medium"]["eso_bw"]**2
        beta3 = scenarios["medium"]["eso_bw"]**3
        z1_dot = z[1] - beta1 * e
        z2_dot = z[2] - beta2 * e + b0 * u_last
        z3_dot = -beta3 * e
        z = z + Ts * np.array([z1_dot, z2_dot, z3_dot])
        error = (0.05 if k > 200 else 0.0) - z[0]
        v = scenarios["medium"]["omega_c"]**2 * error - 2 * scenarios["medium"]["omega_c"] * z[1]
        u = (v - z[2]) / b0
        u = float(np.clip(u, -7.0, 7.0))
        u_last = u
        x = dsys.A @ x + dsys.B * u
        estimation_error[k, 0] = z[0] - y
        estimation_error[k, 1] = z[1] - float(x[1, 0])

    plt.figure(figsize=(6.5, 3.5))
    plt.plot(time, estimation_error[:, 0], label=r"$z_1 - y$")
    plt.plot(time, estimation_error[:, 1], label=r"$z_2 - \dot{y}$")
    plt.xlabel("Time [s]")
    plt.ylabel("Estimation error")
    plt.title("ESO Estimation Error (Medium Bandwidth)")
    plt.grid(True)
    plt.legend()
    plt.savefig(output_dir / "adrc_eso_errors.png", dpi=300, bbox_inches="tight")
    plt.close()
def tutorial_classical(output_dir: Path) -> None:
    """Generate lead-compensated response plots for a position control example."""
    # Plant P(s) = 1 / (s (s + 4))
    P_num = [1.0]
    P_den = [1.0, 4.0, 0.0]

    # Lead compensator parameters
    K = 6.5
    z = 1.3
    p = 7.2

    C_num = K * np.array([1.0, z])
    C_den = np.array([1.0, p])

    L_num = np.polymul(C_num, P_num)
    L_den = np.polymul(C_den, P_den)

    # Closed-loop transfer function T(s) = L(s) / (1 + L(s))
    T_num = L_num
    T_den = np.polyadd(L_den, L_num)
    sys = signal.TransferFunction(T_num, T_den)
    t = np.linspace(0, 6, 1000)
    tout, yout = signal.step(sys, T=t)

    plt.figure(figsize=(6, 4))
    plt.plot(tout, yout, label="Lead-compensated output")
    plt.xlabel("Time [s]")
    plt.ylabel("Position")
    plt.title("Lead-Compensated Step Response")
    plt.grid(True)
    plt.legend()
    _ensure_dir(output_dir)
    plt.savefig(output_dir / "classical_step.png", dpi=300, bbox_inches="tight")
    plt.close()


TUTORIALS: dict[str, Callable[[Path], None]] = {
    "modeling": tutorial_modeling,
    "time": tutorial_time_domain,
    "frequency": tutorial_frequency,
    "modern": tutorial_modern,
    "adaptive": tutorial_adaptive,
    "learning": tutorial_learning,
    "digital": tutorial_digital,
    "adrc": tutorial_adrc,
    "classical": tutorial_classical,
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate tutorial figures referenced in the notes")
    parser.add_argument(
        "tutorial",
        choices=TUTORIALS.keys(),
        help="Tutorial identifier (e.g., 'frequency').",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("figures"),
        help="Directory where figures will be saved.",
    )
    args = parser.parse_args()

    TUTORIALS[args.tutorial](args.output_dir)
    print(f"Tutorial '{args.tutorial}' figures saved to {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
