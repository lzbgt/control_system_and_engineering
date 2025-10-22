"""Plotting helpers for simulation outputs."""
from __future__ import annotations

import contextlib
from pathlib import Path
from typing import Optional

try:
    import matplotlib.pyplot as plt
except Exception as exc:  # pragma: no cover - optional dependency guard
    raise ImportError(
        "matplotlib is required for plotting. Install it via `pip install matplotlib`."
    ) from exc

from .simulation import SimulationResult


def plot_response(result: SimulationResult, title: str = "Response", save_path: Optional[Path] = None) -> None:
    plt.figure(figsize=(8, 4))
    plt.plot(result.time, result.reference, label="Reference", linestyle="--")
    plt.plot(result.time, result.outputs, label="Output")
    plt.xlabel("Time [s]")
    plt.ylabel("Output")
    plt.title(title)
    plt.grid(True)
    plt.legend()

    plt.figure(figsize=(8, 3))
    plt.plot(result.time, result.control)
    plt.xlabel("Time [s]")
    plt.ylabel("Control")
    plt.title("Control Effort")
    plt.grid(True)

    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        with contextlib.suppress(Exception):
            plt.savefig(save_path)

    plt.show()
