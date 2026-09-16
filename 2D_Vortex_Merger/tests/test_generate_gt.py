from __future__ import annotations

import importlib.util
import math
import sys
from pathlib import Path

import numpy as np


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "generate_gt.py"
SPEC = importlib.util.spec_from_file_location("vortex_merger_generate_gt", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_initial_condition_is_periodic_compatible_and_filtered() -> None:
    config = MODULE.MergerConfig(grid_size=64, final_time=0.1, output_frames=2)
    solver, state, metadata = MODULE.make_solver(config)
    omega, _, u, v = solver.fields_from_hat(state)
    assert abs(float(np.mean(omega))) < 1e-14
    assert np.all(state[~solver.dealias] == 0.0)
    assert abs(metadata["measured_reynolds_number"] - 350.0) < 1e-12
    assert np.max(np.abs(u)) > 0.0
    assert np.max(np.abs(v)) > 0.0


def test_taylor_green_decay_and_incompressibility() -> None:
    n = 48
    viscosity = 0.02
    dt = 0.01
    final_time = 0.2
    axis = np.arange(n) * 2.0 * math.pi / n
    x, y = np.meshgrid(axis, axis, indexing="xy")
    omega0 = 2.0 * np.sin(x) * np.sin(y)
    solver = MODULE.VorticityDNS(n, 2.0 * math.pi, viscosity, dt)
    state = solver.filtered_hat(omega0)
    for _ in range(round(final_time / dt)):
        state = solver.step(state)
    omega, _, u, v = solver.fields_from_hat(state)
    exact = np.exp(-2.0 * viscosity * final_time) * omega0
    relative_error = np.linalg.norm(omega - exact) / np.linalg.norm(exact)
    assert relative_error < 2e-11
    divergence_hat = 1j * solver.kx * np.fft.fft2(u) + 1j * solver.ky * np.fft.fft2(v)
    assert np.max(np.abs(np.fft.ifft2(divergence_hat))) < 1e-12


def test_short_merger_run_dissipates_energy_and_enstrophy() -> None:
    config = MODULE.MergerConfig(
        grid_size=48,
        final_time=0.06,
        time_step=0.01,
        output_frames=4,
    )
    _, _, records, _ = MODULE.integrate(config, progress=False)
    energy = np.asarray([record["kinetic_energy"] for record in records])
    enstrophy = np.asarray([record["enstrophy"] for record in records])
    assert np.all(np.diff(energy) < 0.0)
    assert np.all(np.diff(enstrophy) < 0.0)
    assert max(record["max_abs_divergence"] for record in records) < 1e-12
