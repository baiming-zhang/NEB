"""Generate a dealiased Fourier DNS for a pair of merging Gaussian vortices."""
from __future__ import annotations
import argparse
import json
import math
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
import h5py
import matplotlib
import numpy as np
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
TWO_PI = 2.0 * math.pi

@dataclass(frozen=True)
class MergerConfig:
    grid_size: int = 128
    domain_length: float = TWO_PI
    reynolds_number: float = 350.0
    vortex_sigma: float = 0.1 * math.pi
    vortex_circulation: float = 0.2 * math.pi
    vortex_separation: float = 0.59
    final_time: float = 6.0 * math.pi
    time_step: float = 0.005
    output_frames: int = 61

class VorticityDNS:
    """Fourier pseudo-spectral vorticity solver with ETDRK4 time stepping."""

    def __init__(self, n: int, length: float, viscosity: float, dt: float):
        if n < 12 or n % 2:
            raise ValueError('grid size must be an even integer >= 12')
        if viscosity <= 0.0 or dt <= 0.0:
            raise ValueError('viscosity and dt must be positive')
        self.n = n
        self.length = length
        self.viscosity = viscosity
        self.dt = dt
        self.dx = length / n
        wave = TWO_PI * np.fft.fftfreq(n, d=self.dx)
        self.kx, self.ky = np.meshgrid(wave, wave, indexing='xy')
        self.k2 = self.kx * self.kx + self.ky * self.ky
        self.inv_k2 = np.zeros_like(self.k2)
        nonzero = self.k2 > 0.0
        self.inv_k2[nonzero] = 1.0 / self.k2[nonzero]
        integer_mode = np.fft.fftfreq(n) * n
        mx, my = np.meshgrid(integer_mode, integer_mode, indexing='xy')
        cutoff = n / 3.0
        self.dealias = (np.abs(mx) < cutoff) & (np.abs(my) < cutoff)
        self.dealias[0, 0] = True
        linear = -viscosity * self.k2
        self._build_etdrk4_coefficients(linear)

    def _build_etdrk4_coefficients(self, linear: np.ndarray) -> None:
        """Stable contour-integral ETDRK4 coefficients (Kassam--Trefethen)."""
        h = self.dt
        self.e = np.exp(h * linear)
        self.e2 = np.exp(0.5 * h * linear)
        roots = np.exp(1j * math.pi * (np.arange(1, 33) - 0.5) / 32.0)
        lr = h * linear[..., None] + roots
        self.q = h * np.real(np.mean((np.exp(lr / 2.0) - 1.0) / lr, axis=-1))
        self.f1 = h * np.real(np.mean((-4.0 - lr + np.exp(lr) * (4.0 - 3.0 * lr + lr * lr)) / lr ** 3, axis=-1))
        self.f2 = h * np.real(np.mean((2.0 + lr + np.exp(lr) * (-2.0 + lr)) / lr ** 3, axis=-1))
        self.f3 = h * np.real(np.mean((-4.0 - 3.0 * lr - lr * lr + np.exp(lr) * (4.0 - lr)) / lr ** 3, axis=-1))

    def filtered_hat(self, vorticity: np.ndarray) -> np.ndarray:
        result = np.fft.fft2(vorticity)
        result[~self.dealias] = 0.0
        result[0, 0] = 0.0
        return result

    def fields_from_hat(self, omega_hat: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        psi_hat = omega_hat * self.inv_k2
        u_hat = 1j * self.ky * psi_hat
        v_hat = -1j * self.kx * psi_hat
        omega = np.fft.ifft2(omega_hat).real
        psi = np.fft.ifft2(psi_hat).real
        u = np.fft.ifft2(u_hat).real
        v = np.fft.ifft2(v_hat).real
        return (omega, psi, u, v)

    def nonlinear(self, omega_hat: np.ndarray) -> np.ndarray:
        """Return -FFT(u dot grad(omega)), with input/output dealiased."""
        state = omega_hat * self.dealias
        psi_hat = state * self.inv_k2
        u = np.fft.ifft2(1j * self.ky * psi_hat).real
        v = np.fft.ifft2(-1j * self.kx * psi_hat).real
        omega_x = np.fft.ifft2(1j * self.kx * state).real
        omega_y = np.fft.ifft2(1j * self.ky * state).real
        result = -np.fft.fft2(u * omega_x + v * omega_y)
        result[~self.dealias] = 0.0
        result[0, 0] = 0.0
        return result

    def step(self, state: np.ndarray) -> np.ndarray:
        nv = self.nonlinear(state)
        a = self.e2 * state + self.q * nv
        na = self.nonlinear(a)
        b = self.e2 * state + self.q * na
        nb = self.nonlinear(b)
        c = self.e2 * a + self.q * (2.0 * nb - nv)
        nc = self.nonlinear(c)
        updated = self.e * state + self.f1 * nv + 2.0 * self.f2 * (na + nb) + self.f3 * nc
        updated[~self.dealias] = 0.0
        updated[0, 0] = 0.0
        return updated

def periodic_delta(values: np.ndarray, center: float, length: float) -> np.ndarray:
    return (values - center + 0.5 * length) % length - 0.5 * length

def initial_vorticity(config: MergerConfig) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    n, length = (config.grid_size, config.domain_length)
    axis = np.arange(n, dtype=np.float64) * length / n
    x, y = np.meshgrid(axis, axis, indexing='xy')
    cy = 0.5 * length
    centers = (0.5 * length - 0.5 * config.vortex_separation, 0.5 * length + 0.5 * config.vortex_separation)
    amplitude = config.vortex_circulation / (2.0 * math.pi * config.vortex_sigma ** 2)
    omega = np.zeros((n, n), dtype=np.float64)
    dy = periodic_delta(y, cy, length)
    for cx in centers:
        dx = periodic_delta(x, cx, length)
        omega += amplitude * np.exp(-(dx * dx + dy * dy) / (2.0 * config.vortex_sigma ** 2))
    removed_mean = float(np.mean(omega))
    omega -= removed_mean
    return (axis, omega, np.asarray(centers, dtype=np.float64), removed_mean)

def make_solver(config: MergerConfig) -> tuple[VorticityDNS, np.ndarray, dict[str, float]]:
    axis, omega, _, removed_mean = initial_vorticity(config)
    provisional = VorticityDNS(config.grid_size, config.domain_length, 1.0, config.time_step)
    initial_hat = provisional.filtered_hat(omega)
    filtered_omega, _, u, v = provisional.fields_from_hat(initial_hat)
    mean_speed = float(np.mean(np.hypot(u, v)))
    viscosity = config.domain_length * mean_speed / config.reynolds_number
    solver = VorticityDNS(config.grid_size, config.domain_length, viscosity, config.time_step)
    state = solver.filtered_hat(filtered_omega)
    metadata = {'initial_mean_speed': mean_speed, 'kinematic_viscosity': viscosity, 'measured_reynolds_number': config.domain_length * mean_speed / viscosity, 'grid_spacing': float(axis[1] - axis[0]), 'removed_mean_vorticity': removed_mean}
    return (solver, state, metadata)

def diagnostics(solver: VorticityDNS, state: np.ndarray, time: float) -> dict[str, float]:
    omega, _, u, v = solver.fields_from_hat(state)
    omega_x = np.fft.ifft2(1j * solver.kx * state).real
    omega_y = np.fft.ifft2(1j * solver.ky * state).real
    div = np.fft.ifft2(1j * solver.kx * (1j * solver.ky * state * solver.inv_k2) + 1j * solver.ky * (-1j * solver.kx * state * solver.inv_k2)).real
    spectral_power = np.abs(state) ** 2
    kr = np.sqrt(solver.k2)
    retained_max = float(np.max(kr[solver.dealias]))
    high_shell = solver.dealias & (kr >= 0.8 * retained_max)
    resolved = solver.dealias & (kr > 0.0)
    tail_ratio = float(np.sum(spectral_power[high_shell]) / max(np.sum(spectral_power[resolved]), 1e-300))
    return {'time': float(time), 'kinetic_energy': float(0.5 * np.mean(u * u + v * v)), 'enstrophy': float(0.5 * np.mean(omega * omega)), 'palinstrophy': float(0.5 * np.mean(omega_x * omega_x + omega_y * omega_y)), 'mean_vorticity': float(np.mean(omega)), 'max_abs_divergence': float(np.max(np.abs(div))), 'max_abs_vorticity': float(np.max(np.abs(omega))), 'spectral_tail_power_ratio': tail_ratio}

def integrate(config: MergerConfig, *, progress: bool=True) -> tuple[np.ndarray, dict[str, np.ndarray], list[dict[str, float]], dict[str, float]]:
    solver, state, metadata = make_solver(config)
    output_times = np.linspace(0.0, config.final_time, config.output_frames)
    fields = {name: np.empty((config.output_frames, config.grid_size, config.grid_size), dtype=np.float64) for name in ('vorticity', 'streamfunction', 'velocity_u', 'velocity_v')}
    records: list[dict[str, float]] = []
    time = 0.0
    for frame, target in enumerate(output_times):
        while time < target - 32.0 * np.finfo(float).eps * max(1.0, target):
            remaining = target - time
            if remaining < solver.dt * (1.0 - 1e-12):
                short_solver = VorticityDNS(solver.n, solver.length, solver.viscosity, remaining)
                state = short_solver.step(state)
                time = target
            else:
                state = solver.step(state)
                time += solver.dt
        omega, psi, u, v = solver.fields_from_hat(state)
        fields['vorticity'][frame] = omega
        fields['streamfunction'][frame] = psi
        fields['velocity_u'][frame] = u
        fields['velocity_v'][frame] = v
        records.append(diagnostics(solver, state, float(target)))
        if progress:
            print(f"frame {frame + 1:03d}/{config.output_frames}: t={target:.8f}, E={records[-1]['kinetic_energy']:.9e}, Z={records[-1]['enstrophy']:.9e}", flush=True)
    return (output_times, fields, records, metadata)

def write_h5(path: Path, config: MergerConfig, times: np.ndarray, fields: dict[str, np.ndarray], records: list[dict[str, float]], metadata: dict[str, float]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    axis = np.arange(config.grid_size, dtype=np.float64) * (config.domain_length / config.grid_size)
    with h5py.File(path, 'w') as handle:
        handle.create_dataset('time', data=times)
        handle.create_dataset('x', data=axis)
        handle.create_dataset('y', data=axis)
        for name, values in fields.items():
            handle.create_dataset(name, data=values, compression='gzip', compression_opts=4, shuffle=True, chunks=(1, config.grid_size, config.grid_size))
        diagnostic_group = handle.create_group('diagnostics')
        for key in records[0]:
            diagnostic_group.create_dataset(key, data=np.asarray([record[key] for record in records]))
        for key, value in asdict(config).items():
            handle.attrs[key] = value
        for key, value in metadata.items():
            handle.attrs[key] = value
        handle.attrs['equations'] = '2D incompressible Navier-Stokes in vorticity-streamfunction form'
        handle.attrs['boundary_condition'] = 'doubly periodic'
        handle.attrs['spatial_discretization'] = 'Fourier pseudo-spectral, Orszag 2/3 dealiased'
        handle.attrs['time_integrator'] = 'Kassam-Trefethen ETDRK4'
        handle.attrs['vorticity_convention'] = 'omega = partial_x(v) - partial_y(u)'
        handle.attrs['uniform_vorticity_mode'] = 'removed for periodic compatibility'

def render_preview(output_stem: Path, config: MergerConfig, times: np.ndarray, vorticity: np.ndarray) -> None:
    targets = np.asarray([0.0, 2.0 * math.pi, 4.0 * math.pi, 6.0 * math.pi])
    indices = [int(np.argmin(np.abs(times - target))) for target in targets]
    selected = vorticity[indices]
    limit = float(np.max(np.abs(selected)))
    norm = TwoSlopeNorm(vmin=-limit, vcenter=0.0, vmax=limit)
    fig, axes = plt.subplots(1, 4, figsize=(13.0, 3.45), constrained_layout=True)
    image = None
    for axis, index in zip(axes, indices, strict=True):
        image = axis.imshow(vorticity[index], origin='lower', extent=(0.0, config.domain_length, 0.0, config.domain_length), cmap='RdBu_r', norm=norm, interpolation='bicubic')
        axis.set_title(f'$t={times[index] / math.pi:.0f}\\pi$')
        axis.set_xlabel('$x$')
        axis.set_aspect('equal')
    axes[0].set_ylabel('$y$')
    assert image is not None
    colorbar = fig.colorbar(image, ax=axes, shrink=0.82, pad=0.015)
    colorbar.set_label('vorticity $\\omega$')
    fig.suptitle(f'2D vortex merger DNS, $128^2$, $Re={config.reynolds_number:.0f}$', fontsize=12)
    fig.savefig(output_stem.with_suffix('.png'), dpi=240, bbox_inches='tight')
    fig.savefig(output_stem.with_suffix('.pdf'), bbox_inches='tight')
    plt.close(fig)

def validate_time_refinement(config: MergerConfig, fine_fields: dict[str, np.ndarray]) -> dict[str, float]:
    coarse = MergerConfig(**{**asdict(config), 'time_step': 2.0 * config.time_step})
    _, coarse_fields, _, _ = integrate(coarse, progress=False)
    delta = coarse_fields['vorticity'][-1] - fine_fields['vorticity'][-1]
    reference = fine_fields['vorticity'][-1]
    return {'coarse_time_step': coarse.time_step, 'fine_time_step': config.time_step, 'final_vorticity_relative_l2': float(np.linalg.norm(delta) / np.linalg.norm(reference)), 'final_vorticity_linf': float(np.max(np.abs(delta)))}

def parse_args() -> argparse.Namespace:
    default_output = Path(__file__).resolve().parents[1] / 'outputs' / 'gt_128_re350_t6pi'
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=default_output)
    parser.add_argument('--grid-size', type=int, default=128)
    parser.add_argument('--dt', type=float, default=0.005)
    parser.add_argument('--frames', type=int, default=61)
    parser.add_argument('--final-time', type=float, default=6.0 * math.pi)
    parser.add_argument('--separation', type=float, default=0.59)
    parser.add_argument('--validate-time', action='store_true')
    return parser.parse_args()

def main() -> int:
    args = parse_args()
    config = MergerConfig(grid_size=args.grid_size, time_step=args.dt, output_frames=args.frames, final_time=args.final_time, vortex_separation=args.separation)
    output_dir = args.output.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    h5_path = output_dir / 'vortex_merger_gt.h5'
    print(json.dumps(asdict(config), indent=2), flush=True)
    times, fields, records, metadata = integrate(config)
    write_h5(h5_path, config, times, fields, records, metadata)
    render_preview(output_dir / 'vortex_merger_gt_preview', config, times, fields['vorticity'])
    energy = np.asarray([record['kinetic_energy'] for record in records])
    enstrophy = np.asarray([record['enstrophy'] for record in records])
    palinstrophy = np.asarray([record['palinstrophy'] for record in records])
    viscosity = metadata['kinematic_viscosity']
    energy_change = energy[-1] - energy[0]
    enstrophy_change = enstrophy[-1] - enstrophy[0]
    energy_budget_residual = energy_change + 2.0 * viscosity * np.trapezoid(enstrophy, times)
    enstrophy_budget_residual = enstrophy_change + 2.0 * viscosity * np.trapezoid(palinstrophy, times)
    summary: dict[str, object] = {'config': asdict(config), 'metadata': metadata, 'checks': {'all_finite': bool(all((np.all(np.isfinite(values)) for values in fields.values()))), 'energy_monotone_nonincreasing': bool(np.all(np.diff(energy) <= 1e-13)), 'enstrophy_monotone_nonincreasing': bool(np.all(np.diff(enstrophy) <= 1e-13)), 'max_abs_divergence': float(max((record['max_abs_divergence'] for record in records))), 'max_abs_mean_vorticity': float(max((abs(record['mean_vorticity']) for record in records))), 'max_spectral_tail_power_ratio': float(max((record['spectral_tail_power_ratio'] for record in records))), 'kinetic_energy_decay_fraction': float(1.0 - energy[-1] / energy[0]), 'enstrophy_decay_fraction': float(1.0 - enstrophy[-1] / enstrophy[0]), 'energy_budget_relative_residual': float(abs(energy_budget_residual) / abs(energy_change)), 'enstrophy_budget_relative_residual': float(abs(enstrophy_budget_residual) / abs(enstrophy_change))}, 'initial_diagnostics': records[0], 'final_diagnostics': records[-1]}
    if args.validate_time:
        print('running dt-refinement validation ...', flush=True)
        summary['time_refinement'] = validate_time_refinement(config, fields)
    summary['files'] = {'h5': {'path': str(h5_path)}, 'preview_png': str(output_dir / 'vortex_merger_gt_preview.png'), 'preview_pdf': str(output_dir / 'vortex_merger_gt_preview.pdf')}
    summary_path = output_dir / 'summary.json'
    summary_path.write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print(json.dumps(summary, indent=2), flush=True)
    print(f'wrote {h5_path}', flush=True)
    return 0
if __name__ == '__main__':
    sys.exit(main())
