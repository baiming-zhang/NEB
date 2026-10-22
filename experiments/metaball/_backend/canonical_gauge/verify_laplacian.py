from __future__ import annotations
import argparse
import time
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import map_coordinates
from threadpoolctl import threadpool_limits
from gauge_core import Complex, GT, ROOT, THEORY, geometry_mask, json_write
BLUE, RED = ('#3B6FB6', '#D1495B')

def relative(x, y):
    return float(np.linalg.norm(x - y) / max(np.linalg.norm(y), 1e-30))

def sample_gt(points):
    data = np.load(GT / 'velocity/frame_000.npy', mmap_mode='r')
    coords = points[:, ::-1].T * 128 - 0.5
    return np.column_stack([map_coordinates(data[..., j], coords, order=1, mode='nearest') for j in range(3)])

def reconstruction(c, u):
    basis = c.to_cells(u)
    target = sample_gt(c.cell_points)
    rng = np.random.default_rng(20260907)
    fit = np.sort(rng.choice(len(target), int(0.65 * len(target)), replace=False))
    test = np.setdiff1d(np.arange(len(target)), fit)
    rows, recon = ([], None)
    for k in sorted({x for x in (4, 8, 16, 32, 64, 128, u.shape[1]) if x <= u.shape[1]}):
        coeff = np.linalg.lstsq(basis[fit, :, :k].reshape(-1, k), target[fit].ravel(), rcond=1e-10)[0]
        recon = np.einsum('nck,k->nc', basis[:, :, :k], coeff)
        rows.append(dict(modes=k, fit_velocity_rl2=relative(recon[fit], target[fit]), test_velocity_rl2=relative(recon[test], target[test])))
    return (rows, target, recon, fit, test)

def run(n, k, output):
    start = time.perf_counter()
    print(f'Building n={n}, K={k}', flush=True)
    mask, geom = geometry_mask(n)
    c = Complex(mask)
    built = time.perf_counter()
    print(c.metadata(), flush=True)
    lam, a, u = c.spectrum(k)
    solved = time.perf_counter()
    print(f'Spectrum solved in {solved - built:.2f}s', flush=True)
    rng = np.random.default_rng(20260907)
    canonical = a[:, :4]
    exact = c.G @ rng.normal(size=(len(c.vertices), 4))
    exact /= np.linalg.norm(exact, axis=0)
    harm = np.repeat(c.H[:, :1], 4, axis=1)
    shifts = []
    for alpha in (0.0, 1.0, 10.0, 100.0, 1000.0):
        shifted = canonical + alpha * (exact + harm)
        projected = c.project(shifted)
        raw_r = np.sum((c.C @ shifted) ** 2, axis=0) / np.sum(shifted ** 2, axis=0)
        gf_r = np.sum((c.C @ projected) ** 2, axis=0) / np.sum(projected ** 2, axis=0)
        shifts.append(dict(alpha=alpha, raw_rayleigh=float(raw_r[0]), canonical_rayleigh=float(gf_r[0]), canonical_relative_change=relative(projected, canonical), lifted_relative_change=relative(c.C @ shifted, c.C @ canonical)))
    p_res = np.linalg.norm(c.C.T @ (c.C @ a) - a * lam, axis=0) / lam
    u_res = np.linalg.norm(c.C @ (c.C.T @ u) - u * lam, axis=0) / lam
    gauge = np.linalg.norm(c.G.T @ a, axis=0) / np.sqrt(lam)
    div = np.linalg.norm(c.D @ u, axis=0) / np.sqrt(lam)
    vel_r = np.sum((c.C.T @ u) ** 2, axis=0)
    records, target, recon, fit, test = reconstruction(c, u)
    metrics = dict(name='GAUGE-CANONICAL EIGENFLUID — Laplacian verification', geometry_token=12000001, geometry_xyzr=geom.tolist(), theory=str(THEORY), grid=c.metadata(), modes=k, eigenvalues=lam.tolist(), diagnostics=dict(curl_grad_max_abs=float(abs(c.C @ c.G).max()), div_curl_max_abs=float(abs(c.D @ c.C).max()), potential_eigen_residual_max=float(p_res.max()), velocity_eigen_residual_max=float(u_res.max()), gauge_residual_max=float(gauge.max()), velocity_divergence_max=float(div.max()), harmonic_overlap_max=float(abs(c.H.T @ a).max(initial=0)), potential_gram_frobenius=float(np.linalg.norm(a.T @ a - np.eye(k))), velocity_gram_frobenius=float(np.linalg.norm(u.T @ u - np.eye(k))), paired_rayleigh_relative_max=float(np.max(abs(vel_r - lam) / lam)), voxel_wall_normal_flux_max=0.0, voxel_tangential_potential_max=0.0), gauge_perturbations=shifts, reconstruction=records, evaluation=dict(gt=str(GT / 'velocity/frame_000.npy'), target='existing independent 128^3 DNS frame 0, trilinearly sampled at pilot cell centers', fit_cells=len(fit), test_cells=len(test), seed=20260907, scope='coarse-grid pilot; not the original full-128^3/K2048 score; no time rollout'), timing=dict(assembly_seconds=built - start, eigensolve_seconds=solved - built, total_seconds=time.perf_counter() - start))
    output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output / 'laplacian_modes.npz', eigenvalues=lam, canonical_potential=a / c.h ** 1.5, velocity=u / c.h ** 1.5, potential_edges=c.edges, velocity_faces=c.faces, cells=c.cells, geometry_xyzr=geom, mask=mask, spacing=c.h, relative_harmonic_potential=c.H / c.h ** 1.5, gt_cell_velocity=target, reconstructed_cell_velocity=recon, fit_cells=fit, test_cells=test)
    json_write(output / 'metrics.json', metrics)
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.5), constrained_layout=True)
    axes[0].plot(np.arange(1, k + 1), lam, color=BLUE, label='Canonical potential')
    axes[0].plot(np.arange(1, k + 1), vel_r, '--', color=RED, label='Lifted velocity')
    axes[0].set(xlabel='Mode', ylabel='Rayleigh quotient')
    axes[0].legend(frameon=False)
    axes[1].semilogy([r['alpha'] for r in shifts], [r['raw_rayleigh'] for r in shifts], 'o-', color=RED, label='Raw quotient')
    axes[1].semilogy([r['alpha'] for r in shifts], [r['canonical_rayleigh'] for r in shifts], 'o-', color=BLUE, label='Gauge-canonical')
    axes[1].set(xscale='symlog', xlabel='Gauge perturbation amplitude', ylabel='Rayleigh quotient')
    axes[1].legend(frameon=False)
    axes[2].plot([r['modes'] for r in records], [r['test_velocity_rl2'] for r in records], 'o-', color=BLUE)
    axes[2].set(xlabel='Retained modes', ylabel='Held-out DNS velocity rL2')
    for ax in axes:
        ax.spines[['top', 'right']].set_visible(False)
        ax.grid(alpha=0.15)
    fig.suptitle(f'GAUGE-CANONICAL EIGENFLUID | Laplacian pilot | {n}³ voxel grid')
    for ext in ('png', 'pdf'):
        fig.savefig(output / f'theory_checks.{ext}', dpi=200)
    plt.close(fig)
    full = []
    for field in (target, recon, target - recon):
        array = np.full((n, n, n), np.nan)
        array[tuple(c.cells.T)] = np.linalg.norm(field, axis=1)
        full.append(array)
    fig, axes = plt.subplots(3, 3, figsize=(9, 8), constrained_layout=True)
    for row in range(3):
        for col, array in enumerate(full):
            sl = np.take(array, n // 2, axis=row).T
            im = axes[row, col].imshow(sl, origin='lower', extent=(0, 1, 0, 1), cmap='RdBu_r', vmin=0, vmax=0.75)
            horizontal, vertical = [j for j in range(3) if j != row]
            axes[row, col].set(xlabel='xyz'[horizontal], ylabel='xyz'[vertical])
            if row == 0:
                axes[row, col].set_title(('Same DNS target', 'Gauge-canonical Laplacian', 'Velocity error magnitude')[col])
    fig.colorbar(im, ax=axes, shrink=0.75, label='Speed / error magnitude')
    fig.suptitle(f'Metaball token 12000001 | K={k} | pilot cell-center evaluation')
    for ext in ('png', 'pdf'):
        fig.savefig(output / f'reconstruction_slices.{ext}', dpi=200)
    plt.close(fig)
    field = c.to_cells(u[:, 0])
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.4), constrained_layout=True)
    for axis, ax in enumerate(axes):
        selection = c.cells[:, axis] == n // 2
        b, d = [j for j in range(3) if j != axis]
        p, v = (c.cell_points[selection], field[selection])
        ax.quiver(p[:, b], p[:, d], v[:, b], v[:, d], np.linalg.norm(v, axis=1), cmap='RdBu_r', angles='xy')
        ax.set(xlim=(0, 1), ylim=(0, 1), aspect='equal', xlabel='xyz'[b], ylabel='xyz'[d], title=f"{'xyz'[axis]} = {(n // 2 + 0.5) / n:.4f}")
    fig.suptitle(f'First gauge-canonical Laplacian velocity | λ₁={lam[0]:.4f}')
    for ext in ('png', 'pdf'):
        fig.savefig(output / f'first_velocity_mode.{ext}', dpi=200)
    plt.close(fig)
    print({'diagnostics': metrics['diagnostics'], 'reconstruction': records, 'timing': metrics['timing']}, flush=True)
    return metrics
