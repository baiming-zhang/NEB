"""Validate learned harmonics and assemble them with learned positive modes."""
from __future__ import annotations
import argparse
import csv
import json
import sys
from pathlib import Path
import cupy as cp
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import numpy as np
from matplotlib.ticker import FormatStrFormatter
ROOT = Path(__file__).resolve().parents[1]
METABALL = ROOT.parent
POSITIVE = METABALL / 'positive'
sys.path.insert(0, str(POSITIVE / 'src'))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from cupy_ascii_include import install_ascii_include_alias
install_ascii_include_alias(cp)
from neb2d.basisnet.metaball2d.geometry import get_obstacles, obstacle_boundary, obstacle_gradient, outer_boundary, outer_gradient
from neb2d.basisnet.metaball2d.hardcoded_backend import HardcodedModel, cfg_namespace
from harmonic_common import learned_harmonic_fields, velocity_matrix
from mesh_builder import build_domain_mesh
from solver import TopologyNS

def load_checkpoint(path: Path):
    metadata = json.loads(path.with_suffix('.json').read_text(encoding='utf-8'))
    with np.load(path, allow_pickle=False) as archive:
        arrays = {name: archive[name] for name in archive.files}
    return (metadata, arrays)

def scalar(value) -> float:
    return float(value.item())

def positive_velocity_matrix(network, cfg, points) -> cp.ndarray:
    result = cp.empty((2 * len(points), int(cfg.modes)), dtype=cp.float64)
    for mode, field, _ in network.forward_stream_jet1(cfg, points.astype(np.float32), 0.0, 0.0, False):
        gradient = field.grad[:, 0]
        velocity = cp.stack((gradient[:, 1], -gradient[:, 0]), axis=1)
        result[:, int(mode.mode_index)] = velocity.reshape(-1).astype(cp.float64)
    return result

def project_errors(weighted_basis: cp.ndarray, weighted_gt: cp.ndarray):
    scales = cp.sqrt(cp.maximum(cp.sum(weighted_basis * weighted_basis, axis=0), 1e-30))
    normalized = weighted_basis / scales[None, :]
    gram = 0.5 * (normalized.T @ normalized + (normalized.T @ normalized).T)
    eigenvalues, eigenvectors = cp.linalg.eigh(gram)
    maximum = cp.max(eigenvalues)
    keep = eigenvalues > maximum * 1e-10
    rhs = normalized.T @ weighted_gt
    coefficients = eigenvectors[:, keep] @ (eigenvectors[:, keep].T @ rhs / eigenvalues[keep, None])
    reconstruction = normalized @ coefficients
    errors = cp.sqrt(cp.sum((reconstruction - weighted_gt) ** 2, axis=0) / cp.maximum(cp.sum(weighted_gt * weighted_gt, axis=0), 1e-30))
    return (cp.asnumpy(errors), {'rank': int(cp.count_nonzero(keep).item()), 'dimension': int(weighted_basis.shape[1]), 'gram_condition_kept': scalar(maximum / cp.min(eigenvalues[keep])), 'gram_eigenvalue_min': scalar(cp.min(eigenvalues)), 'gram_eigenvalue_max': scalar(maximum)})

def whiten(weighted: cp.ndarray) -> cp.ndarray:
    gram = 0.5 * (weighted.T @ weighted + (weighted.T @ weighted).T)
    values, vectors = cp.linalg.eigh(gram)
    if scalar(cp.min(values)) <= 1e-12 * scalar(cp.max(values)):
        raise RuntimeError('harmonic block is rank deficient')
    return weighted @ (vectors / cp.sqrt(values)[None, :])

def boundary_samples(cfg):
    pieces = [outer_boundary(cfg.outer_tau, 720)]
    normals = [outer_gradient(pieces[0], cfg.outer_tau)]
    targets = [np.zeros((len(pieces[0]), int(cfg.harmonic_dimension)))]
    labels = [np.zeros(len(pieces[0]), dtype=np.int32)]
    for index, spec in enumerate(get_obstacles(cfg.subcase), start=1):
        points = obstacle_boundary(spec, 512)
        pieces.append(points)
        normals.append(-obstacle_gradient(points, spec))
        target = np.zeros((len(points), int(cfg.harmonic_dimension)))
        target[:, index - 1] = 1.0
        targets.append(target)
        labels.append(np.full(len(points), index, dtype=np.int32))
    normal = np.vstack(normals)
    normal /= np.linalg.norm(normal, axis=1, keepdims=True)
    return (np.vstack(pieces), normal, np.vstack(targets), np.concatenate(labels))

def render_harmonics(output: Path, mesh, fem_modes, learned_aligned, case_name: str) -> None:
    count = fem_modes.shape[1]
    triangulation = mtri.Triangulation(mesh.xy[:, 0], mesh.xy[:, 1], mesh.tri)
    plt.rcParams.update({'font.size': 17, 'axes.titlesize': 17, 'axes.labelsize': 17, 'xtick.labelsize': 15, 'ytick.labelsize': 15})
    fig, axes = plt.subplots(count, 3, figsize=(10.8, 2.55 * count), squeeze=False)
    columns = ('FEM harmonic', 'Learned harmonic', 'Difference')
    for column, label in enumerate(columns):
        axes[0, column].set_title(label, pad=8)
    for row in range(count):
        reference = fem_modes[:, row]
        learned = learned_aligned[:, row]
        difference = learned - reference
        scale = max(np.max(np.abs(reference)), np.max(np.abs(learned)), 1e-12)
        difference_scale = max(np.max(np.abs(difference)), 1e-12)
        for column, (field, limit) in enumerate(((reference, scale), (learned, scale), (difference, difference_scale))):
            axis = axes[row, column]
            artist = axis.tripcolor(triangulation, field, shading='gouraud', cmap='RdBu_r', vmin=-limit, vmax=limit, rasterized=False)
            axis.set_aspect('equal', adjustable='box')
            axis.set_xlim(1.0, 9.0)
            axis.set_ylim(-4.0, 4.0)
            axis.set_xticks([])
            axis.set_yticks([])
            for spine in axis.spines.values():
                spine.set_visible(False)
            colorbar = fig.colorbar(artist, ax=axis, fraction=0.043, pad=0.018)
            colorbar.ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
            colorbar.ax.tick_params(labelsize=13, length=2)
        axes[row, 0].text(-0.06, 0.5, f'$h={row + 1}$', transform=axes[row, 0].transAxes, ha='right', va='center', fontsize=17)
    fig.subplots_adjust(left=0.07, right=0.965, bottom=0.025, top=0.93, wspace=0.2, hspace=0.12)
    fig.savefig(output / f'{case_name}_harmonic_fields.png', dpi=300)
    fig.savefig(output / f'{case_name}_harmonic_fields.pdf')
    plt.close(fig)

def render_errorbar(output: Path, learned_errors: np.ndarray, laplacian_errors: np.ndarray, case_name: str) -> None:
    plt.rcParams.update({'font.size': 18, 'axes.labelsize': 18, 'xtick.labelsize': 16, 'ytick.labelsize': 16})
    fig, ax = plt.subplots(figsize=(5.8, 4.2))
    colors = ('#3B6FB6', '#C4473A')
    labels = ('Laplacian\n+ FEM $H$', 'Potential\n+ learned $H$')
    for x, values, color in zip((0, 1), (laplacian_errors, learned_errors), colors):
        ax.errorbar(x, np.mean(values), yerr=np.std(values, ddof=1), fmt='o', ms=8, capsize=6, capthick=1.8, elinewidth=1.8, color=color, zorder=3)
        jitter = np.linspace(-0.075, 0.075, len(values))
        ax.scatter(x + jitter, values, s=20, alpha=0.42, color=color, edgecolors='none', zorder=2)
    ax.set_xticks((0, 1), labels)
    ax.set_ylabel('Relative velocity $L^2$ error')
    ax.spines[['top', 'right']].set_visible(False)
    ax.grid(axis='y', alpha=0.18, linewidth=0.8)
    ax.margins(x=0.28)
    fig.subplots_adjust(left=0.18, right=0.975, bottom=0.24, top=0.97)
    fig.savefig(output / f'{case_name}_assembled_errorbar.png', dpi=300)
    fig.savefig(output / f'{case_name}_assembled_errorbar.pdf')
    plt.close(fig)

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--harmonic-checkpoint', type=Path, required=True)
    parser.add_argument('--positive-checkpoint', type=Path, required=True)
    parser.add_argument('--baseline-dir', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    harmonic_meta, harmonic_arrays = load_checkpoint(args.harmonic_checkpoint.resolve())
    positive_meta, positive_arrays = load_checkpoint(args.positive_checkpoint.resolve())
    hcfg = cfg_namespace(harmonic_meta['config'])
    pcfg = cfg_namespace(positive_meta['config'])
    if hcfg.subcase != pcfg.subcase:
        raise RuntimeError('positive and harmonic checkpoints use different geometries')
    if int(pcfg.modes) + int(hcfg.harmonic_dimension) != 1024:
        raise RuntimeError('assembled basis must have exactly 1024 coordinates')
    with np.load(args.baseline_dir / 'mesh_bases_gt_reconstructions.npz', allow_pickle=False) as archive:
        baseline = {name: archive[name] for name in archive.files}
    mesh = build_domain_mesh(hcfg.subcase, hcfg.outer_tau, 0.04, 200, 96, 28.0)
    if not np.array_equal(mesh.tri, baseline['tri']) or not np.allclose(mesh.xy, baseline['xy']):
        raise RuntimeError('mesh differs from the frozen K1024 baseline')
    fem = TopologyNS(mesh, viscosity=0.0013333333333333333)
    harmonic_model = HardcodedModel(hcfg, harmonic_arrays)
    positive_model = HardcodedModel(pcfg, positive_arrays)
    centroids = np.mean(mesh.xy[mesh.tri], axis=1).astype(np.float32)
    nodes = mesh.xy.astype(np.float32)
    area = cp.asnumpy(fem.area)
    sqrt_weight = cp.asarray(np.repeat(np.sqrt(area), 2), dtype=cp.float64)
    positive_velocity = positive_velocity_matrix(positive_model, pcfg, centroids)
    harmonic_centroid_fields = learned_harmonic_fields(harmonic_model, hcfg, centroids, False)
    learned_harmonic_velocity = velocity_matrix(harmonic_centroid_fields).astype(cp.float64)
    total_velocity = cp.column_stack((positive_velocity, learned_harmonic_velocity))
    weighted_basis = total_velocity * sqrt_weight[:, None]
    gt_velocity = cp.column_stack([fem.velocity(cp.asarray(psi, dtype=cp.float64)).reshape(-1) for psi in baseline['gt_psi']])
    weighted_gt = gt_velocity * sqrt_weight[:, None]
    learned_errors, basis_diagnostics = project_errors(weighted_basis, weighted_gt)
    topology_name = f'P{pcfg.modes}H{hcfg.harmonic_dimension}'
    laplacian_errors = []
    for truth, reconstruction in zip(baseline['gt_psi'], baseline[f'{topology_name}_psi']):
        error = cp.asarray(reconstruction - truth, dtype=cp.float64)
        truth_device = cp.asarray(truth, dtype=cp.float64)
        laplacian_errors.append(scalar(cp.sqrt(error @ fem.S @ error / cp.maximum(truth_device @ fem.S @ truth_device, 1e-30))))
    laplacian_errors = np.asarray(laplacian_errors)
    fem_harmonic = cp.asarray(baseline['harmonic_modes'], dtype=cp.float64)
    fem_harmonic_velocity = fem.velocity(fem_harmonic).reshape(2 * len(mesh.tri), -1)
    learned_weighted = learned_harmonic_velocity * sqrt_weight[:, None]
    fem_weighted = fem_harmonic_velocity * sqrt_weight[:, None]
    learned_q = whiten(learned_weighted)
    fem_q = whiten(fem_weighted)
    singular_values = cp.linalg.svd(learned_q.T @ fem_q, compute_uv=False)
    singular_values = cp.clip(singular_values, 0.0, 1.0)
    principal_angles = cp.asnumpy(cp.arccos(singular_values) * 180.0 / np.pi)
    transform = cp.linalg.lstsq(learned_weighted, fem_weighted, rcond=None)[0]
    aligned_velocity = learned_weighted @ transform
    aligned_errors = cp.sqrt(cp.sum((aligned_velocity - fem_weighted) ** 2, axis=0) / cp.maximum(cp.sum(fem_weighted ** 2, axis=0), 1e-30))
    node_fields = learned_harmonic_fields(harmonic_model, hcfg, nodes, False)
    learned_nodes = cp.stack([field.value[:, 0] for _, field, _ in node_fields], axis=1).astype(cp.float64)
    aligned_nodes = learned_nodes @ transform
    weak_residuals = []
    for index in range(int(hcfg.harmonic_dimension)):
        values = learned_nodes[:, index]
        residual = fem.S @ values
        denominator = cp.maximum(cp.linalg.norm(fem.Sf @ values[fem.F]), 1e-30)
        weak_residuals.append(scalar(cp.linalg.norm(residual[fem.F]) / denominator))
    zero = cp.zeros(fem.n, dtype=cp.float64)
    learned_period = cp.stack([fem.gamma_from_state(learned_nodes[:, index], zero) for index in range(int(hcfg.harmonic_dimension))], axis=1)
    boundary_points, boundary_normals, boundary_targets, boundary_labels = boundary_samples(hcfg)
    boundary_fields = learned_harmonic_fields(harmonic_model, hcfg, boundary_points.astype(np.float32), False)
    boundary_values = cp.stack([field.value[:, 0] for _, field, _ in boundary_fields], axis=1)
    boundary_velocity = velocity_matrix(boundary_fields).reshape(len(boundary_points), 2, -1)
    normals_device = cp.asarray(boundary_normals, dtype=cp.float64)
    normal_velocity = cp.sum(boundary_velocity * normals_device[:, :, None], axis=1)
    trace_error = cp.abs(boundary_values - cp.asarray(boundary_targets, dtype=cp.float32))
    trace_spreads = []
    for component in range(int(hcfg.harmonic_dimension) + 1):
        selected = cp.asarray(boundary_labels == component)
        trace_spreads.append(cp.asnumpy(cp.max(boundary_values[selected], axis=0) - cp.min(boundary_values[selected], axis=0)).tolist())
    cross = (positive_velocity * sqrt_weight[:, None]).T @ learned_weighted
    rows = []
    for index, (learned, laplacian) in enumerate(zip(learned_errors, laplacian_errors)):
        rows.append({'seed_index': int(baseline['gt_seed_index'][index]), 'potential_plus_learned_harmonic_relative_velocity_L2': float(learned), 'laplacian_plus_fem_harmonic_relative_velocity_L2': float(laplacian), 'learned_better': bool(learned < laplacian)})
    with (args.output / 'per_field_metrics.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    result = {'status': 'PASSED' if np.isfinite(learned_errors).all() and basis_diagnostics['rank'] == 1024 and (scalar(cp.max(trace_error)) < 0.0002) and (scalar(cp.max(cp.abs(normal_velocity))) < 0.002) and (max(principal_angles) < 5.0) else 'FAILED_GATE', 'subcase': hcfg.subcase, 'total_capacity': 1024, 'assembled_model': topology_name, 'positive_checkpoint_epoch': int(positive_meta['epoch']), 'harmonic_checkpoint_epoch': int(harmonic_meta['epoch']), 'harmonic_treatment': 'learned hard-Dirichlet extension; no FEM harmonic columns in assembled learned basis', 'basis': basis_diagnostics, 'learned_relative_velocity_L2': {'mean': float(np.mean(learned_errors)), 'std': float(np.std(learned_errors, ddof=1)), 'median': float(np.median(learned_errors)), 'max': float(np.max(learned_errors))}, 'laplacian_relative_velocity_L2': {'mean': float(np.mean(laplacian_errors)), 'std': float(np.std(laplacian_errors, ddof=1)), 'median': float(np.median(laplacian_errors)), 'max': float(np.max(laplacian_errors))}, 'learned_wins': int(np.count_nonzero(learned_errors < laplacian_errors)), 'harmonic_validation': {'principal_angles_degrees': principal_angles.tolist(), 'fem_basis_relative_velocity_L2_after_alignment': cp.asnumpy(aligned_errors).tolist(), 'weak_laplacian_free_residual_scaled': weak_residuals, 'trace_max_abs_error': scalar(cp.max(trace_error)), 'trace_spreads_by_component': trace_spreads, 'normal_velocity_max_abs': scalar(cp.max(cp.abs(normal_velocity))), 'normal_velocity_rms': scalar(cp.sqrt(cp.mean(normal_velocity ** 2))), 'learned_period_matrix': cp.asnumpy(learned_period).tolist(), 'learned_period_rank': int(np.linalg.matrix_rank(cp.asnumpy(learned_period))), 'learned_period_condition': scalar(cp.linalg.cond(learned_period)), 'raw_energy_gram': cp.asnumpy(learned_weighted.T @ learned_weighted).tolist()}, 'positive_learned_harmonic_energy_cross_max_abs': scalar(cp.max(cp.abs(cross))), 'checkpoints': {'positive': str(args.positive_checkpoint.resolve()), 'harmonic': str(args.harmonic_checkpoint.resolve()), 'baseline': str(args.baseline_dir.resolve())}}
    (args.output / 'metrics.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    np.savez_compressed(args.output / 'learned_harmonic_and_assembly.npz', xy=mesh.xy, tri=mesh.tri, learned_harmonic_nodes=cp.asnumpy(learned_nodes), learned_harmonic_aligned_nodes=cp.asnumpy(aligned_nodes), fem_harmonic_nodes=baseline['harmonic_modes'], learned_errors=learned_errors, laplacian_errors=laplacian_errors, principal_angles_degrees=principal_angles)
    render_harmonics(args.output, mesh, baseline['harmonic_modes'], cp.asnumpy(aligned_nodes), hcfg.subcase)
    render_errorbar(args.output, learned_errors, laplacian_errors, hcfg.subcase)
    print(json.dumps(result, indent=2))
if __name__ == '__main__':
    main()
