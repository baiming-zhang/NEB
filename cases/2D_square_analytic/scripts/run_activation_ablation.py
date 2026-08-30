"""Sine-versus-ELU BasisNet ablation on the analytic unit-square spectrum."""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
import time
from dataclasses import asdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.colors import LinearSegmentedColormap
from scipy.optimize import linear_sum_assignment


CASE_ROOT = Path(__file__).resolve().parents[1]
BASISNET_ROOT = CASE_ROOT.parents[1]
sys.path.insert(0, str(BASISNET_ROOT))

from basisnet import (  # noqa: E402
    BasisNet,
    BasisNetConfig,
    discrete_fields,
    infer_basis,
    make_optimizers,
    sss_step,
)


def square_envelope(points: torch.Tensor, _: torch.Tensor) -> torch.Tensor:
    """Hard Dirichlet streamfunction boundary on [0,1]^2."""
    x, y = points.unbind(dim=1)
    return (x * (1.0 - x) * y * (1.0 - y)).unsqueeze(1)


def analytic_pairs(count: int) -> list[tuple[int, int]]:
    limit = max(4, int(math.ceil(math.sqrt(2 * count))) + 2)
    pairs = [(m, n) for m in range(1, limit + 1) for n in range(1, limit + 1)]
    return sorted(pairs, key=lambda pair: (pair[0] ** 2 + pair[1] ** 2, pair))[:count]


def analytic_basis(
    points: np.ndarray, pairs: list[tuple[int, int]], weights: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x, y = points[:, 0:1], points[:, 1:2]
    potentials, velocities, eigenvalues = [], [], []
    for m, n in pairs:
        psi = np.sin(m * math.pi * x) * np.sin(n * math.pi * y)
        u = n * math.pi * np.sin(m * math.pi * x) * np.cos(n * math.pi * y)
        v = -m * math.pi * np.cos(m * math.pi * x) * np.sin(n * math.pi * y)
        velocity = np.concatenate((u, v), axis=1)
        mass = np.sum(weights[:, None] * velocity**2)
        scale = 1.0 / math.sqrt(max(float(mass), 1.0e-15))
        potentials.append(psi[:, 0] * scale)
        velocities.append(velocity * scale)
        eigenvalues.append(math.pi**2 * (m * m + n * n))
    return (
        np.stack(potentials, axis=1),
        np.stack(velocities, axis=1),
        np.asarray(eigenvalues),
    )


def evaluation_grid(size: int) -> tuple[np.ndarray, np.ndarray]:
    axis = np.linspace(0.0, 1.0, size, dtype=np.float32)
    x, y = np.meshgrid(axis, axis, indexing="xy")
    points = np.column_stack((x.reshape(-1), y.reshape(-1)))
    trap = np.ones((size, size), dtype=np.float64)
    trap[[0, -1], :] *= 0.5
    trap[:, [0, -1]] *= 0.5
    weights = trap.reshape(-1) / (size - 1) ** 2
    return points, weights


def eigenspace_groups(pairs: list[tuple[int, int]]) -> list[list[int]]:
    groups: dict[int, list[int]] = {}
    for index, (m, n) in enumerate(pairs):
        groups.setdefault(m * m + n * n, []).append(index)
    return list(groups.values())


def align_degenerate_groups(
    learned_velocity: np.ndarray,
    learned_potential: np.ndarray,
    exact_velocity: np.ndarray,
    weights: np.ndarray,
    pairs: list[tuple[int, int]],
) -> tuple[np.ndarray, np.ndarray]:
    aligned_velocity = learned_velocity.copy()
    aligned_potential = learned_potential.copy()
    for indices in eigenspace_groups(pairs):
        learned = learned_velocity[:, indices, :]
        exact = exact_velocity[:, indices, :]
        cross = np.einsum("n,nid,njd->ij", weights, learned, exact)
        left, _, right = np.linalg.svd(cross, full_matrices=False)
        rotation = left @ right
        aligned_velocity[:, indices, :] = np.einsum("nid,ij->njd", learned, rotation)
        aligned_potential[:, indices] = learned_potential[:, indices] @ rotation
    return aligned_velocity, aligned_potential


def weighted_cross(
    first: np.ndarray, second: np.ndarray, weights: np.ndarray
) -> np.ndarray:
    return np.einsum("n,nid,njd->ij", weights, first, second)


def align_matched_eigenspaces(
    learned_velocity: np.ndarray,
    learned_potential: np.ndarray,
    exact_velocity: np.ndarray,
    weights: np.ndarray,
    pairs: list[tuple[int, int]],
    assignment: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[int, float]]:
    """Remove arbitrary rotations inside the actually learned eigenspaces."""
    aligned_velocity = learned_velocity.copy()
    aligned_potential = learned_potential.copy()
    errors = np.empty(len(assignment), dtype=np.float64)
    group_rms_sine: dict[int, float] = {}
    eigenvalue_keys = np.asarray([m * m + n * n for m, n in pairs])
    for key in np.unique(eigenvalue_keys[assignment]):
        learned_indices = np.flatnonzero(eigenvalue_keys[assignment] == key)
        exact_indices = assignment[learned_indices]
        learned = learned_velocity[:, learned_indices, :]
        exact = exact_velocity[:, exact_indices, :]
        cross = weighted_cross(learned, exact, weights)
        left, singular, right = np.linalg.svd(cross, full_matrices=False)
        rotation = left @ right
        aligned_velocity[:, learned_indices, :] = np.einsum(
            "nid,ij->njd", learned, rotation
        )
        aligned_potential[:, learned_indices] = (
            learned_potential[:, learned_indices] @ rotation
        )
        difference = aligned_velocity[:, learned_indices, :] - exact
        errors[learned_indices] = np.sqrt(
            np.einsum("n,nkd,nkd->k", weights, difference, difference)
        )
        group_rms_sine[int(key)] = float(
            np.sqrt(np.mean(np.maximum(0.0, 1.0 - singular**2)))
        )
    return aligned_velocity, aligned_potential, errors, group_rms_sine


def save_figure(fig: plt.Figure, stem: Path) -> None:
    fig.savefig(stem.with_suffix(".png"), dpi=240, bbox_inches="tight")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def model_config(args: argparse.Namespace, activation: str) -> BasisNetConfig:
    return BasisNetConfig(
        spatial_dim=2,
        parameter_dim=0,
        modes=args.modes,
        hidden=args.hidden,
        hidden_layers=args.layers,
        frequencies=args.frequencies,
        bounds=((0.0, 1.0), (0.0, 1.0)),
        periodic=False,
        difference_step=args.difference_step,
        ordering="dynamic",
        mass_floor=1.0e-10,
        rayleigh_scale=50.0,
        grad_clip=1.0,
        activation=activation,
    )


def train_one(
    activation: str, args: argparse.Namespace, output: Path
) -> tuple[BasisNet, list[dict[str, float | int | str]]]:
    # Matched random stream: activation is the only ablated model choice.
    seed = args.seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    device = torch.device(args.device)
    config = model_config(args, activation)
    model = BasisNet(config).to(device)
    optimizers = make_optimizers(model, args.learning_rate)
    engine = torch.quasirandom.SobolEngine(2, scramble=True, seed=seed)
    validation = torch.quasirandom.SobolEngine(
        2, scramble=True, seed=seed + 9000
    ).draw(args.validation_points).to(device)
    parameters = torch.empty(0, device=device)
    weight = 1.0 / args.points
    validation_weight = 1.0 / args.validation_points
    run_dir = output / activation
    (run_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    history: list[dict[str, float | int | str]] = []
    best_loss = math.inf
    started = time.time()

    for step in range(1, args.steps + 1):
        model.train()
        points = engine.draw(args.points).to(device)
        loss, diagnostics, _ = sss_step(
            model,
            points,
            parameters,
            weight,
            optimizers=optimizers,
            envelope=square_envelope,
        )
        if not torch.isfinite(loss):
            raise RuntimeError(f"{activation} produced non-finite loss at step {step}")
        if step == 1 or step % args.print_every == 0 or step == args.steps:
            model.eval()
            with torch.no_grad():
                val_loss, val_diagnostics, _ = sss_step(
                    model,
                    validation,
                    parameters,
                    validation_weight,
                    envelope=square_envelope,
                )
            row: dict[str, float | int | str] = {
                "activation": activation,
                "step": step,
                "train_loss": diagnostics["loss"],
                "validation_loss": float(val_loss.cpu()),
                "mean_rayleigh": val_diagnostics["mean_rayleigh"],
                "max_rayleigh": val_diagnostics["max_rayleigh"],
                "raw_rank": val_diagnostics["effective_raw_rank"],
                "elapsed_seconds": time.time() - started,
            }
            history.append(row)
            print(
                f"[{activation}] step={step:05d} val={row['validation_loss']:.6f} "
                f"R={row['mean_rayleigh']:.3f} rank={row['raw_rank']} "
                f"elapsed={row['elapsed_seconds']:.1f}s",
                flush=True,
            )
            if float(val_loss) < best_loss:
                best_loss = float(val_loss)
                torch.save(
                    {"config": asdict(config), "model": model.state_dict(), "step": step},
                    run_dir / "checkpoints" / "best.pt",
                )

    with (run_dir / "history.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)
    payload = torch.load(run_dir / "checkpoints" / "best.pt", map_location=device)
    model.load_state_dict(payload["model"])
    model.eval()
    return model, history


def evaluate_one(
    activation: str,
    model: BasisNet,
    args: argparse.Namespace,
    output: Path,
) -> tuple[dict[str, object], np.ndarray, np.ndarray]:
    points, weights = evaluation_grid(args.eval_grid)
    device = next(model.parameters()).device
    point_tensor = torch.from_numpy(points).to(device)
    parameters = torch.empty(0, device=device)
    weight_tensor = torch.from_numpy(weights.astype(np.float32)).to(device)
    with torch.no_grad():
        diagnostics, basis = infer_basis(
            model,
            point_tensor,
            parameters,
            weight_tensor,
            envelope=square_envelope,
        )
        fields = discrete_fields(
            model, point_tensor, parameters, envelope=square_envelope
        )
    learned_velocity = basis["velocity"].cpu().numpy().astype(np.float64)
    learned_potential = basis["potential"][:, :, 0].cpu().numpy().astype(np.float64)
    raw_velocity = fields["velocity"].cpu().numpy().astype(np.float64)
    raw_mass = np.einsum("n,nkd,nkd->k", weights, raw_velocity, raw_velocity)

    pairs = analytic_pairs(max(args.candidate_modes, args.modes))
    exact_potential, exact_velocity, exact_values = analytic_basis(points, pairs, weights)
    cross_first = weighted_cross(
        learned_velocity, exact_velocity[:, : args.modes], weights
    )
    singular = np.linalg.svd(cross_first, compute_uv=False)
    rms_sine = float(np.sqrt(np.mean(np.maximum(0.0, 1.0 - singular**2))))
    cross_candidates = weighted_cross(learned_velocity, exact_velocity, weights)
    learned_indices, exact_indices = linear_sum_assignment(-np.abs(cross_candidates))
    assignment = np.empty(args.modes, dtype=np.int64)
    assignment[learned_indices] = exact_indices
    matched_correlation = cross_candidates[np.arange(args.modes), assignment]
    matched_exact_potential = exact_potential[:, assignment]
    individual_mode_errors = np.sqrt(
        np.maximum(0.0, 2.0 - 2.0 * np.abs(matched_correlation))
    )
    (
        _,
        eigenspace_aligned_potential,
        eigenspace_aligned_errors,
        group_rms_sine,
    ) = align_matched_eigenspaces(
        learned_velocity,
        learned_potential,
        exact_velocity,
        weights,
        pairs,
        assignment,
    )
    candidate_capture = np.sum(cross_candidates**2, axis=1)
    first_capture = np.sum(cross_first**2, axis=1)
    learned_values = np.asarray(diagnostics["peeled_rayleigh_in_causal_order"])

    rows = []
    for index, error in enumerate(individual_mode_errors):
        matched_index = int(assignment[index])
        m, n = pairs[matched_index]
        rows.append(
            {
                "mode": index + 1,
                "matched_analytic_pair": f"({m},{n})",
                "matched_analytic_index": matched_index + 1,
                "matched_analytic_eigenvalue": float(exact_values[matched_index]),
                "learned_rayleigh": float(learned_values[index]),
                "rayleigh_relative_error": float(
                    abs(learned_values[index] - exact_values[matched_index])
                    / exact_values[matched_index]
                ),
                "matched_absolute_correlation": float(abs(matched_correlation[index])),
                "matched_velocity_relative_l2": float(error),
                "eigenspace_aligned_velocity_relative_l2": float(
                    eigenspace_aligned_errors[index]
                ),
                "matched_eigenspace_rms_sine": group_rms_sine[m * m + n * n],
                "first_k_spectral_capture": float(first_capture[index]),
                "candidate_spectral_capture": float(candidate_capture[index]),
                "candidate_spectral_residual": float(max(0.0, 1.0 - candidate_capture[index])),
                "raw_velocity_mass": float(raw_mass[index]),
            }
        )
    summary: dict[str, object] = {
        "activation": activation,
        "modes": args.modes,
        "effective_raw_rank": diagnostics["effective_raw_rank"],
        "orthogonality_max_abs": diagnostics["orthogonality_max_abs"],
        "hard_discrete_divergence_max_abs": diagnostics[
            "hard_discrete_divergence_max_abs"
        ],
        "raw_mass_min": float(raw_mass.min()),
        "raw_mass_median": float(np.median(raw_mass)),
        "mean_rayleigh_relative_error": float(
            np.mean(
                np.abs(learned_values - exact_values[assignment])
                / exact_values[assignment]
            )
        ),
        "mean_matched_velocity_relative_l2": float(np.mean(individual_mode_errors)),
        "mean_eigenspace_aligned_velocity_relative_l2": float(
            np.mean(eigenspace_aligned_errors)
        ),
        "mean_matched_absolute_correlation": float(np.mean(np.abs(matched_correlation))),
        "first_k_subspace_rms_sine": rms_sine,
        "mean_first_k_spectral_capture": float(np.mean(first_capture)),
        "mean_candidate_spectral_capture": float(np.mean(candidate_capture)),
        "mode_metrics": rows,
    }
    run_dir = output / activation
    (run_dir / "metrics.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    with (run_dir / "mode_metrics.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return summary, eigenspace_aligned_potential, matched_exact_potential


def plot_training(histories: dict[str, list[dict[str, object]]], output: Path) -> None:
    colors = {"sine": "#2166AC", "elu": "#B2182B"}
    fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.2), constrained_layout=True)
    for activation, history in histories.items():
        steps = [int(row["step"]) for row in history]
        axes[0].plot(
            steps,
            [float(row["validation_loss"]) for row in history],
            color=colors[activation],
            label=activation.upper(),
        )
        axes[1].plot(
            steps,
            [float(row["mean_rayleigh"]) for row in history],
            color=colors[activation],
            label=activation.upper(),
        )
    axes[0].set(title="Validation objective", xlabel="Step", ylabel="Mean R / 50")
    axes[1].set(title="Dirichlet energy", xlabel="Step", ylabel="Mean Rayleigh")
    for axis in axes:
        axis.grid(alpha=0.2)
        axis.legend(frameon=False)
    save_figure(fig, output / "training_ablation")


def plot_modes(
    learned: dict[str, np.ndarray],
    exact: dict[str, np.ndarray],
    summaries: dict[str, dict[str, object]],
    args: argparse.Namespace,
    output: Path,
) -> None:
    cmap = LinearSegmentedColormap.from_list(
        "nature_redblue", ["#2166AC", "#F7F7F7", "#B2182B"], N=256
    )
    activations = list(learned)
    columns = [item for name in activations for item in (f"{name} analytic match", f"{name} learned")]
    fig, axes = plt.subplots(
        args.modes,
        len(columns),
        figsize=(2.35 * len(columns), 2.05 * args.modes),
        constrained_layout=True,
    )
    for mode in range(args.modes):
        fields = [
            item
            for name in activations
            for item in (exact[name][:, mode], learned[name][:, mode])
        ]
        scale = max(np.percentile(np.abs(field), 99.0) for field in fields)
        for column, (name, field) in enumerate(zip(columns, fields, strict=True)):
            axis = axes[mode, column]
            axis.imshow(
                field.reshape(args.eval_grid, args.eval_grid),
                origin="lower",
                extent=(0, 1, 0, 1),
                cmap=cmap,
                vmin=-scale,
                vmax=scale,
                interpolation="bilinear",
            )
            if mode == 0:
                axis.set_title(name.upper())
            if column == 0:
                matches = ", ".join(
                    str(summaries[name]["mode_metrics"][mode]["matched_analytic_pair"])
                    for name in activations
                )
                axis.set_ylabel(f"causal {mode + 1}\n{matches}")
            axis.set_xticks([])
            axis.set_yticks([])
    fig.suptitle("Unit-square streamfunction modes: analytic vs learned")
    save_figure(fig, output / "square_modes_analytic_sine_elu")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output")
    parser.add_argument("--activations", nargs="+", choices=("sine", "elu"), default=("sine", "elu"))
    parser.add_argument("--steps", type=int, default=400)
    parser.add_argument("--points", type=int, default=512)
    parser.add_argument("--validation-points", type=int, default=1024)
    parser.add_argument("--modes", type=int, default=8)
    parser.add_argument("--hidden", type=int, default=32)
    parser.add_argument("--layers", type=int, default=3)
    parser.add_argument("--frequencies", type=int, default=4)
    parser.add_argument("--difference-step", type=float, default=0.01)
    parser.add_argument("--learning-rate", type=float, default=3.0e-4)
    parser.add_argument("--print-every", type=int, default=20)
    parser.add_argument("--eval-grid", type=int, default=65)
    parser.add_argument("--candidate-modes", type=int, default=36)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--evaluate-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch.set_num_threads(args.threads)
    stamp = time.strftime("run_%Y%m%d_%H%M%S")
    output = Path(args.output) if args.output else CASE_ROOT / "results" / stamp
    output.mkdir(parents=True, exist_ok=True)
    if not args.evaluate_only:
        (output / "config.json").write_text(
            json.dumps(vars(args), indent=2), encoding="utf-8"
        )
    histories: dict[str, list[dict[str, object]]] = {}
    summaries: dict[str, dict[str, object]] = {}
    potentials: dict[str, np.ndarray] = {}
    exact: dict[str, np.ndarray] = {}
    for activation in args.activations:
        if args.evaluate_only:
            config = model_config(args, activation)
            model = BasisNet(config).to(torch.device(args.device))
            checkpoint = torch.load(
                output / activation / "checkpoints" / "best.pt",
                map_location=args.device,
            )
            model.load_state_dict(checkpoint["model"])
            with (output / activation / "history.csv").open(
                newline="", encoding="utf-8"
            ) as handle:
                history = list(csv.DictReader(handle))
        else:
            model, history = train_one(activation, args, output)
        summary, aligned, matched_exact = evaluate_one(activation, model, args, output)
        histories[activation] = history
        summaries[activation] = summary
        potentials[activation] = aligned
        exact[activation] = matched_exact
    plot_training(histories, output)
    if exact:
        plot_modes(potentials, exact, summaries, args, output)
    (output / "summary.json").write_text(
        json.dumps(summaries, indent=2), encoding="utf-8"
    )
    print(str(output.resolve()), flush=True)


if __name__ == "__main__":
    main()
