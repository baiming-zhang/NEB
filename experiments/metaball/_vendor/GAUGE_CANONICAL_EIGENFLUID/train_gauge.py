







from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import json
import math
from _neb_paths import Path
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from threadpoolctl import threadpool_limits

from gauge_core import Complex, CONFIG, ROOT, THEORY, geometry_mask, json_write, legacy
from verify_laplacian import reconstruction


class GaugeProjection(torch.autograd.Function):
    @staticmethod
    def forward(ctx, values, complex_):
        ctx.complex_ = complex_
        p = complex_.project(values.detach().cpu().double().numpy())
        return torch.as_tensor(p, dtype=values.dtype, device=values.device)

    @staticmethod
    def backward(ctx, gradient):


        p = ctx.complex_.project(gradient.detach().cpu().double().numpy())
        return torch.as_tensor(p, dtype=gradient.dtype, device=gradient.device), None


def torch_sparse(matrix, device, dtype):
    coo = matrix.tocoo()
    return torch.sparse_coo_tensor(
        np.vstack((coo.row, coo.col)), coo.data, coo.shape, device=device, dtype=dtype
    ).coalesce()


class Geometry:
    def __init__(self, cfg, n, token, device):
        mask, self.geometry = geometry_mask(n, token)
        self.complex = Complex(mask)
        self.token = token
        points = torch.tensor(
            self.complex.edge_points, device=device, dtype=torch.float32
        )
        geo = torch.tensor(self.geometry, device=device, dtype=torch.float32)
        self.features, _ = legacy().normalized_features(cfg, points, geo)


        outer = 1 - torch.sum(
            ((points - 0.5) / cfg.outer_half_extent) ** cfg.outer_superellipsoid_power,
            dim=1,
        )
        distances = (
            torch.linalg.vector_norm(points[:, None] - geo[None, :, :3], dim=2)
            - geo[None, :, 3]
        )
        obstacle = -cfg.metaball_softmin_tau * torch.logsumexp(
            -distances / cfg.metaball_softmin_tau, dim=1
        )
        outer, obstacle = outer.clamp_min(0), obstacle.clamp_min(0)
        self.envelope = (
            outer
            / (cfg.outer_envelope_transition + outer)
            * obstacle
            / (cfg.obstacle_envelope_transition + obstacle)
        )
        self.axes = torch.tensor(self.complex.edges[:, 0], device=device)
        self.rows = torch.arange(len(self.axes), device=device)
        self.curl = torch_sparse(self.complex.C, device, torch.float32)

    def predict(self, network):
        value = self.features
        for i, layer in enumerate(network.layers):
            omega = (
                network.cfg.siren_first_omega0
                if i == 0
                else network.cfg.siren_hidden_omega0
            )
            value = torch.sin(omega * layer(value))
        raw = self.envelope * network.output(value)[self.rows, self.axes]
        return GaugeProjection.apply(raw, self.complex)


def step(model, geometry, optimizers=None):
    k = len(model.modes)
    with torch.no_grad():
        energies = []
        for net in model.modes:
            a = geometry.predict(net)
            ca = torch.mv(geometry.curl, a)
            if a.square().sum() < 1e-18:
                raise FloatingPointError("Canonical candidate has zero norm")
            energies.append((ca.square().sum() / a.square().sum()).item())
        order = np.argsort(energies, kind="stable")
    q = torch.empty((len(geometry.axes), k), device=geometry.features.device)
    losses = []
    for rank, index in enumerate(order):
        a = geometry.predict(model.modes[index])
        a = a / torch.linalg.vector_norm(a)
        if rank:
            previous = q[:, :rank].detach()
            for _ in range(2):
                a = a - previous @ (previous.T @ a)
        mass = a.square().sum()
        if not torch.isfinite(mass) or mass.item() < 1e-12:
            raise FloatingPointError(f"Canonical MGS rank collapse: {rank}")
        a = a / torch.sqrt(mass)
        curl = torch.mv(geometry.curl, a)
        loss = curl.square().sum()
        if not torch.isfinite(loss):
            raise FloatingPointError("Nonfinite canonical Rayleigh quotient")
        if optimizers is not None:
            opt = optimizers[index]
            opt.zero_grad(set_to_none=True)
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(model.modes[index].parameters(), 8.0)
            if not torch.isfinite(norm):
                raise FloatingPointError("Nonfinite projected gradient")
            opt.step()
        q[:, rank].copy_(a.detach())
        losses.append(loss.item())
    return dict(
        mean_rayleigh=float(np.mean(losses)),
        min_rayleigh=min(losses),
        max_rayleigh=max(losses),
        potential_gram_max=float(
            (q.T @ q - torch.eye(k, device=q.device)).abs().max().item()
        ),
    ), q


def evaluate(model, geom, reference):
    with torch.no_grad():
        metrics, q = step(model, geom)
    c = geom.complex
    a = c.project(q.cpu().double().numpy())
    a = np.linalg.qr(a)[0]
    ca = c.C @ a
    lam, rotation = np.linalg.eigh(ca.T @ ca)
    a = a @ rotation
    u = c.C @ a / np.sqrt(lam)
    ref_lam, ref_a, ref_u = reference
    singular = np.linalg.svd(ref_a.T @ a, compute_uv=False)
    metrics.update(
        ritz_eigenvalues=lam.tolist(),
        reference_eigenvalues=ref_lam.tolist(),
        mean_eigenvalue_relative_error=float(np.mean(abs(lam - ref_lam) / ref_lam)),
        potential_subspace_projector_frobenius=float(
            np.sqrt(max(0, 2 * len(lam) - 2 * np.sum(singular**2)))
        ),
        gauge_residual_max=float(
            np.max(np.linalg.norm(c.G.T @ a, axis=0) / np.sqrt(lam))
        ),
        velocity_residual_max=float(
            np.max(np.linalg.norm(c.C @ (c.C.T @ u) - u * lam, axis=0) / lam)
        ),
        velocity_gram_frobenius=float(np.linalg.norm(u.T @ u - np.eye(len(lam)))),
    )
    return metrics, (lam, a, u)


def checkpoint(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp.pt")
    torch.save(payload, tmp)
    tmp.replace(path)


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(20260907)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    old = legacy()
    cfg = replace(
        old.load_config(CONFIG, {}),
        modes=args.modes,
        hidden=args.hidden,
        hidden_layers=args.depth,
        siren_first_omega0=args.omega,
        epochs=args.epochs,
    )
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    protocol = dict(
        name="GAUGE-CANONICAL EIGENFLUID",
        phase="compatible-grid neural pilot",
        resolution=args.resolution,
        modes=args.modes,
        epochs=args.epochs,
        hidden=args.hidden,
        hidden_layers=args.depth,
        first_omega=args.omega,
        hidden_omega=cfg.siren_hidden_omega0,
        seed=20260907,
        training_geometry_tokens=args.tokens,
        validation_geometry_token=12000001,
        validation_role="training geometry (fixed-case optimization check)"
        if 12000001 in args.tokens
        else "unseen geometry",
        theory=str(THEORY),
        theory_sha256=hashlib.sha256(THEORY.read_bytes()).hexdigest(),
        objective="sum ||C P_gf A||^2 / ||P_gf A||^2 after canonical normalization and causal two-pass MGS",
        projection="Dirichlet scalar Poisson plus all relative boundary harmonic gradients; implicit self-adjoint backward",
        derivatives="compatible discrete incidence curl on voxel grid; continuum AD/MC implementation is a separate full-resolution stage",
        boundary="zero relative edge trace and zero normal face flux on voxel boundary",
        training_labels="none; DNS and eigensolver reference used only for evaluation",
        geometry_family="same connected three-sphere Victoria-v1 prior as the 50.5-hour campaign",
        initialization="fresh independent SIRENs; original raw-potential checkpoint not relabeled",
    )
    json_write(output / "protocol.json", protocol)
    model = old.BasisModel(cfg).to(device)
    optimizers = [torch.optim.Adam(net.parameters(), lr=args.lr) for net in model.modes]
    print("Preparing geometry Poisson factors", flush=True)
    training = [Geometry(cfg, args.resolution, token, device) for token in args.tokens]
    validation = Geometry(cfg, args.resolution, 12000001, device)
    reference = validation.complex.spectrum(args.modes)
    start = time.perf_counter()
    history = []
    initial_epoch = 0
    if args.resume and (output / "checkpoints/latest.pt").exists():
        saved = torch.load(
            output / "checkpoints/latest.pt", weights_only=False, map_location=device
        )
        for key in (
            "resolution",
            "modes",
            "hidden",
            "hidden_layers",
            "first_omega",
            "training_geometry_tokens",
            "theory_sha256",
        ):
            if saved["protocol"][key] != protocol[key]:
                raise ValueError(f"Resume contract changed: {key}")
        model.load_state_dict(saved["model"])
        for opt, state in zip(optimizers, saved["optimizers"]):
            opt.load_state_dict(state)
        history = saved["history"]
        initial_epoch = saved["epoch"]
    else:
        initial, _ = evaluate(model, validation, reference)
        json_write(output / "initial_validation.json", initial)
        print("Initial validation", initial["mean_rayleigh"], flush=True)
    json_write(
        output / "status.json",
        dict(status="running", epoch=initial_epoch, protocol=protocol),
    )
    for epoch in range(initial_epoch + 1, args.epochs + 1):
        epoch_start = time.perf_counter()
        lr = 1e-6 + 0.5 * (args.lr - 1e-6) * (
            1 + math.cos(math.pi * (epoch - 1) / max(args.epochs - 1, 1))
        )
        for opt in optimizers:
            for group in opt.param_groups:
                group["lr"] = lr
        geom = training[(epoch - 1) % len(training)]
        metrics, _ = step(model, geom, optimizers)
        row = dict(epoch=epoch, geometry_token=geom.token, lr=lr, **metrics)
        if epoch % args.validate_every == 0 or epoch == args.epochs:
            evaluation, _ = evaluate(model, validation, reference)
            row["validation"] = evaluation
            json_write(output / "latest_validation.json", evaluation)
        torch.cuda.synchronize() if device.type == "cuda" else None
        row["seconds"] = time.perf_counter() - epoch_start
        history.append(row)
        json_write(output / "history.json", history)
        json_write(
            output / "status.json",
            dict(
                status="running",
                epoch=epoch,
                epochs=args.epochs,
                seconds_this_session=time.perf_counter() - start,
                last_epoch=row,
                estimated_remaining_seconds=(args.epochs - epoch)
                * np.mean([r["seconds"] for r in history[-20:]]),
            ),
        )
        if epoch % 10 == 0 or epoch == 1:
            print(
                f"epoch {epoch}/{args.epochs} train={metrics['mean_rayleigh']:.5f} sec={row['seconds']:.3f}",
                flush=True,
            )
        if epoch % args.checkpoint_every == 0 or epoch == args.epochs:
            checkpoint(
                output / "checkpoints/latest.pt",
                dict(
                    epoch=epoch,
                    model=model.state_dict(),
                    optimizers=[opt.state_dict() for opt in optimizers],
                    history=history,
                    protocol=protocol,
                    config=cfg.__dict__,
                ),
            )
    evaluation, (lam, a, u) = evaluate(model, validation, reference)
    records, target, recon, fit, test = reconstruction(validation.complex, u)
    reference_records, *_ = reconstruction(validation.complex, reference[2])
    np.savez_compressed(
        output / "evaluation_modes.npz",
        eigenvalues=lam,
        canonical_potential=a / validation.complex.h**1.5,
        velocity=u / validation.complex.h**1.5,
        mass_per_dof=validation.complex.h**3,
        mask=validation.complex.mask,
        potential_edges=validation.complex.edges,
        velocity_faces=validation.complex.faces,
        geometry_xyzr=validation.geometry,
        gt_cell_velocity=target,
        reconstructed_cell_velocity=recon,
        fit_cells=fit,
        test_cells=test,
    )
    result = dict(
        status="complete",
        epoch=args.epochs,
        protocol=protocol,
        validation=evaluation,
        neural_reconstruction=records,
        laplacian_reconstruction=reference_records,
        initial=json.loads((output / "initial_validation.json").read_text()),
        training_seconds_sum=sum(r["seconds"] for r in history),
        session_seconds=time.perf_counter() - start,
    )
    json_write(output / "completion.json", result)
    json_write(
        output / "status.json",
        dict(
            status="complete", epoch=args.epochs, result=str(output / "completion.json")
        ),
    )
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.5), constrained_layout=True)
    axes[0].plot(
        [r["epoch"] for r in history],
        [r["mean_rayleigh"] for r in history],
        color="#3B6FB6",
        label="Train geometry",
    )
    vh = [r for r in history if "validation" in r]
    axes[0].plot(
        [r["epoch"] for r in vh],
        [r["validation"]["mean_rayleigh"] for r in vh],
        color="#D1495B",
        label="Evaluation geometry",
    )
    axes[0].axhline(
        np.mean(reference[0]),
        color="#243447",
        ls="--",
        label="Evaluation Laplacian reference",
    )
    axes[0].set(xlabel="Epoch", ylabel="Mean canonical Rayleigh", yscale="log")
    axes[0].legend(frameon=False)
    axes[1].plot(reference[0], color="#3B6FB6", label="Laplacian")
    axes[1].plot(lam, color="#D1495B", label="Fresh neural pilot")
    axes[1].set(xlabel="Mode index (0-based)", ylabel="Evaluation Ritz eigenvalue")
    axes[1].legend(frameon=False)
    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle(
        f"GAUGE-CANONICAL EIGENFLUID | K={args.modes} | {args.resolution}³ discrete pilot"
    )
    for ext in ("png", "pdf"):
        fig.savefig(output / f"training_summary.{ext}", dpi=200)
    plt.close(fig)
    print(
        json.dumps(
            dict(
                status="complete",
                output=str(output),
                seconds=result["training_seconds_sum"],
                mean_eigenvalue_relative_error=evaluation[
                    "mean_eigenvalue_relative_error"
                ],
            )
        ),
        flush=True,
    )
