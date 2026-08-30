"""Camera-ready equation reproduction of Neural Kinematic Bases for Fluids.

The SIGGRAPH Asia 2025 authors did not publish training code or weights.  This
script therefore reproduces the camera-ready architecture and six equations on
one fixed airfoil domain.  It deliberately does not use GT velocity data.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
from scipy.stats import qmc


class NeuralKinematicBases(torch.nn.Module):
    def __init__(self, bases: int = 10, width: int = 256, layers: int = 8) -> None:
        super().__init__()
        modules: list[torch.nn.Module] = []
        in_features = 2
        for _ in range(layers - 1):
            linear = torch.nn.Linear(in_features, width)
            torch.nn.init.kaiming_normal_(linear.weight, a=0.01, nonlinearity="leaky_relu")
            torch.nn.init.zeros_(linear.bias)
            modules.extend((linear, torch.nn.LeakyReLU(0.01)))
            in_features = width
        final = torch.nn.Linear(in_features, 2 * bases)
        torch.nn.init.kaiming_normal_(final.weight, nonlinearity="linear")
        torch.nn.init.zeros_(final.bias)
        modules.extend((final, torch.nn.ELU()))
        self.network = torch.nn.Sequential(*modules)
        self.bases = bases

    def forward(self, points: torch.Tensor) -> torch.Tensor:
        return self.network(points).reshape(-1, self.bases, 2)


def body_coordinates(points: torch.Tensor, alpha_deg: float) -> tuple[torch.Tensor, torch.Tensor]:
    angle = math.radians(-alpha_deg)
    c, s = math.cos(angle), math.sin(angle)
    dx = points[:, 0] - 3.0
    dy = points[:, 1]
    return c * dx + s * dy + 0.5, -s * dx + c * dy


def airfoil_level_normal(
    physical: torch.Tensor, beta: float, alpha_deg: float
) -> tuple[torch.Tensor, torch.Tensor]:
    xb, yb = body_coordinates(physical, alpha_deg)
    thickness = 0.08 + (0.18 - 0.08) * beta
    a, b = 1.0, thickness
    level = ((xb - a) / a) ** 2 + (yb / b) ** 2 - 1.0
    grad_body = torch.stack((2.0 * (xb - a) / (a * a), 2.0 * yb / (b * b)), dim=1)
    angle = math.radians(alpha_deg)
    c, s = math.cos(angle), math.sin(angle)
    normal = torch.stack(
        (c * grad_body[:, 0] - s * grad_body[:, 1], s * grad_body[:, 0] + c * grad_body[:, 1]),
        dim=1,
    )
    return level, torch.nn.functional.normalize(normal, dim=1, eps=1.0e-12)


def normalized_to_physical(points: torch.Tensor) -> torch.Tensor:
    return torch.stack((5.0 * (points[:, 0] + 1.0), 4.0 * points[:, 1]), dim=1)


def sample_sobol(count: int, seed: int, device: torch.device) -> torch.Tensor:
    sampler = qmc.Sobol(2, scramble=True, seed=seed)
    values = sampler.random(count).astype(np.float32)
    return torch.as_tensor(2.0 * values - 1.0, device=device)


def gradient_components(
    values: torch.Tensor, points: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    derivatives = []
    for basis in range(values.shape[1]):
        basis_components = []
        for component in range(2):
            derivative = torch.autograd.grad(
                values[:, basis, component].sum(),
                points,
                create_graph=True,
                retain_graph=True,
            )[0]
            basis_components.append(derivative)
        derivatives.append(torch.stack(basis_components, dim=1))
    jacobian = torch.stack(derivatives, dim=1)
    divergence = jacobian[:, :, 0, 0] + jacobian[:, :, 1, 1]
    return jacobian, divergence


def write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--samples-per-epoch", type=int, default=1_000_000)
    parser.add_argument("--batch-size", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=20250812)
    parser.add_argument("--beta", type=float, default=0.5)
    parser.add_argument("--alpha-deg", type=float, default=10.0)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda")
    model = NeuralKinematicBases().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=5.0e-4)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.96)
    start_epoch = 0
    checkpoint = args.output_dir / "latest.pt"
    history_path = args.output_dir / "training_history.json"
    history: list[dict] = []
    if args.resume and checkpoint.exists():
        state = torch.load(checkpoint, map_location=device, weights_only=False)
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        start_epoch = int(state["epoch"])
        if history_path.exists():
            history = json.loads(history_path.read_text(encoding="utf-8"))

    protocol = {
        "paper": "Neural Kinematic Bases for Fluids, SIGGRAPH Asia 2025 camera-ready",
        "reproduction_status": "equation reproduction; author repository publishes no code or weights",
        "adaptation": "single fixed C2 airfoil domain; normalized physical box [0,10]x[-4,4]",
        "architecture": "10 vector bases; 8 fully connected layers; 256 channels; leaky ReLU; final ELU",
        "loss_weights": {"smooth": 0.01, "div": 5.0, "orth": 100.0, "bc": 30.0, "len": 100.0, "small": 100.0},
        "delta": 0.05,
        "target_average_length": 0.37,
        "optimizer": "Adam, lr=0.0005, ExponentialLR gamma=0.96 per epoch",
        "samples_per_epoch": args.samples_per_epoch,
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "ground_truth_supervision": False,
        "boundary": "airfoil slip alignment; open box edges excluded from boundary loss",
        "camera_ready_pdf": str((Path(__file__).resolve().parents[3] / "tmp" / "Neural_Kinematic_Bases_for_Fluids_SIGGRAPH_Asia_2025_camera_ready.pdf")),
    }
    write_json(args.output_dir / "protocol.json", protocol)

    started = time.perf_counter()
    for epoch in range(start_epoch, args.epochs):
        totals = {key: 0.0 for key in ("loss", "smooth", "div", "orth", "bc", "len", "small")}
        batches = math.ceil(args.samples_per_epoch / args.batch_size)
        for batch in range(batches):
            count = min(args.batch_size, args.samples_per_epoch - batch * args.batch_size)
            points = sample_sobol(count, args.seed + epoch * batches + batch, device)
            points.requires_grad_(True)
            physical = normalized_to_physical(points)
            level, normal = airfoil_level_normal(physical, args.beta, args.alpha_deg)
            fluid = level > 0.0
            if int(fluid.sum()) < count // 2:
                raise RuntimeError("unexpectedly few fluid points")
            values = model(points)
            jacobian, _ = gradient_components(values, points)
            # Convert derivatives of normalized coordinates to derivatives in
            # the physical 10-by-8 airfoil box before evaluating Eqs. (2),(7).
            jacobian_physical = jacobian.clone()
            jacobian_physical[:, :, :, 0] /= 5.0
            jacobian_physical[:, :, :, 1] /= 4.0
            divergence = (
                jacobian_physical[:, :, 0, 0]
                + jacobian_physical[:, :, 1, 1]
            )
            fv = values[fluid]
            fj = jacobian_physical[fluid]
            fd = divergence[fluid]
            # Camera-ready Eqs. (2)-(7), evaluated as normalized Monte Carlo means.
            div_loss = (fd * fd).mean()
            smooth_loss = (fj * fj).sum(dim=(2, 3)).mean()
            gram = torch.einsum("nki,nli->kl", fv, fv) / fv.shape[0]
            offdiag = gram - torch.diag(torch.diagonal(gram))
            orth_loss = (offdiag * offdiag).sum() / (model.bases * (model.bases - 1))
            lengths = torch.linalg.vector_norm(fv, dim=2)
            len_loss = ((lengths.mean(dim=0) - 0.37) ** 2).mean()
            small_loss = torch.relu(0.05 - lengths).mean()
            # The paper's boundary indicator is supported in a narrow signed-distance band.
            # level is converted to an approximate physical distance using |grad level|.
            xb, yb = body_coordinates(physical, args.alpha_deg)
            thickness = 0.08 + (0.18 - 0.08) * args.beta
            grad_norm = torch.sqrt((2.0 * (xb - 1.0)) ** 2 + (2.0 * yb / (thickness * thickness)) ** 2).clamp_min(1.0e-8)
            distance = torch.abs(level) / grad_norm
            eps = 0.05
            z = (distance / eps).clamp(0.0, 1.0)
            wb = torch.where(distance <= eps, (2.0 * z**3 - 3.0 * z**2 + 1.0).clamp_min(0.0) ** 4, torch.zeros_like(z))
            cosine = (values * normal[:, None, :]).sum(dim=2) / torch.linalg.vector_norm(values, dim=2).clamp_min(1.0e-8)
            bc_loss = ((cosine * cosine) * wb[:, None]).sum() / (wb.sum().clamp_min(1.0e-8) * model.bases)
            loss = 0.01 * smooth_loss + 5.0 * div_loss + 100.0 * orth_loss + 30.0 * bc_loss + 100.0 * len_loss + 100.0 * small_loss
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            for key, value in (("loss", loss), ("smooth", smooth_loss), ("div", div_loss), ("orth", orth_loss), ("bc", bc_loss), ("len", len_loss), ("small", small_loss)):
                totals[key] += float(value.detach())
            if (batch + 1) % 10 == 0 or batch + 1 == batches:
                print(f"[epoch {epoch + 1:02d}/{args.epochs:02d} batch {batch + 1:03d}/{batches:03d}] loss={float(loss.detach()):.6e}", flush=True)
        scheduler.step()
        row = {"epoch": epoch + 1, "lr": float(optimizer.param_groups[0]["lr"]), "elapsed_seconds": time.perf_counter() - started}
        row.update({key: value / batches for key, value in totals.items()})
        history.append(row)
        write_json(history_path, history)
        state = {"epoch": epoch + 1, "model": model.state_dict(), "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(), "protocol": protocol}
        torch.save(state, checkpoint)
        torch.save(state, args.output_dir / f"epoch{epoch + 1:02d}.pt")
        print(json.dumps(row), flush=True)
    write_json(args.output_dir / "completion.json", {"status": "complete", "epoch": args.epochs, "elapsed_seconds": time.perf_counter() - started})


if __name__ == "__main__":
    main()
