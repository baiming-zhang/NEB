"""Native-CuPy P1 full-order vorticity-streamfunction Navier--Stokes solver."""
from __future__ import annotations

from dataclasses import dataclass
import math
import time

import cupy as cp
import numpy as np
from cupyx.scipy.linalg import solve_triangular

from .geometry import fluid_mask, obstacle_boundary, outer_boundary
from .mesh_builder import DomainMesh, mesh_diagnostics


def scalar(value) -> float:
    return float(value.item())


def solve_cholesky(cholesky, rhs):
    return solve_triangular(
        cholesky.T,
        solve_triangular(cholesky, rhs, lower=True),
        lower=False,
    )


@dataclass
class InitialSpecification:
    seed: int
    rng: str
    centers: np.ndarray
    radii: np.ndarray
    amplitudes: np.ndarray
    rejected_draws: int
    harmonic_trace_coefficients: np.ndarray
    harmonic_to_vortical_energy_ratio: float
    target_peak_speed: float

    def as_dict(self) -> dict:
        return {
            "seed": int(self.seed),
            "rng": self.rng,
            "vortex_profile": "compact Wendland C4",
            "vortex_count": int(len(self.centers)),
            "centers": self.centers.tolist(),
            "support_radii": self.radii.tolist(),
            "amplitudes": self.amplitudes.tolist(),
            "rejected_draws": int(self.rejected_draws),
            "harmonic_trace_coefficients": self.harmonic_trace_coefficients.tolist(),
            "harmonic_to_vortical_energy_ratio": float(self.harmonic_to_vortical_energy_ratio),
            "target_peak_speed": float(self.target_peak_speed),
            "constructed_from_modes": False,
        }


class TopologyNS:
    """Full nodal P1 NS state with one circulation coordinate per obstacle."""

    def __init__(self, mesh: DomainMesh, viscosity: float, pool_limit_bytes: int = 5 * 1024**3):
        self.mesh = mesh
        self.nu = float(viscosity)
        cp.get_default_memory_pool().set_limit(size=pool_limit_bytes)
        self.xy = cp.asarray(mesh.xy, dtype=cp.float64)
        self.tri = cp.asarray(mesh.tri, dtype=cp.int32)
        self.n = len(mesh.xy)
        self.boundary_nodes = tuple(cp.asarray(nodes, dtype=cp.int32) for nodes in mesh.boundary_nodes)
        self.harmonic_dimension = len(self.boundary_nodes) - 1
        self.D = cp.asarray(np.unique(mesh.boundary_edges), dtype=cp.int32)
        free_mask = cp.ones(self.n, dtype=cp.bool_)
        free_mask[self.D] = False
        self.F = cp.where(free_mask)[0]

        points = self.xy[self.tri]
        det = ((points[:, 1, 0] - points[:, 0, 0]) * (points[:, 2, 1] - points[:, 0, 1])
               - (points[:, 2, 0] - points[:, 0, 0]) * (points[:, 1, 1] - points[:, 0, 1]))
        if not bool(cp.all(det > 0.0).item()):
            raise RuntimeError("mesh contains non-positive triangles")
        self.area = 0.5 * det
        self.grad = cp.stack([
            cp.stack((points[:, 1, 1] - points[:, 2, 1], points[:, 2, 0] - points[:, 1, 0]), 1),
            cp.stack((points[:, 2, 1] - points[:, 0, 1], points[:, 0, 0] - points[:, 2, 0]), 1),
            cp.stack((points[:, 0, 1] - points[:, 1, 1], points[:, 1, 0] - points[:, 0, 0]), 1),
        ], 1) / det[:, None, None]
        local_s = cp.einsum("tid,tjd,t->tij", self.grad, self.grad, self.area)
        local_m = self.area[:, None, None] * (
            cp.ones((3, 3), dtype=cp.float64) + cp.eye(3, dtype=cp.float64)
        ) / 12.0
        rows = cp.broadcast_to(self.tri[:, :, None], local_s.shape).ravel()
        cols = cp.broadcast_to(self.tri[:, None, :], local_s.shape).ravel()
        self.S = cp.zeros((self.n, self.n), dtype=cp.float64)
        self.M = cp.zeros_like(self.S)
        cp.add.at(self.S, (rows, cols), local_s.ravel())
        cp.add.at(self.M, (rows, cols), local_m.ravel())
        self.Sf = self.S[cp.ix_(self.F, self.F)]
        self.Mf = self.M[cp.ix_(self.F, self.F)]
        self.LS = cp.linalg.cholesky(self.Sf)
        self.LM = cp.linalg.cholesky(self.Mf)
        self.fluid_area = cp.sum(self.area)

        self.H = cp.zeros((self.n, self.harmonic_dimension), dtype=cp.float64)
        for index in range(self.harmonic_dimension):
            self.H[self.boundary_nodes[index + 1], index] = 1.0
            self.H[self.F, index] = solve_cholesky(
                self.LS,
                -(self.S @ self.H[:, index])[self.F],
            )
        self.period = cp.empty((self.harmonic_dimension, self.harmonic_dimension), dtype=cp.float64)
        for i in range(self.harmonic_dimension):
            residual = self.S @ self.H
            self.period[i] = -cp.sum(residual[self.boundary_nodes[i + 1]], axis=0)
        self.harmonic_gram = self.H.T @ self.S @ self.H
        self.H_unit_circulation = self.H @ cp.linalg.inv(self.period)

        edge_owner = {}
        tri_cpu = mesh.tri
        for cell, vertices in enumerate(tri_cpu):
            for a, b in ((vertices[0], vertices[1]), (vertices[1], vertices[2]), (vertices[2], vertices[0])):
                edge_owner[tuple(sorted((int(a), int(b))))] = cell
        owners = np.asarray([edge_owner[tuple(sorted(map(int, edge)))] for edge in mesh.boundary_edges], dtype=np.int32)
        self.boundary_edges = cp.asarray(mesh.boundary_edges, dtype=cp.int32)
        self.boundary_edge_labels = cp.asarray(mesh.boundary_edge_labels, dtype=cp.int32)
        self.boundary_edge_cells = cp.asarray(owners, dtype=cp.int32)
        edge_points = self.xy[self.boundary_edges]
        tangent_raw = edge_points[:, 1] - edge_points[:, 0]
        self.edge_length = cp.linalg.norm(tangent_raw, axis=1)
        normal = cp.stack((tangent_raw[:, 1], -tangent_raw[:, 0]), axis=1) / self.edge_length[:, None]
        midpoint = cp.mean(edge_points, axis=1)
        centroid = cp.mean(self.xy[self.tri[self.boundary_edge_cells]], axis=1)
        sign = cp.where(cp.sum(normal * (midpoint - centroid), axis=1) >= 0.0, 1.0, -1.0)
        self.edge_normal = normal * sign[:, None]

    def boundary_sums(self, values):
        return cp.stack([cp.sum(values[nodes]) for nodes in self.boundary_nodes[1:]])

    def velocity(self, psi):
        gradient = cp.einsum("tjd,tj...->td...", self.grad, psi[self.tri])
        return cp.stack((gradient[:, 1], -gradient[:, 0]), axis=1)

    def scatter(self, local):
        result = cp.zeros(self.n, dtype=cp.float64)
        cp.add.at(result, self.tri.ravel(), local.ravel())
        return result

    def advection(self, omega, psi):
        velocity = self.velocity(psi)
        grad_omega = cp.einsum("tjd,tj->td", self.grad, omega[self.tri])
        value = cp.sum(velocity * grad_omega, axis=1) * self.area / 3.0
        return self.scatter(cp.broadcast_to(value[:, None], self.tri.shape))

    def zero_trace_potential(self, omega):
        psi = cp.zeros(self.n, dtype=cp.float64)
        psi[self.F] = solve_cholesky(self.LS, (self.M @ omega)[self.F])
        return psi

    def gamma_from_state(self, psi, omega):
        residual = self.S @ psi - self.M @ omega
        return -self.boundary_sums(residual)

    def reconstruct(self, omega, gamma):
        psi_zero = self.zero_trace_potential(omega)
        gamma_zero = self.gamma_from_state(psi_zero, omega)
        coefficients = cp.linalg.solve(self.period, gamma - gamma_zero)
        return psi_zero + self.H @ coefficients

    def time_derivative(self, omega, psi):
        advection = self.advection(omega, psi)
        domega = cp.zeros(self.n, dtype=cp.float64)
        domega[self.F] = solve_cholesky(
            self.LM,
            (-advection - self.nu * (self.S @ omega))[self.F],
        )
        dgamma = self.boundary_sums(self.M @ domega + advection + self.nu * (self.S @ omega))
        dpsi_zero = self.zero_trace_potential(domega)
        dgamma_zero = self.gamma_from_state(dpsi_zero, domega)
        dpsi = dpsi_zero + self.H @ cp.linalg.solve(self.period, dgamma - dgamma_zero)
        return domega, dgamma, dpsi

    def harmonic_diagnostics(self) -> dict:
        residual = self.S @ self.H
        scale = cp.maximum(cp.linalg.norm(self.Sf @ self.H[self.F], axis=0), 1e-30)
        free_residual = cp.linalg.norm(residual[self.F], axis=0)
        traces = []
        for basis_index in range(self.harmonic_dimension):
            traces.append([
                {
                    "mean": scalar(cp.mean(self.H[nodes, basis_index])),
                    "spread": scalar(cp.max(self.H[nodes, basis_index]) - cp.min(self.H[nodes, basis_index])),
                }
                for nodes in self.boundary_nodes
            ])
        boundary = [self.boundary_diagnostics(self.H[:, index], cp.zeros(self.n))
                    for index in range(self.harmonic_dimension)]
        return {
            "weak_laplacian_free_residual_absolute": cp.asnumpy(free_residual).tolist(),
            "weak_laplacian_free_residual_scaled": cp.asnumpy(free_residual / scale).tolist(),
            "boundary_traces_by_basis": traces,
            "period_matrix": cp.asnumpy(self.period).tolist(),
            "period_rank": int(np.linalg.matrix_rank(cp.asnumpy(self.period))),
            "period_condition_number": scalar(cp.linalg.cond(self.period)),
            "energy_gram": cp.asnumpy(self.harmonic_gram).tolist(),
            "boundary_diagnostics_by_basis": boundary,
        }

    def poisson_residual(self, psi, omega, gamma) -> float:
        reconstructed = self.reconstruct(omega, gamma)
        difference = reconstructed - psi
        numerator = cp.sqrt(difference @ self.S @ difference)
        denominator = cp.sqrt(cp.maximum(psi @ self.S @ psi, 1e-30))
        return scalar(numerator / denominator)

    def boundary_diagnostics(self, psi, omega) -> dict:
        velocity = self.velocity(psi)
        edge_velocity = velocity[self.boundary_edge_cells]
        tangent = cp.stack((-self.edge_normal[:, 1], self.edge_normal[:, 0]), axis=1)
        normal_velocity = cp.sum(edge_velocity * self.edge_normal, axis=1)
        tangential_velocity = cp.sum(edge_velocity * tangent, axis=1)
        component_normal = []
        line_circulation = []
        for component in range(len(self.boundary_nodes)):
            selected = self.boundary_edge_labels == component
            component_normal.append(scalar(cp.max(cp.abs(normal_velocity[selected]))))
            line_circulation.append(scalar(cp.sum(self.edge_length[selected] * tangential_velocity[selected])))
        trace = []
        omega_trace = []
        for nodes in self.boundary_nodes:
            values = psi[nodes]
            trace.append({
                "mean": scalar(cp.mean(values)),
                "spread": scalar(cp.max(values) - cp.min(values)),
            })
            omega_trace.append(scalar(cp.max(cp.abs(omega[nodes]))))
        gamma_weak = cp.asnumpy(self.gamma_from_state(psi, omega))
        line_inner = np.asarray(line_circulation[1:])
        return {
            "component_normal_velocity_max": component_normal,
            "normal_velocity_max": max(component_normal),
            "psi_trace": trace,
            "psi_trace_spread_max": max(item["spread"] for item in trace),
            "omega_trace_max_by_component": omega_trace,
            "omega_trace_max": max(omega_trace),
            "net_flux": scalar(cp.sum(self.edge_length * normal_velocity)),
            "line_circulation": line_circulation,
            "gamma_weak": gamma_weak.tolist(),
            "inner_line_vs_gamma_max_abs": float(np.max(np.abs(line_inner - gamma_weak))),
        }

    def energy_diagnostic(self, psi, omega) -> dict:
        _, _, dpsi = self.time_derivative(omega, psi)
        energy = 0.5 * (psi @ self.S @ psi)
        energy_rate = psi @ self.S @ dpsi
        dissipation = self.nu * (omega @ self.M @ omega)
        defect = cp.abs(energy_rate + dissipation) / cp.maximum(
            cp.maximum(cp.abs(energy_rate), dissipation), 1e-30
        )
        return {
            "kinetic_energy": scalar(energy),
            "energy_rate": scalar(energy_rate),
            "viscous_dissipation": scalar(dissipation),
            "semidiscrete_relative_balance_defect": scalar(defect),
        }

    def tendency_diagnostic(self, psi, omega) -> dict:
        advection = self.advection(omega, psi)[self.F]
        viscous = (self.nu * (self.S @ omega))[self.F]
        adv_rate = solve_cholesky(self.LM, -advection)
        visc_rate = solve_cholesky(self.LM, -viscous)
        adv_norm = cp.sqrt(adv_rate @ self.Mf @ adv_rate)
        visc_norm = cp.sqrt(visc_rate @ self.Mf @ visc_rate)
        return {
            "nonlinear_vorticity_tendency_L2": scalar(adv_norm),
            "viscous_vorticity_tendency_L2": scalar(visc_norm),
            "nonlinear_to_viscous_ratio": scalar(adv_norm / cp.maximum(visc_norm, 1e-30)),
        }

    def snapshot_diagnostics(self, psi, omega, gamma, dt: float) -> dict:
        speed = cp.linalg.norm(self.velocity(psi), axis=1)
        min_altitude = mesh_diagnostics(self.mesh)["minimum_altitude"]
        return {
            "finite": bool(cp.all(cp.isfinite(psi)).item() and cp.all(cp.isfinite(omega)).item()),
            "peak_speed": scalar(cp.max(speed)),
            "advective_cfl_using_min_altitude": scalar(cp.max(speed)) * dt / min_altitude,
            "poisson_reconstruction_relative_velocity_L2": self.poisson_residual(psi, omega, gamma),
            "boundary": self.boundary_diagnostics(psi, omega),
            "energy": self.energy_diagnostic(psi, omega),
            "tendency": self.tendency_diagnostic(psi, omega),
        }

    def advance(self, omega0, gamma0, dt: float, final_time: float, snapshot_dt: float,
                label: str, timeout_seconds: float = 1700.0):
        steps = int(round(final_time / dt))
        if abs(steps * dt - final_time) > 1e-12:
            raise ValueError("final_time must be an integer multiple of dt")
        stride = max(1, int(round(snapshot_dt / dt)))
        implicit = cp.linalg.cholesky(self.Mf + 0.5 * dt * self.nu * self.Sf)
        omega = omega0.copy()
        gamma = gamma0.copy()
        psi = self.reconstruct(omega, gamma)
        snapshots = []
        started = time.perf_counter()
        for step in range(steps + 1):
            if step % stride == 0 or step == steps:
                snapshots.append((step * dt, psi.copy(), omega.copy(), gamma.copy()))
            if step == steps:
                break
            if time.perf_counter() - started > timeout_seconds:
                raise TimeoutError(f"{label} exceeded bounded {timeout_seconds:.0f} second solve budget")
            advection = self.advection(omega, psi)
            omega_half = cp.zeros(self.n, dtype=cp.float64)
            omega_half[self.F] = solve_cholesky(
                implicit,
                (self.M @ omega - 0.5 * dt * advection)[self.F],
            )
            gamma_half = gamma + self.boundary_sums(
                self.M @ (omega_half - omega)
                + 0.5 * dt * (advection + self.nu * (self.S @ omega_half))
            )
            psi_half = self.reconstruct(omega_half, gamma_half)
            advection_half = self.advection(omega_half, psi_half)
            omega_next = cp.zeros(self.n, dtype=cp.float64)
            omega_next[self.F] = solve_cholesky(
                implicit,
                (self.M @ omega - 0.5 * dt * self.nu * (self.S @ omega) - dt * advection_half)[self.F],
            )
            gamma_next = gamma + self.boundary_sums(
                self.M @ (omega_next - omega)
                + dt * (advection_half + 0.5 * self.nu * (self.S @ (omega_next + omega)))
            )
            omega, gamma = omega_next, gamma_next
            psi = self.reconstruct(omega, gamma)
            if not bool(cp.all(cp.isfinite(omega)).item()):
                raise FloatingPointError(f"non-finite vorticity in {label} at step {step + 1}")
        cp.cuda.Stream.null.synchronize()
        return snapshots, time.perf_counter() - started


def wendland_c4(radius_fraction: np.ndarray) -> np.ndarray:
    s = np.asarray(radius_fraction, dtype=np.float64)
    q = np.maximum(1.0 - s, 0.0)
    return (q**6) * (35.0 * s * s + 18.0 * s + 3.0) / 3.0


def vorticity_from_spec(xy: np.ndarray, initial: InitialSpecification) -> np.ndarray:
    delta = xy[:, None, :] - initial.centers[None, :, :]
    radii = np.linalg.norm(delta, axis=2) / initial.radii[None, :]
    return wendland_c4(radii) @ initial.amplitudes


def state_from_initial_spec(model: TopologyNS, initial: InitialSpecification):
    omega_cpu = vorticity_from_spec(model.mesh.xy, initial)
    omega = cp.asarray(omega_cpu, dtype=cp.float64)
    omega[model.D] = 0.0
    psi_zero = model.zero_trace_potential(omega)
    coefficients = cp.asarray(initial.harmonic_trace_coefficients, dtype=cp.float64)
    psi = psi_zero + model.H @ coefficients
    gamma = model.gamma_from_state(psi, omega)
    reconstructed = model.reconstruct(omega, gamma)
    reconstruction_error = cp.sqrt(
        (reconstructed - psi) @ model.S @ (reconstructed - psi)
        / cp.maximum(psi @ model.S @ psi, 1e-30)
    )
    return omega, gamma, psi, scalar(reconstruction_error)


def make_random_initial(
    model: TopologyNS,
    seed: int,
    vortex_count_range: tuple[int, int] = (3, 7),
    harmonic_energy_ratio: float = 0.12,
    target_peak_speed: float = 0.5,
    support_radius_range: tuple[float, float] = (0.42, 0.68),
) -> tuple[InitialSpecification, tuple]:
    rng = np.random.Generator(np.random.PCG64(seed))
    vortex_count = int(rng.integers(vortex_count_range[0], vortex_count_range[1] + 1))
    boundaries = [outer_boundary(model.mesh.tau, 1024)]
    boundaries.extend(obstacle_boundary(spec, 512) for spec in model.mesh.obstacles)
    boundary_points = np.vstack(boundaries)
    centers, radii, amplitudes = [], [], []
    rejected = 0
    while len(centers) < vortex_count:
        candidate = np.array([rng.uniform(1.45, 8.55), rng.uniform(-3.45, 3.45)])
        support = rng.uniform(*support_radius_range)
        clearance = np.min(np.linalg.norm(boundary_points - candidate, axis=1))
        separation = min([np.linalg.norm(candidate - center) for center in centers], default=math.inf)
        if (not bool(fluid_mask(candidate[None, :], model.mesh.case_name, model.mesh.tau)[0])
                or clearance < support + 0.12 or separation < 0.35):
            rejected += 1
            if rejected > 10000:
                raise RuntimeError("could not sample admissible physical-space vortices")
            continue
        centers.append(candidate)
        radii.append(support)
        sign = float(rng.choice((-1.0, 1.0)))
        amplitudes.append(sign * rng.uniform(0.8, 1.35))
    centers_array = np.asarray(centers)
    radii_array = np.asarray(radii)
    amplitudes_array = np.asarray(amplitudes)
    provisional = InitialSpecification(
        seed=seed,
        rng="NumPy PCG64",
        centers=centers_array,
        radii=radii_array,
        amplitudes=amplitudes_array,
        rejected_draws=rejected,
        harmonic_trace_coefficients=np.zeros(model.harmonic_dimension),
        harmonic_to_vortical_energy_ratio=harmonic_energy_ratio,
        target_peak_speed=target_peak_speed,
    )
    omega = cp.asarray(vorticity_from_spec(model.mesh.xy, provisional), dtype=cp.float64)
    omega[model.D] = 0.0
    psi_zero = model.zero_trace_potential(omega)
    vortical_energy2 = psi_zero @ model.S @ psi_zero
    raw_harmonic = cp.asarray(rng.normal(size=model.harmonic_dimension), dtype=cp.float64)
    raw_harmonic /= cp.linalg.norm(raw_harmonic)
    harmonic_energy2 = raw_harmonic @ model.harmonic_gram @ raw_harmonic
    harmonic_coefficients = raw_harmonic * cp.sqrt(
        harmonic_energy_ratio * vortical_energy2 / cp.maximum(harmonic_energy2, 1e-30)
    )
    psi = psi_zero + model.H @ harmonic_coefficients
    peak_speed = cp.max(cp.linalg.norm(model.velocity(psi), axis=1))
    scale = target_peak_speed / peak_speed
    amplitudes_array *= scalar(scale)
    harmonic_coefficients *= scale
    final = InitialSpecification(
        seed=seed,
        rng="NumPy PCG64",
        centers=centers_array,
        radii=radii_array,
        amplitudes=amplitudes_array,
        rejected_draws=rejected,
        harmonic_trace_coefficients=cp.asnumpy(harmonic_coefficients),
        harmonic_to_vortical_energy_ratio=harmonic_energy_ratio,
        target_peak_speed=target_peak_speed,
    )
    state = state_from_initial_spec(model, final)
    return final, state


def pack_snapshots(snapshots) -> dict[str, np.ndarray]:
    return {
        "times": np.asarray([frame[0] for frame in snapshots], dtype=np.float64),
        "psi": cp.asnumpy(cp.stack([frame[1] for frame in snapshots])),
        "omega": cp.asnumpy(cp.stack([frame[2] for frame in snapshots])),
        "gamma": cp.asnumpy(cp.stack([frame[3] for frame in snapshots])),
    }
