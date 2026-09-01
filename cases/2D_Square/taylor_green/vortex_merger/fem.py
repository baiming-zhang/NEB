"""Mass-lumped P1 finite-element vorticity-streamfunction Navier--Stokes."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.sparse.linalg import splu
from skfem import Basis, MeshTri, asm
from skfem.element import ElementTriP1
from skfem.models.poisson import laplace, mass

FloatArray = NDArray[np.float64]


@dataclass
class FEMDiagnostics:
    time: float
    kinetic_energy: float
    enstrophy: float
    circulation: float
    max_speed: float


class VorticityStreamfunctionFEM:
    """P1 FEM solver with no-penetration/free-slip square boundaries.

    ``psi=0`` on the outer boundary enforces ``u.n=0``.  The vorticity
    diffusion term receives its natural homogeneous Neumann condition.
    Advection is evaluated elementwise and the vorticity mass is lumped.
    """

    def __init__(self, cells_per_side: int, viscosity: float, length: float = np.pi):
        if cells_per_side < 8:
            raise ValueError("cells_per_side must be at least 8")
        self.cells_per_side = int(cells_per_side)
        self.viscosity = float(viscosity)
        self.length = float(length)
        axis = np.linspace(0.0, self.length, self.cells_per_side + 1)
        self.mesh = MeshTri.init_tensor(axis, axis)
        self.basis = Basis(self.mesh, ElementTriP1())
        self.mass = asm(mass, self.basis).tocsr()
        self.stiffness = asm(laplace, self.basis).tocsr()
        self.lumped_mass = np.asarray(self.mass.sum(axis=1)).ravel()
        self.boundary = self.basis.get_dofs().all()
        self.interior = np.setdiff1d(np.arange(self.basis.N), self.boundary)
        self._poisson_lu = splu(self.stiffness[self.interior][:, self.interior].tocsc())
        self._prepare_elements()

    @property
    def x(self) -> FloatArray:
        return np.linspace(0.0, self.length, self.cells_per_side + 1)

    @property
    def y(self) -> FloatArray:
        return np.linspace(0.0, self.length, self.cells_per_side + 1)

    def _prepare_elements(self) -> None:
        tri = self.mesh.t
        px = self.mesh.p[0, tri]
        py = self.mesh.p[1, tri]
        det = (px[1] - px[0]) * (py[2] - py[0]) - (px[2] - px[0]) * (py[1] - py[0])
        self.area = 0.5 * np.abs(det)
        self.grad_x = np.vstack((py[1] - py[2], py[2] - py[0], py[0] - py[1])) / det
        self.grad_y = np.vstack((px[2] - px[1], px[0] - px[2], px[1] - px[0])) / det

    def tensor_to_dofs(self, values: FloatArray) -> FloatArray:
        values = np.asarray(values, dtype=np.float64)
        if values.shape != (self.x.size, self.y.size):
            raise ValueError("tensor field shape does not match the FEM grid")
        # MeshTri.init_tensor orders vertices with y as the fast coordinate.
        return values.reshape(-1)

    def dofs_to_tensor(self, values: FloatArray) -> FloatArray:
        return np.asarray(values, dtype=np.float64).reshape(self.x.size, self.y.size)

    def initial_two_vortices(self, peak_vorticity: float, core_radius: float, separation: float) -> FloatArray:
        xx, yy = np.meshgrid(self.x, self.y, indexing="ij")
        cx = self.length / 2.0
        cy = self.length / 2.0
        offsets = (-separation / 2.0, separation / 2.0)
        field = np.zeros_like(xx)
        for offset in offsets:
            radius2 = (xx - (cx + offset)) ** 2 + (yy - cy) ** 2
            field += peak_vorticity * np.exp(-radius2 / (2.0 * core_radius**2))
        return self.tensor_to_dofs(field)

    def streamfunction(self, vorticity: FloatArray) -> FloatArray:
        rhs = self.mass @ vorticity
        psi = np.zeros(self.basis.N, dtype=np.float64)
        psi[self.interior] = self._poisson_lu.solve(rhs[self.interior])
        return psi

    def rhs(self, vorticity: FloatArray) -> tuple[FloatArray, float]:
        tri = self.mesh.t
        psi = self.streamfunction(vorticity)
        psi_e = psi[tri]
        omega_e = vorticity[tri]
        dpsi_dx = np.sum(psi_e * self.grad_x, axis=0)
        dpsi_dy = np.sum(psi_e * self.grad_y, axis=0)
        velocity_x = dpsi_dy
        velocity_y = -dpsi_dx
        domega_dx = np.sum(omega_e * self.grad_x, axis=0)
        domega_dy = np.sum(omega_e * self.grad_y, axis=0)
        advective_derivative = velocity_x * domega_dx + velocity_y * domega_dy
        element_load = -self.area * advective_derivative / 3.0
        advection = np.bincount(
            tri.reshape(-1), weights=np.tile(element_load, 3), minlength=self.basis.N
        )
        diffusion = -self.viscosity * (self.stiffness @ vorticity)
        derivative = (advection + diffusion) / self.lumped_mass
        max_speed = float(np.sqrt(velocity_x**2 + velocity_y**2).max(initial=0.0))
        return derivative, max_speed

    def advance_to(
        self,
        vorticity: FloatArray,
        start_time: float,
        end_time: float,
        max_dt: float,
        cfl: float = 0.35,
    ) -> tuple[FloatArray, float]:
        """Advance with explicit midpoint RK2 and CFL-limited substeps."""
        state = np.asarray(vorticity, dtype=np.float64).copy()
        time = float(start_time)
        max_seen_speed = 0.0
        h = self.length / self.cells_per_side
        while time < end_time - 1e-14:
            k1, speed = self.rhs(state)
            max_seen_speed = max(max_seen_speed, speed)
            stable_dt = cfl * h / max(speed, 1e-12)
            dt = min(max_dt, stable_dt, end_time - time)
            midpoint = state + 0.5 * dt * k1
            k2, speed_mid = self.rhs(midpoint)
            max_seen_speed = max(max_seen_speed, speed_mid)
            state += dt * k2
            time += dt
            if not np.all(np.isfinite(state)):
                raise FloatingPointError("non-finite FEM state; reduce max_dt")
        return state, max_seen_speed

    def diagnostics(self, vorticity: FloatArray, time: float, max_speed: float = 0.0) -> FEMDiagnostics:
        psi = self.streamfunction(vorticity)
        kinetic = 0.5 * float(psi @ (self.mass @ vorticity))
        enstrophy = 0.5 * float(vorticity @ (self.mass @ vorticity))
        circulation = float(np.ones(self.basis.N) @ (self.mass @ vorticity))
        return FEMDiagnostics(time, kinetic, enstrophy, circulation, max_speed)
