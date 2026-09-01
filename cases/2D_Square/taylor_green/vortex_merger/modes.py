"""Closed-form Laplacian eigenfunctions on a two-dimensional square."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

FloatArray = NDArray[np.float64]


def _lowest_wave_numbers(count: int) -> tuple[NDArray[np.int32], NDArray[np.int32]]:
    """Return the first ``count`` positive wave-number pairs by eigenvalue."""
    if count < 1:
        raise ValueError("count must be positive")

    bound = max(2, int(np.ceil(np.sqrt(count))))
    while True:
        pairs = [(i * i + j * j, i, j) for i in range(1, bound + 1) for j in range(1, bound + 1)]
        pairs.sort(key=lambda item: (item[0], item[1], item[2]))
        if len(pairs) >= count and pairs[count - 1][0] < (bound + 1) ** 2 + 1:
            selected = pairs[:count]
            return (
                np.asarray([p[1] for p in selected], dtype=np.int32),
                np.asarray([p[2] for p in selected], dtype=np.int32),
            )
        bound *= 2


@dataclass(frozen=True)
class SquareLaplacianBasis:
    """Analytic free-slip Laplacian basis on ``[0, length]^2``.

    The paper uses ``length = pi``.  With ``alpha_i = k_i*pi/length``, the
    vorticity and velocity modes are

    ``phi_k = sin(alpha_1*x) sin(alpha_2*y)`` and

    ``Phi_k = (alpha_2 sin(alpha_1*x) cos(alpha_2*y),
               -alpha_1 cos(alpha_1*x) sin(alpha_2*y)) / |alpha|^2``.
    """

    k1: NDArray[np.int32]
    k2: NDArray[np.int32]
    length: float = np.pi

    @classmethod
    def lowest(cls, count: int, length: float = np.pi) -> SquareLaplacianBasis:
        k1, k2 = _lowest_wave_numbers(count)
        return cls(k1=k1, k2=k2, length=float(length))

    @property
    def count(self) -> int:
        return int(self.k1.size)

    @property
    def alpha1(self) -> FloatArray:
        return self.k1.astype(np.float64) * np.pi / self.length

    @property
    def alpha2(self) -> FloatArray:
        return self.k2.astype(np.float64) * np.pi / self.length

    @property
    def eigenvalues(self) -> FloatArray:
        return -(self.alpha1**2 + self.alpha2**2)

    @property
    def wave_number_squared(self) -> FloatArray:
        return self.k1.astype(np.float64) ** 2 + self.k2.astype(np.float64) ** 2

    def vorticity_mode(self, index: int, x: ArrayLike, y: ArrayLike) -> FloatArray:
        x_arr, y_arr = np.broadcast_arrays(np.asarray(x, dtype=float), np.asarray(y, dtype=float))
        return np.sin(self.alpha1[index] * x_arr) * np.sin(self.alpha2[index] * y_arr)

    def velocity_mode(self, index: int, x: ArrayLike, y: ArrayLike) -> tuple[FloatArray, FloatArray]:
        x_arr, y_arr = np.broadcast_arrays(np.asarray(x, dtype=float), np.asarray(y, dtype=float))
        a1 = self.alpha1[index]
        a2 = self.alpha2[index]
        q2 = a1 * a1 + a2 * a2
        ux = a2 * np.sin(a1 * x_arr) * np.cos(a2 * y_arr) / q2
        uy = -a1 * np.cos(a1 * x_arr) * np.sin(a2 * y_arr) / q2
        return ux, uy

    def project_vorticity(self, field: ArrayLike, x: ArrayLike, y: ArrayLike, block: int = 128) -> FloatArray:
        """Project a nodal field on an inclusive uniform tensor grid."""
        values = np.asarray(field, dtype=np.float64)
        x_arr = np.asarray(x, dtype=np.float64)
        y_arr = np.asarray(y, dtype=np.float64)
        if values.shape != (x_arr.size, y_arr.size):
            raise ValueError("field must have shape (len(x), len(y))")
        if x_arr.size < 3 or y_arr.size < 3:
            raise ValueError("projection grid is too small")
        dx = np.diff(x_arr)
        dy = np.diff(y_arr)
        if not np.allclose(dx, dx[0]) or not np.allclose(dy, dy[0]):
            raise ValueError("projection requires a uniform tensor grid")

        wx = np.ones_like(x_arr)
        wy = np.ones_like(y_arr)
        wx[[0, -1]] = 0.5
        wy[[0, -1]] = 0.5
        weighted = values * wx[:, None] * wy[None, :]
        coeff = np.empty(self.count, dtype=np.float64)
        norm = 4.0 * dx[0] * dy[0] / (self.length * self.length)
        for start in range(0, self.count, block):
            stop = min(start + block, self.count)
            sx = np.sin(x_arr[:, None] * self.alpha1[start:stop])
            sy = np.sin(y_arr[:, None] * self.alpha2[start:stop])
            coeff[start:stop] = norm * np.einsum("ik,ij,jk->k", sx, weighted, sy, optimize=True)
        return coeff

    def reconstruct_vorticity(self, coefficients: ArrayLike, x: ArrayLike, y: ArrayLike, block: int = 128) -> FloatArray:
        coeff = np.asarray(coefficients, dtype=np.float64)
        if coeff.shape != (self.count,):
            raise ValueError(f"coefficients must have shape ({self.count},)")
        x_arr = np.asarray(x, dtype=np.float64)
        y_arr = np.asarray(y, dtype=np.float64)
        result = np.zeros((x_arr.size, y_arr.size), dtype=np.float64)
        for start in range(0, self.count, block):
            stop = min(start + block, self.count)
            sx = np.sin(x_arr[:, None] * self.alpha1[start:stop])
            sy = np.sin(y_arr[:, None] * self.alpha2[start:stop])
            result += np.einsum("ik,k,jk->ij", sx, coeff[start:stop], sy, optimize=True)
        return result

    def reconstruct_velocity(self, coefficients: ArrayLike, x: ArrayLike, y: ArrayLike, block: int = 128) -> tuple[FloatArray, FloatArray]:
        coeff = np.asarray(coefficients, dtype=np.float64)
        if coeff.shape != (self.count,):
            raise ValueError(f"coefficients must have shape ({self.count},)")
        x_arr = np.asarray(x, dtype=np.float64)
        y_arr = np.asarray(y, dtype=np.float64)
        ux = np.zeros((x_arr.size, y_arr.size), dtype=np.float64)
        uy = np.zeros_like(ux)
        for start in range(0, self.count, block):
            stop = min(start + block, self.count)
            a1 = self.alpha1[start:stop]
            a2 = self.alpha2[start:stop]
            q2 = a1 * a1 + a2 * a2
            sx, cx = np.sin(x_arr[:, None] * a1), np.cos(x_arr[:, None] * a1)
            sy, cy = np.sin(y_arr[:, None] * a2), np.cos(y_arr[:, None] * a2)
            ux += np.einsum("ik,k,jk->ij", sx, coeff[start:stop] * a2 / q2, cy, optimize=True)
            uy -= np.einsum("ik,k,jk->ij", cx, coeff[start:stop] * a1 / q2, sy, optimize=True)
        return ux, uy

    def kinetic_energy(self, coefficients: ArrayLike) -> float:
        """Physical kinetic energy of Eq. (2), including its non-unit norm."""
        coeff = np.asarray(coefficients, dtype=np.float64)
        if coeff.shape != (self.count,):
            raise ValueError(f"coefficients must have shape ({self.count},)")
        return float(self.length**2 / 8.0 * np.sum(coeff * coeff / (-self.eigenvalues)))
