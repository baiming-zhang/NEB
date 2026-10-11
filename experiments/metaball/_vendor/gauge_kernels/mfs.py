import numpy as np
import cupy as cp
def levels(x):
    q = (x - 0.5) / c.outer_half_extent
    outer = 1 - cp.sum(q ** 8, axis=-1)
    go = -8 / c.outer_half_extent * q ** 7
    delta = x[..., None, :] - geo[:, :3]
    length = cp.linalg.norm(delta, axis=-1)
    distances = length - geo[:, 3]
    low = distances.min(-1)
    w = cp.exp(-(distances - low[..., None]) / c.metaball_softmin_tau)
    inner = low - c.metaball_softmin_tau * cp.log(w.sum(-1))
    w /= w.sum(-1)[..., None]
    gi = cp.sum(w[..., None] * delta / length[..., None], axis=-2)
    return (outer, inner, go, gi)

def directions(n, phase=0.0):
    j = cp.arange(n, dtype='f8')
    z = 1 - 2 * (j + 0.5) / n
    phi = j * (np.pi * (3 - np.sqrt(5))) + phase
    r = cp.sqrt(1 - z * z)
    return cp.stack((r * cp.cos(phi), r * cp.sin(phi), z), axis=1)

def walls(d):
    radii, points, normals = ([], [], [])
    for outer_wall in (False, True):
        low = cp.zeros(len(d))
        high = cp.full(len(d), 1.8)
        for _ in range(48):
            mid = (low + high) * 0.5
            o, i, _, _ = levels(center + mid[:, None] * d)
            below = o > 0 if outer_wall else i < 0
            low = cp.where(below, mid, low)
            high = cp.where(below, high, mid)
        r = 0.5 * (low + high)
        x = center + r[:, None] * d
        _, _, go, gi = levels(x)
        n = -go if outer_wall else -gi
        n /= cp.linalg.norm(n, axis=1)[:, None]
        radii.append(r)
        points.append(x)
        normals.append(n)
    return (radii, points, normals)

def matrix(x, sources):
    return 1 / cp.sqrt(cp.maximum(cp.sum(x * x, 1)[:, None] + cp.sum(sources * sources, 1)[None, :] - 2 * x @ sources.T, 1e-30))