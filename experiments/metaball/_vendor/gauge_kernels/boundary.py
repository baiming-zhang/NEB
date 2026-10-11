import numpy as np, gc
from pathlib import Path
import cupy as cp
from cupyx.scipy.linalg import solve_triangular
import mfs
RUN=Path("runs/gauge")
def clear():
 gc.collect();cp.get_default_memory_pool().free_all_blocks()
class Boundary:
    """Cached incremental QR, with saved Q blocks for its exact transpose."""

    def __init__(self, sources, xb):
        self.sources = sources
        self.xb = xb
        self.path = RUN / 'boundary_qr'
        self.path.mkdir(exist_ok=True)
        if not (self.path / 'factor.npz').exists():
            scales = cp.zeros(len(sources) + 1)
            for off in range(0, len(xb), 2048):
                A = cp.concatenate((mfs.matrix(xb[off:off + 2048], sources), cp.ones((min(2048, len(xb) - off), 1))), axis=1)
                scales += cp.sum(A * A, axis=0)
            scales = cp.sqrt(scales)
            del A
            R = cp.empty((0, len(scales)))
            sizes = []
            for index, off in enumerate(range(0, len(xb), 2048)):
                A = cp.concatenate((mfs.matrix(xb[off:off + 2048], sources), cp.ones((min(2048, len(xb) - off), 1))), axis=1) / scales
                previous = R.shape[0]
                stacked = cp.concatenate((R, A))
                del A, R
                Q, R = cp.linalg.qr(stacked, mode='reduced')
                del stacked
                np.save(self.path / f'Q{index}.npy', cp.asnumpy(Q))
                sizes.append([off, previous, Q.shape[0] - previous])
                del Q
                clear()
            np.savez(self.path / 'factor.npz', R=cp.asnumpy(R), scales=cp.asnumpy(scales), sizes=sizes)
        with np.load(self.path / 'factor.npz') as z:
            self.R = np.array(z['R'])
            self.scales = cp.asarray(z['scales'])
            self.sizes = z['sizes'].tolist()

    def forward(self, b):
        rhs = cp.empty((0, b.shape[1]))
        for index, (off, previous, count) in enumerate(self.sizes):
            Q = cp.asarray(np.load(self.path / f'Q{index}.npy', mmap_mode='r'))
            rhs = Q.T @ cp.concatenate((rhs, b[off:off + count]))
            del Q
            clear()
        R = cp.asarray(self.R)
        co = solve_triangular(R, rhs) / self.scales[:, None]
        del R
        clear()
        return co

    def transpose(self, cot):
        R = cp.asarray(self.R)
        adj = solve_triangular(R.T, cot / self.scales[:, None], lower=True)
        del R
        clear()
        out = cp.empty((len(self.xb), cot.shape[1]))
        for index in range(len(self.sizes) - 1, -1, -1):
            off, previous, count = self.sizes[index]
            Q = cp.asarray(np.load(self.path / f'Q{index}.npy', mmap_mode='r'))
            g = Q @ adj
            del Q
            out[off:off + count] = g[previous:]
            adj = g[:previous].copy()
            del g
            clear()
        return out
def mfs_evaluate(x, sources, co, grad=True):
    x = cp.asarray(x, dtype='f8')
    vals = []
    grads = []
    for off in range(0, len(x), 512):
        xx = x[off:off + 512]
        delta = xx[:, None, :] - sources[None, :, :]
        inv = 1 / cp.linalg.norm(delta, axis=2)
        vals.append(inv @ co[:-1] + co[-1])
        if grad:
            grads.append(cp.stack([-delta[:, :, j] * inv ** 3 @ co[:-1] for j in range(3)], axis=1))
    out = dict(value=cp.concatenate(vals))
    if grad:
        out['gradient'] = cp.concatenate(grads)
    return out