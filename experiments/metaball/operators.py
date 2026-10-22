"""Full-grid, full-mass reduced M/A/Q operators for Metaball coefficient dynamics.

The original rotational generator uses zero-extended basis fields, FFT
Leray/dealias projection, viscosity and normal penalization. All fields,
points and modes enter each contraction. CPU file I/O is bounded.
"""
from pathlib import Path
import argparse, ast, json, math, os, shutil, time
import numpy as np
import psutil
cp = np

def guard():
    vm = psutil.virtual_memory()
    if psutil.Process().memory_info().rss >= 62500000000 or vm.total - vm.available >= 62500000000 or vm.available <= 9 * 2 ** 30:
        raise MemoryError('CPU memory upper bound or host-free reserve')

class Spectral:

    def __init__(self, normal, chi, n=128, viscosity=0.00025, penalty_time=0.001):
        self.dtype = 'f8'
        self.n = n
        k = (2 * np.pi * cp.fft.fftfreq(n, d=1 / n)).astype('f8')
        self.kz = k[:, None, None]
        self.ky = k[None, :, None]
        self.kx = k[None, None, :]
        self.k2 = self.kx ** 2 + self.ky ** 2 + self.kz ** 2
        self.safe = self.k2.copy()
        self.safe[0, 0, 0] = 1
        m = cp.fft.fftfreq(n, d=1 / n)
        valid = cp.abs(m) <= n // 3
        self.dealias = valid[:, None, None] & valid[None, :, None] & valid[None, None, :]
        self.normal = cp.asarray(normal.transpose(3, 0, 1, 2), dtype='f8')
        self.chi = cp.asarray(chi, dtype='f8')
        if normal.shape != (n, n, n, 3) or chi.shape != (n, n, n) or (not np.isfinite(normal).all()) or (not np.isfinite(chi).all()):
            raise ValueError('Complete same-grid normal and solid-fraction arrays required')
        self.nu = viscosity
        self.eta = penalty_time

    def fft(self, u):
        return cp.fft.fftn(u, axes=(1, 2, 3))

    def real(self, v):
        return cp.fft.ifftn(v, axes=(1, 2, 3)).real

    def project(self, v):
        dot = (self.kx * v[0] + self.ky * v[1] + self.kz * v[2]) / self.safe
        r = cp.stack((v[0] - self.kx * dot, v[1] - self.ky * dot, v[2] - self.kz * dot))
        r *= self.dealias
        r[:, 0, 0, 0] = 0
        return r

    def curl(self, v):
        return cp.stack((1j * (self.ky * v[2] - self.kz * v[1]), 1j * (self.kz * v[0] - self.kx * v[2]), 1j * (self.kx * v[1] - self.ky * v[0])))

    def nonlinear_hat(self, v):
        u = self.real(v)
        w = self.real(self.curl(v))
        return self.project(self.fft(cp.cross(u, w, axisa=0, axisb=0, axisc=0)))

    def momentum(self, u, components=False):
        v = self.fft(u)
        n = self.nonlinear_hat(v)
        visc = self.project(-self.nu * self.k2 * v)
        penalty = self.project(self.fft(-self.chi[None] / self.eta * cp.sum(u * self.normal, axis=0)[None] * self.normal))
        if components:
            return dict(nonlinear=self.real(n), viscous=self.real(visc), normal_penalty=self.real(penalty))
        return self.real(n + visc + penalty)

def add(raw, b, p, w, cp, first_output, mode_block=8):
    k, points = b.shape[1:]
    outputs = raw.shape[1]
    p = p[:, first_output:first_output + outputs]
    for j in range(0, k, mode_block):
        end = min(k, j + mode_block)
        target = raw[j:end].reshape(-1, k)
        for a, bb, c, sign in ((0, 1, 2, 1), (0, 2, 1, -1), (1, 2, 0, 1), (1, 0, 2, -1), (2, 0, 1, 1), (2, 1, 0, -1)):
            product = (b[bb, j:end, None, :] * p[a, None, :, :]).reshape(-1, points)
            target += sign * (product @ w[c].T)

class Reader:

    def __init__(self, path, shape=None):
        self.path = Path(path)
        self.file = self.path.open('rb')
        version = np.lib.format.read_magic(self.file)
        self.shape, self.order, self.dtype = np.lib.format._read_array_header(self.file, version)
        self.offset = self.file.tell()
        if self.order or self.dtype.hasobject or self.dtype.kind != 'f' or (self.dtype.itemsize not in (4, 8)) or (shape is not None and tuple(self.shape) != tuple(shape)):
            self.close()
            raise ValueError('Complete C-order FP32/FP64 field NPY required')
        if self.path.stat().st_size != self.offset + math.prod(self.shape) * self.dtype.itemsize:
            self.close()
            raise ValueError('Truncated/appended NPY')

    def rows(self, first, count):
        guard()
        width = math.prod(self.shape[1:])
        self.file.seek(self.offset + first * width * self.dtype.itemsize)
        a = np.fromfile(self.file, dtype=self.dtype, count=count * width).reshape((count, *self.shape[1:])).astype('f8')
        if not np.isfinite(a).all():
            raise FloatingPointError('Nonfinite field')
        return a

    def mode(self, index):
        n, three, K = self.shape
        out = np.empty((n, three), dtype='f8')
        chunk = max(1, 16 * 2 ** 20 // (three * K * self.dtype.itemsize))
        for first in range(0, n, chunk):
            out[first:first + chunk] = self.rows(first, min(chunk, n - first))[:, :, index]
        return out

    def close(self):
        self.file.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

def new_npy(path, shape):
    f = Path(path).open('xb')
    np.lib.format.write_array_header_2_0(f, dict(shape=tuple(shape), fortran_order=False, descr='<f8'))
    offset = f.tell()
    f.truncate(offset + math.prod(shape) * 8)
    f.flush()
    os.fsync(f.fileno())
    return (f, offset)

def seal_json(path, value):
    with Path(path).open('x', encoding='utf-8', newline='\n') as f:
        json.dump(value, f, indent=2, allow_nan=False)
        f.flush()
        os.fsync(f.fileno())

def derive(basis_path, flat_path, normal_path, solid_path, output, grid=128, viscosity=0.00025, penalty_time=0.001):
    start = time.perf_counter()
    out = Path(output)
    out.mkdir(parents=True, exist_ok=False)
    flat = np.load(flat_path, allow_pickle=False)
    if flat.ndim != 1 or flat.dtype.kind not in 'iu' or len(flat) == 0 or np.any(flat < 0) or np.any(flat >= grid ** 3) or np.any(np.diff(flat) <= 0):
        raise ValueError('Ordered unique full fluid flat indices required')
    normal = np.load(normal_path, allow_pickle=False)
    chi = np.load(solid_path, allow_pickle=False)
    spectral = Spectral(normal, chi, grid, viscosity, penalty_time)
    with Reader(basis_path) as r:
        N, three, K = r.shape
        if N != len(flat) or three != 3:
            raise ValueError('Basis/full-fluid point-order mismatch')
        required = 2 * N * 9 * K * 8 + 2 * 2 ** 30
        if shutil.disk_usage(out).free < required:
            raise OSError('Full derived-field transpose plus 2 GiB reserve required')
        f, offset = new_npy(out / 'derived_mode_major.npy', (K, N, 9))
        try:
            for mode in range(K):
                values = r.mode(mode)
                u = np.zeros((grid ** 3, 3), dtype='f8')
                u[flat] = values
                u = np.ascontiguousarray(u.reshape(grid, grid, grid, 3).transpose(3, 0, 1, 2))
                v = spectral.fft(u)
                projected = spectral.real(spectral.project(v))
                curl = spectral.real(spectral.curl(v))
                viscous = spectral.project(-spectral.nu * spectral.k2 * v)
                penalty = spectral.project(spectral.fft(-spectral.chi[None] / spectral.eta * np.sum(u * spectral.normal, axis=0)[None] * spectral.normal))
                linear = spectral.real(viscous + penalty)
                selected = [x.transpose(1, 2, 3, 0).reshape(-1, 3)[flat] for x in (projected, curl, linear)]
                data = np.concatenate(selected, axis=1)
                if not np.isfinite(data).all():
                    raise FloatingPointError('Nonfinite derived field')
                f.seek(offset + mode * N * 9 * 8)
                data.astype('<f8', copy=False).tofile(f)
                f.flush()
                guard()
        finally:
            f.flush()
            os.fsync(f.fileno())
            f.close()
        destination, dst_offset = new_npy(out / 'derived.npy', (N, 9, K))
        try:
            with (out / 'derived_mode_major.npy').open('rb') as source:
                point_block = max(1, 16 * 2 ** 20 // (9 * K * 8))
                for first in range(0, N, point_block):
                    count = min(point_block, N - first)
                    modes = np.empty((K, count, 9))
                    for mode in range(K):
                        source.seek(offset + (mode * N * 9 + first * 9) * 8)
                        modes[mode] = np.fromfile(source, dtype='<f8', count=count * 9).reshape(count, 9)
                    destination.seek(dst_offset + first * 9 * K * 8)
                    np.ascontiguousarray(modes.transpose(1, 2, 0)).tofile(destination)
                    guard()
        finally:
            destination.flush()
            os.fsync(destination.fileno())
            destination.close()
        scratch = (out / 'derived_mode_major.npy').resolve()
        if scratch.parent != out.resolve() or scratch.name != 'derived_mode_major.npy':
            raise ValueError('Owned scratch path mismatch')
        scratch.unlink()
    seal_json(out / 'derived.json', dict(grid=grid, K=K, points=N, order='point/channel/mode; projected_B,curl_B,linear_force', zero_extended_solid=True, precision='float64', viscosity=viscosity, normal_penalty_time=penalty_time, actual_derivation_seconds=time.perf_counter() - start))
    return out / 'derived.npy'

def assemble(basis_path, derived_path, output, grid=128, point_block=1024, slab_bytes=32 * 2 ** 20):
    start = time.perf_counter()
    out = Path(output)
    out.mkdir(parents=True, exist_ok=False)
    with Reader(basis_path) as br:
        N, three, K = br.shape
        if three != 3:
            raise ValueError('Basis shape must be (points,3,K)')
        with Reader(derived_path, (N, 9, K)) as dr:
            dV = 1 / grid ** 3
            M = np.zeros((K, K))
            A = np.zeros((K, K))
            tick = time.perf_counter()
            for first in range(0, N, point_block):
                count = min(point_block, N - first)
                b = br.rows(first, count).reshape(count * 3, K)
                d = dr.rows(first, count)
                M += dV * (b.T @ b)
                A += dV * (b.T @ d[:, 6:9, :].reshape(count * 3, K))
            if np.linalg.eigvalsh((M + M.T) / 2)[0] <= 0:
                raise ArithmeticError('Full mass must be positive definite')
            np.savez(out / 'operators.npz', M=M, A=A)
            linear_seconds = time.perf_counter() - tick
            pairs = K * (K + 1) // 2
            required = K * pairs * 8 + 2 * 2 ** 30
            if shutil.disk_usage(out).free < required:
                raise OSError('Full packed-Q output plus 2 GiB disk reserve required')
            q, offset = new_npy(out / 'quadratic.npy', (K, pairs))
            outputs = max(1, min(K, slab_bytes // (8 * K * K)))
            jj, kk = np.triu_indices(K)
            tick = time.perf_counter()
            try:
                for first_output in range(0, K, outputs):
                    count_output = min(outputs, K - first_output)
                    raw = np.zeros((K, count_output, K), dtype='f8')
                    for first in range(0, N, point_block):
                        count = min(point_block, N - first)
                        b = br.rows(first, count).transpose(1, 2, 0)
                        d = dr.rows(first, count).transpose(1, 2, 0)
                        add(raw, b, d[:3], d[3:6], np, first_output)
                        guard()
                    for i in range(count_output):
                        packed = raw[jj, i, kk].copy()
                        different = jj != kk
                        packed[different] += raw[kk[different], i, jj[different]]
                        packed *= dV
                        q.seek(offset + (first_output + i) * pairs * 8)
                        packed.astype('<f8', copy=False).tofile(q)
                    q.flush()
                    os.fsync(q.fileno())
            finally:
                q.close()
            quadratic_seconds = time.perf_counter() - tick
    seal_json(out / 'assembly.json', dict(grid=grid, points=N, K=K, full_mass=True, all_modes=True, all_points=True, precision='float64', convective_form='PB_i dot (B_j cross curlB_k)', pair_order='numpy.triu_indices(K); sum off-diagonal inputs', linear_form='B transpose times unsplit Leray-projected viscous plus normal-penalty force', linear_assembly_seconds=linear_seconds, quadratic_assembly_seconds=quadratic_seconds, total_including_IO_seconds=time.perf_counter() - start))
    return (out / 'operators.npz', out / 'quadratic.npy')

def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='stage', required=True)
    d = sub.add_parser('derive')
    d.add_argument('--basis', required=True)
    d.add_argument('--flat', required=True)
    d.add_argument('--normal', required=True)
    d.add_argument('--solid', required=True)
    d.add_argument('--output', required=True)
    d.add_argument('--viscosity', type=float, default=0.00025)
    d.add_argument('--penalty-time', type=float, default=0.001)
    a = sub.add_parser('assemble')
    a.add_argument('--basis', required=True)
    a.add_argument('--derived', required=True)
    a.add_argument('--output', required=True)
    a.add_argument('--point-block', type=int, default=1024)
    a.add_argument('--slab-bytes', type=int, default=32 * 2 ** 20)
    args = parser.parse_args(argv)
    if args.stage == 'derive':
        result = derive(args.basis, args.flat, args.normal, args.solid, args.output, viscosity=args.viscosity, penalty_time=args.penalty_time)
    else:
        if args.point_block < 1 or args.slab_bytes < 8:
            parser.error('Positive bounded blocks required')
        result = assemble(args.basis, args.derived, args.output, point_block=args.point_block, slab_bytes=args.slab_bytes)
    print(json.dumps({'output': str(result)}))
if __name__ == '__main__':
    main()
