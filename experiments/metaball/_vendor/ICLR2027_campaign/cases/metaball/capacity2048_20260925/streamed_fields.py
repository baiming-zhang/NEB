





from _neb_paths import Path
import time
import numpy as np


class StreamedFields:
    def __init__(self, path, xp, *, expected_points, expected_modes,
                 block_rows=16384, volume_denominator=128**3):
        self.path = Path(path)
        self.xp = xp
        self.N = int(expected_points)
        self.K = int(expected_modes)
        self.block_rows = int(block_rows)
        self.denominator = float(volume_denominator)
        if min(self.N, self.K, self.block_rows, self.denominator) <= 0:
            raise ValueError('Positive dimensions and integration denominator required')
        with self.path.open('rb') as stream:
            version = np.lib.format.read_magic(stream)
            if version == (1, 0):
                shape, fortran, dtype = np.lib.format.read_array_header_1_0(stream)
            elif version == (2, 0):
                shape, fortran, dtype = np.lib.format.read_array_header_2_0(stream)
            else:
                raise ValueError(f'Unsupported NPY version {version}')
            self.offset = stream.tell()
        if shape != (self.N, 3, self.K) or fortran or dtype != np.dtype('float32'):
            raise ValueError(f'Unexpected field cache layout: {shape}, {fortran}, {dtype}')
        expected_size = self.offset + self.N * 3 * self.K * 4
        if self.path.stat().st_size != expected_size:
            raise ValueError('Incomplete field cache; require completed extraction receipt too')
        self.timings = {}

    def _sync(self):
        if hasattr(self.xp, 'cuda'):
            self.xp.cuda.get_current_stream().synchronize()

    def _start(self):
        self._sync()
        return time.perf_counter()

    def _finish(self, name, tick):
        self._sync()
        row = self.timings.setdefault(name, dict(calls=0, seconds=0.0))
        row['calls'] += 1
        row['seconds'] += time.perf_counter() - tick

    def blocks(self):
        with self.path.open('rb') as stream:
            stream.seek(self.offset)
            for off in range(0, 3 * self.N, self.block_rows):
                count = min(self.block_rows, 3 * self.N - off)
                host = np.fromfile(stream, dtype='f4', count=count * self.K)
                if host.size != count * self.K:
                    raise IOError('Field cache truncated during evaluation')
                block = self.xp.asarray(host.reshape(count, self.K), dtype='f8')
                yield off, block
                del block, host

    def mass(self):
        tick = self._start()
        out = self.xp.zeros((self.K, self.K), dtype='f8')
        for _, block in self.blocks():
            out += block.T @ block
        out /= self.denominator
        self._finish('mass_assembly', tick)
        return out

    def fluid(self, coefficients):
        if coefficients.shape != (self.K,):
            raise ValueError('Expected one coefficient per mode')
        tick = self._start()
        out = self.xp.empty(3 * self.N, dtype='f8')
        for off, block in self.blocks():
            out[off:off + len(block)] = block @ coefficients
        self._finish('forward_all_nodes', tick)
        return out.reshape(self.N, 3)

    def transpose_fluid(self, field):
        if field.shape != (self.N, 3):
            raise ValueError('Expected xyz velocity at every fluid node')
        tick = self._start()
        out = self.xp.zeros(self.K, dtype='f8')
        flat = field.reshape(-1)
        for off, block in self.blocks():
            out += block.T @ flat[off:off + len(block)]
        out /= self.denominator
        self._finish('transpose_all_nodes', tick)
        return out
