




import json
from _neb_paths import Path
import time
import numpy as np


class BlockedCoefficients:
    def __init__(self, folder, xp, *, expected_rows, expected_modes):
        self.folder, self.xp = Path(folder), xp
        self.rows, self.modes = expected_rows, expected_modes
        self.entries = json.loads((self.folder / 'locked.json').read_text())
        self.timings = {}
        eigenvalues = []
        count = 0
        for entry in self.entries:
            path = self.folder / entry['file']

            with path.open('rb') as stream:
                version = np.lib.format.read_magic(stream)
                if version == (1, 0):
                    shape, _, dtype = np.lib.format.read_array_header_1_0(stream)
                elif version == (2, 0):
                    shape, _, dtype = np.lib.format.read_array_header_2_0(stream)
                else:
                    raise ValueError('Unsupported coefficient block format')
                payload = stream.tell()
            if shape != (self.rows, entry['columns']) or dtype != np.dtype('f8'):
                raise ValueError('Coefficient block shape or precision mismatch')
            if path.stat().st_size != payload + int(np.prod(shape)) * dtype.itemsize:
                raise ValueError('Coefficient block is incomplete')
            lam = np.asarray(entry['eigenvalues'], dtype='f8')
            if lam.shape != (entry['columns'],) or not np.isfinite(lam).all() or (lam <= 0).any():
                raise ValueError('Invalid positive eigenvalues')
            eigenvalues.extend(lam)
            count += entry['columns']
        if count != self.modes:
            raise ValueError('Require the complete requested numerical spectrum')
        self.scale = xp.asarray(np.sqrt(eigenvalues), dtype='f8')

    def _sync(self):
        if hasattr(self.xp, 'cuda'):
            self.xp.cuda.Stream.null.synchronize()

    def _record(self, name, start):
        self._sync()
        row = self.timings.setdefault(name, dict(calls=0, seconds=0.0))
        row['calls'] += 1
        row['seconds'] += time.perf_counter() - start

    def blocks(self):
        first = 0
        for entry in self.entries:


            host = np.load(self.folder / entry['file'], allow_pickle=False)
            block = self.xp.asarray(host, dtype='f8')
            stop = first + entry['columns']
            yield first, stop, block
            del host, block
            first = stop

    def forward(self, coefficients):

        xp = self.xp
        c = xp.asarray(coefficients, dtype='f8')
        if c.ndim not in (1, 2) or c.shape[0] != self.modes:
            raise ValueError('Coefficient dimension mismatch')
        self._sync()
        tick = time.perf_counter()
        out = xp.zeros((self.rows,) + c.shape[1:], dtype='f8')
        for first, stop, block in self.blocks():
            scale = self.scale[first:stop]
            if c.ndim == 2:
                scale = scale[:, None]
            out += block @ (c[first:stop] / scale)
            del block
        self._record('forward', tick)
        return out

    def transpose(self, edge_values):

        xp = self.xp
        x = xp.asarray(edge_values, dtype='f8')
        if x.ndim not in (1, 2) or x.shape[0] != self.rows:
            raise ValueError('Edge vector dimension mismatch')
        self._sync()
        tick = time.perf_counter()
        out = xp.empty((self.modes,) + x.shape[1:], dtype='f8')
        for first, stop, block in self.blocks():
            scale = self.scale[first:stop]
            if x.ndim == 2:
                scale = scale[:, None]
            out[first:stop] = (block.T @ x) / scale
            del block
        self._record('transpose', tick)
        return out
