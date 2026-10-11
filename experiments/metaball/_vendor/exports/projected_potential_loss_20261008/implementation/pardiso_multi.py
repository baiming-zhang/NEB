




import ctypes as ct
import os
from _neb_paths import Path

import numpy as np
import scipy.sparse as sp

RUNTIME = Path(os.environ.get('NEB_MKL_RUNTIME', '_data/dependencies/mkl_runtime'))
_DLL_HANDLES = []


class PardisoSPD:
    def __init__(self, matrix, directory, log=None):
        self.log = log or (lambda *a, **k: None)
        self.closed = False
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        os.environ.update(MKL_INTERFACE_LAYER='LP64', MKL_NUM_THREADS='2',
                          MKL_DYNAMIC='FALSE', MKL_PARDISO_OOC_MAX_CORE_SIZE='3072',
                          MKL_PARDISO_OOC_MAX_SWAP_SIZE='0',
                          MKL_PARDISO_OOC_PATH=str(self.directory),
                          MKL_PARDISO_OOC_FILE_NAME=f'canonical_lift_{os.getpid()}')
        dlls = sorted(RUNTIME.rglob('mkl_rt*.dll'))+sorted(RUNTIME.rglob('libmkl_rt.so*'))
        if not dlls:
            raise RuntimeError('Existing isolated oneMKL runtime not found')
        if hasattr(os,'add_dll_directory'):
            for folder in sorted({p.parent for p in RUNTIME.rglob('*.dll')}):
                _DLL_HANDLES.append(os.add_dll_directory(str(folder)))
        self.lib = ct.CDLL(str(dlls[0]))
        self.lib.pardiso.argtypes = [ct.c_void_p] * 16
        self.lib.pardiso.restype = None
        self.lib.pardisoinit.argtypes = [ct.c_void_p] * 3
        self.lib.pardisoinit.restype = None
        self.pt = (ct.c_void_p * 64)()
        self.iparm = np.zeros(64, dtype='i4')
        self.mtype = ct.c_int(2)
        self.lib.pardisoinit(self.pt, ct.byref(self.mtype), self.ptr(self.iparm))
        self.iparm[0] = 1
        self.iparm[1] = 2
        self.iparm[7] = 2
        self.iparm[34] = 1
        self.iparm[59] = 2
        upper = sp.triu(matrix, format='csr')
        upper.sum_duplicates()
        upper.sort_indices()
        if upper.shape[0] > np.iinfo(np.int32).max or upper.nnz > np.iinfo(np.int32).max:
            raise ValueError('Matrix exceeds the declared LP64 API')
        self.a = np.ascontiguousarray(upper.data, dtype='f8')
        self.ia = np.ascontiguousarray(upper.indptr, dtype='i4')
        self.ja = np.ascontiguousarray(upper.indices, dtype='i4')
        self.n = ct.c_int(upper.shape[0])
        self.perm = np.zeros(upper.shape[0], dtype='i4')
        self.maxfct = ct.c_int(1)
        self.mnum = ct.c_int(1)
        self._call(11)
        self.analysis = dict(symbolic_peak_kb=int(self.iparm[14]),
                             permanent_kb=int(self.iparm[15]),
                             in_core_numeric_kb=int(self.iparm[16]),
                             ooc_numeric_kb=int(self.iparm[62]),
                             factor_nonzeros=int(self.iparm[17]),
                             ooc_max_core_MB=3072, matrix_dof=self.n.value,
                             RHS_layout='Fortran column-major', dtype='float64')
        self.log('pardiso_symbolic_complete', **self.analysis)
        self._call(22)
        self.log('pardiso_numeric_complete', **self.analysis)

    @staticmethod
    def ptr(array):
        return ct.c_void_p(array.ctypes.data)

    def _call(self, phase_number, rhs=None):
        phase = ct.c_int(phase_number)
        vector = rhs is None or np.ndim(rhs) == 1
        if rhs is None:
            b = np.zeros((self.n.value, 1), dtype='f8', order='F')
        else:
            b = np.asarray(rhs, dtype='f8')
            if b.ndim == 1:
                b = b[:, None]
            if b.ndim != 2 or b.shape[0] != self.n.value:
                raise ValueError('Incomplete or invalid full-domain RHS')
            b = np.asfortranarray(b)
        nrhs = ct.c_int(b.shape[1])
        x = np.empty_like(b, order='F')
        msglvl = ct.c_int(0)
        error = ct.c_int(0)
        self.lib.pardiso(self.pt, ct.byref(self.maxfct), ct.byref(self.mnum),
                         ct.byref(self.mtype), ct.byref(phase), ct.byref(self.n),
                         self.ptr(self.a), self.ptr(self.ia), self.ptr(self.ja),
                         self.ptr(self.perm), ct.byref(nrhs), self.ptr(self.iparm),
                         ct.byref(msglvl), self.ptr(b), self.ptr(x), ct.byref(error))
        if error.value:
            raise RuntimeError(f'PARDISO phase {phase_number} error {error.value}')
        return x[:, 0] if vector else x

    def solve(self, rhs):
        if self.closed:
            raise RuntimeError('Closed preconditioner')
        return self._call(33, rhs)

    def close(self):
        if not self.closed:
            self._call(-1)
            self.closed = True
