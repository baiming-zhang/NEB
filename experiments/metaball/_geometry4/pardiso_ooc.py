import os,ctypes as ct
from pathlib import Path
import numpy as np,scipy.sparse as sp
_DLL_HANDLES=[]
class PardisoOOC:

    def __init__(self, matrix, log=None):
        self.log = log or (lambda *a, **kw: None)
        self.closed = False
        dlls = [Path(os.environ['NEB_MKL_LIBRARY'])]
        if not dlls:
            raise RuntimeError('oneMKL runtime DLL not installed in isolated dependency directory')
        if os.name == 'nt':
            _DLL_HANDLES.append(os.add_dll_directory(str(dlls[0].parent)))
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
        self.a = np.ascontiguousarray(upper.data, dtype='f8')
        self.ia = np.ascontiguousarray(upper.indptr, dtype='i4')
        self.ja = np.ascontiguousarray(upper.indices, dtype='i4')
        self.n = ct.c_int(upper.shape[0])
        self.perm = np.zeros(upper.shape[0], dtype='i4')
        self.maxfct = ct.c_int(1)
        self.mnum = ct.c_int(1)
        self._call(11)
        self.analysis = dict(symbolic_peak_kb=int(self.iparm[14]), permanent_kb=int(self.iparm[15]), in_core_numeric_kb=int(self.iparm[16]), ooc_numeric_kb=int(self.iparm[62]), factor_nonzeros=int(self.iparm[17]), ooc_max_core_MB=3072)
        self.log('pardiso_symbolic_complete', **self.analysis)
        self._call(22)
        self.log('pardiso_factorization_complete', **self.analysis)

    @staticmethod
    def ptr(a):
        return ct.c_void_p(a.ctypes.data)

    def _call(self, phase, b=None):
        phase = ct.c_int(phase)
        nrhs = ct.c_int(1)
        msglvl = ct.c_int(0)
        error = ct.c_int(0)
        b = np.zeros(self.n.value, dtype='f8') if b is None else np.ascontiguousarray(b, dtype='f8')
        x = np.empty_like(b)
        self.lib.pardiso(self.pt, ct.byref(self.maxfct), ct.byref(self.mnum), ct.byref(self.mtype), ct.byref(phase), ct.byref(self.n), self.ptr(self.a), self.ptr(self.ia), self.ptr(self.ja), self.ptr(self.perm), ct.byref(nrhs), self.ptr(self.iparm), ct.byref(msglvl), self.ptr(b), self.ptr(x), ct.byref(error))
        if error.value:
            raise RuntimeError(f'PARDISO phase{phase.value} error{error.value}')
        return x

    def solve(self, b):
        if self.closed:
            raise RuntimeError('PARDISO factor closed')
        return self._call(33, b)

    def close(self):
        if not self.closed:
            self._call(-1)
            self.closed = True
