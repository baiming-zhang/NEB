
from contextlib import contextmanager
import ctypes as ct
import os

from pardiso_multi import PardisoSPD


class ScopedPardisoSPD(PardisoSPD):
    def _call(self, phase_number, rhs=None):


        os.environ['MKL_PARDISO_OOC_PATH'] = str(self.directory)
        os.environ['MKL_PARDISO_OOC_FILE_NAME'] = f'canonical_lift_{os.getpid()}'
        return super()._call(phase_number, rhs)

    @contextmanager
    def local_threads(self, threads):
        if threads not in (2, 4, 8):
            raise ValueError('Only declared finite CPU thread candidates allowed')
        setter = None
        for name in ('MKL_Set_Num_Threads_Local', 'mkl_set_num_threads_local'):
            try:
                setter = getattr(self.lib, name)
                break
            except AttributeError:
                pass
        if setter is None:
            raise RuntimeError('Existing oneMKL lacks its local thread scope API')
        setter.argtypes = [ct.c_int]
        setter.restype = ct.c_int
        previous = setter(ct.c_int(threads))
        try:
            yield
        finally:
            setter(ct.c_int(previous))
