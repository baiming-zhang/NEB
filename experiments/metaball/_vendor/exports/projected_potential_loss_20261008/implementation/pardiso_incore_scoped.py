






from pardiso_scoped import ScopedPardisoSPD


class InCoreScopedPardisoSPD(ScopedPardisoSPD):
    MAX_SYMBOLIC_FACTOR_BYTES = 4 * 2**30

    def __init__(self, matrix, directory, log=None):
        super().__init__(matrix, directory, log)
        self.analysis.update(factor_storage='in-core', iparm59=0,
                             parallel_forward_backward_iparm24=0,
                             double_precision_iparm27=0,
                             in_core_symbolic_peak_bytes=self.in_core_symbolic_peak_bytes,
                             symbolic_memory_gate_bytes=self.MAX_SYMBOLIC_FACTOR_BYTES,
                             ooc_environment_ignored_in_incore_mode=True)
        self.log('in_core_factor_configuration', **self.analysis)

    def _call(self, phase_number, rhs=None):
        if phase_number == 11:


            self.iparm[59] = 0
            self.iparm[24] = 0
            self.iparm[27] = 0
        result = super()._call(phase_number, rhs)
        if phase_number == 11:
            self.in_core_symbolic_peak_bytes = 1024 * max(
                int(self.iparm[14]), int(self.iparm[15]) + int(self.iparm[16]))
            if self.in_core_symbolic_peak_bytes >= self.MAX_SYMBOLIC_FACTOR_BYTES:
                raise MemoryError('Actual symbolic in-core PARDISO memory exceeds 4 GiB gate')
        if self.iparm[59] != 0 or self.iparm[27] != 0:
            raise RuntimeError('In-core FP64 factor configuration changed')
        return result
