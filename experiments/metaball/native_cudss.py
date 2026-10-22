from pathlib import Path
from contextlib import contextmanager
import ctypes as C, json, os, threading, time
import numpy as np
import psutil

def need(condition, message):
    if not condition:
        raise ValueError(message)

def preallocate(cp, callback, phase, GPU_bytes=0, CPU_bytes=0, **details):
    row = callback()
    free, total = cp.cuda.runtime.memGetInfo()
    v = psutil.virtual_memory()
    pool = cp.get_default_memory_pool()
    outside = max(0, int(total - free) - pool.total_bytes())
    need(psutil.Process().memory_info().rss + CPU_bytes < 62500000000 and v.used + CPU_bytes < 62500000000 and (v.available - CPU_bytes > 9 * 2 ** 30), 'Host allocation reserve: ' + phase)
    need(int(total - free) + GPU_bytes < 15500000000 and free - GPU_bytes > 500000000, 'Aggregate GPU allocation reserve: ' + phase)
    pool.set_limit(size=max(1, min(12 * 2 ** 30, 15500000000 - outside)))
    need(pool.total_bytes() + GPU_bytes <= pool.get_limit(), 'Bounded pool allocation: ' + phase)
    return row
DEFAULT_DLL = Path(os.environ.get('NEB_CUDSS_LIBRARY', 'cudss64_0.dll'))
ALLOC = C.CFUNCTYPE(C.c_int, C.c_void_p, C.POINTER(C.c_void_p), C.c_size_t, C.c_void_p)
FREE = C.CFUNCTYPE(C.c_int, C.c_void_p, C.c_void_p, C.c_size_t, C.c_void_p)

class DeviceCotreeSuperLU:

    def upload_piece(self, source, label, check_bytes=True):
        cp = self.cp
        source = np.asarray(source)
        if not source.flags.c_contiguous:
            raise ValueError('Exact exported payload is not contiguous: ' + label)
        before = dict(rss=psutil.Process().memory_info().rss, default_pinned_free_blocks=int(cp.get_default_pinned_memory_pool().n_free_blocks()))
        cp.cuda.get_current_stream().synchronize()
        t = time.perf_counter()
        result = cp.empty(source.shape, dtype=source.dtype)
        result.data.copy_from_host(int(source.ctypes.data), int(source.nbytes))
        cp.cuda.Stream.null.synchronize()
        cp.cuda.get_current_stream().synchronize()
        copy_seconds = time.perf_counter() - t
        chunks = max(1, 4 * 2 ** 20 // source.dtype.itemsize)
        flat = source.reshape(-1)
        t = time.perf_counter()
        if check_bytes:
            for offset in range(0, len(flat), chunks):
                count = min(chunks, len(flat) - offset)
                returned = np.empty(count, dtype=source.dtype)
                result.reshape(-1)[offset:offset + count].data.copy_to_host(int(returned.ctypes.data), int(returned.nbytes))
                cp.cuda.Stream.null.synchronize()
                if not np.array_equal(returned.view(np.uint8), flat[offset:offset + count].view(np.uint8)):
                    raise AssertionError('Exact device factor roundtrip bytes changed: ' + label)
        after = dict(rss=psutil.Process().memory_info().rss, default_pinned_free_blocks=int(cp.get_default_pinned_memory_pool().n_free_blocks()))
        self.resource_callback()
        return (result, dict(label=label, shape=list(source.shape), dtype=str(source.dtype), bytes=int(source.nbytes), actual_copy_seconds=copy_seconds, actual_complete_roundtrip_seconds=time.perf_counter() - t, bounded_roundtrip_host_buffer_bytes=min(source.nbytes, 4 * 2 ** 20), before=before, after=after, synchronous_documented_memory_pointer_copy=True, original_allocator_not_changed=True))

class MemoryHandler(C.Structure):
    _fields_ = [('ctx', C.c_void_p), ('device_alloc', ALLOC), ('device_free', FREE), ('name', C.c_char * 64)]

class NativeAPI:
    SPECS = {'cudssCreate': [C.POINTER(C.c_void_p)], 'cudssDestroy': [C.c_void_p], 'cudssConfigCreate': [C.POINTER(C.c_void_p)], 'cudssConfigDestroy': [C.c_void_p], 'cudssDataCreate': [C.c_void_p, C.POINTER(C.c_void_p)], 'cudssDataDestroy': [C.c_void_p, C.c_void_p], 'cudssSetStream': [C.c_void_p, C.c_void_p], 'cudssSetDeviceMemHandler': [C.c_void_p, C.POINTER(MemoryHandler)], 'cudssConfigSet': [C.c_void_p, C.c_int, C.c_void_p, C.c_size_t], 'cudssConfigGet': [C.c_void_p, C.c_int, C.c_void_p, C.c_size_t, C.POINTER(C.c_size_t)], 'cudssDataGet': [C.c_void_p, C.c_void_p, C.c_int, C.c_void_p, C.c_size_t, C.POINTER(C.c_size_t)], 'cudssMatrixCreateCsr': [C.POINTER(C.c_void_p), C.c_int64, C.c_int64, C.c_int64, C.c_void_p, C.c_void_p, C.c_void_p, C.c_void_p, C.c_int, C.c_int, C.c_int, C.c_int, C.c_int, C.c_int], 'cudssMatrixCreateDn': [C.POINTER(C.c_void_p), C.c_int64, C.c_int64, C.c_int64, C.c_void_p, C.c_int, C.c_int], 'cudssMatrixDestroy': [C.c_void_p], 'cudssExecute': [C.c_void_p, C.c_int, C.c_void_p, C.c_void_p, C.c_void_p, C.c_void_p, C.c_void_p], 'cudssGetProperty': [C.c_int, C.POINTER(C.c_int)]}

    def __init__(self, path=DEFAULT_DLL):
        path = Path(path).resolve()
        self.directory_handles = []
        if hasattr(os, 'add_dll_directory'):
            self.directory_handles.append(os.add_dll_directory(str(path.parent)))
        self.library = (C.WinDLL if os.name == 'nt' else C.CDLL)(str(path))
        for name, args in self.SPECS.items():
            fn = getattr(self.library, name)
            fn.argtypes = args
            fn.restype = C.c_int
        self.version = []
        for property_index in (0, 1, 2):
            v = C.c_int()
            self.call('cudssGetProperty', property_index, C.byref(v))
            self.version.append(v.value)
        need(self.version == [0, 8, 0], 'cuDSS0.8.0 API required')

    def call(self, name, *args):
        status = getattr(self.library, name)(*args)
        if status:
            raise RuntimeError('Native cuDSS API failed ' + name + ' status=' + str(status))
        return status

class PoolAllocator:

    def __init__(self, cp, guard, device, stream):
        self.cp, self.guard, self.device, self.stream = (cp, guard, int(device), int(stream))
        self.pool = cp.get_default_memory_pool()
        self.lock = threading.RLock()
        self.live = {}
        self.records = []
        self.pending_error = None
        self.context = id(self)
        self.alloc_callback = ALLOC(self.allocate)
        self.free_callback = FREE(self.free)
        self.handler = MemoryHandler(self.context, self.alloc_callback, self.free_callback, b'Eigenbasis_same_bounded_CuPy_pool')

    def _scope(self, ctx, stream):
        need(int(ctx or 0) == self.context and int(stream or 0) == self.stream, 'Native allocator context and fixed owning stream must be identical.')
        return (self.cp.cuda.Device(self.device), self.cp.cuda.ExternalStream(self.stream, device_id=self.device))

    def _error(self, error):
        if self.pending_error is None:
            self.pending_error = error

    def raise_pending(self):
        with self.lock:
            error = self.pending_error
            self.pending_error = None
        if error is not None:
            raise error

    def allocate(self, ctx, ptr, size, stream):
        address = None
        memory = None
        try:
            with self.lock:
                d, s = self._scope(ctx, stream)
                with d, s:
                    if not size:
                        ptr[0] = None
                        return 0
                    preallocate(self.cp, self.guard, 'native_cudss_allocation', GPU_bytes=int(size) + 512, native_requested_bytes=int(size), native_context=self.context)
                    memory = self.pool.malloc(int(size))
                    address = int(memory.ptr)
                    need(address % 256 == 0 and address not in self.live, 'Native memory must be owned and256-byte aligned.')
                    self.live[address] = (memory, int(size), int(stream or 0))
                    ptr[0] = address
                    self.records.append(dict(operation='alloc', pointer=address, size=int(size), stream=int(stream or 0), context=self.context, live_requested_bytes=sum((v[1] for v in self.live.values())), actual_aggregate_pool_bytes=int(self.pool.total_bytes())))
                    self.guard()
                    return 0
        except BaseException as error:
            with self.lock:
                self._error(error)
                if memory is not None:
                    try:
                        with self.cp.cuda.Device(self.device), self.cp.cuda.ExternalStream(self.stream, device_id=self.device):
                            self.cp.cuda.get_current_stream().synchronize()
                            memory.mem.free()
                        self.live.pop(address, None)
                    except BaseException as rollback_error:
                        self.records.append(dict(operation='failed_allocation_rollback', pointer=address, error=repr(rollback_error), ownership_retained_for_cleanup=True))
            ptr[0] = None
            return 1

    def free(self, ctx, ptr, size, stream):
        try:
            with self.lock:
                d, s = self._scope(ctx, stream)
                with d, s:
                    address = int(ptr or 0)
                    if not address and (not size):
                        return 0
                    need(address in self.live, 'Native free must refer to this owned pool allocation.')
                    memory, original_size, original_stream = self.live[address]
                    need(int(size) == original_size and int(stream or 0) == original_stream, 'Native free size/context/stream must match allocation.')
                    self.cp.cuda.get_current_stream().synchronize()
                    memory.mem.free()
                    del self.live[address]
                    self.records.append(dict(operation='free', pointer=address, size=int(size), stream=int(stream or 0), context=self.context, live_requested_bytes=sum((v[1] for v in self.live.values())), actual_aggregate_pool_bytes=int(self.pool.total_bytes())))
                    return 0
        except BaseException as error:
            with self.lock:
                self._error(error)
            return 1

class NativeCuDSSPair:
    upload_piece = DeviceCotreeSuperLU.upload_piece

    def __init__(self, cp, target_N, target_T, host_N, host_T, resource_callback, *, api=None, dll=DEFAULT_DLL, log=None, future_shared_GPU_reserve_bytes=2 * 2 ** 30):
        self.cp, self.resource_callback, self.log = (cp, resource_callback, log or (lambda *a, **kw: None))
        self.device = int(cp.cuda.runtime.getDevice())
        self.stream = int(cp.cuda.get_current_stream().ptr)
        self.owner_thread = threading.get_ident()
        self.shape = host_N.shape
        self.closed = False
        self.states = {}
        self.calls = []
        self.upload_records = []
        self.factorization_records = []
        self.phase_records = []
        self.analysis = {}
        need(self.shape == (210062, 210062) or getattr(api, 'CPU_fixture', False), 'The full original cotree matrix is required.')
        matrix_byte_checks = []
        for key, host, device in (('N', host_N, target_N), ('T', host_T, target_T)):
            need(host.format == 'csr' and host.dtype == np.dtype('f8') and (host.shape == self.shape) and (host.indices.dtype == host.indptr.dtype == np.dtype('i4')) and host.has_canonical_format and np.isfinite(host.data).all(), 'Exact original full FP64/LP64 CSR values are required.')
            need(device.dtype == host.dtype and device.shape == host.shape, 'The exact device matrix shape/dtype must remain.')
            for name in ('data', 'indices', 'indptr'):
                original = getattr(host, name)
                actual = cp.asnumpy(getattr(device, name))
                need(actual.dtype == original.dtype and actual.shape == original.shape and (None == None), 'Every full native target CSR byte must match its original host payload.')
                matrix_byte_checks.append(dict(orientation=key, array=name, actual_full_byte_identity_checked=True, bytes=int(actual.nbytes)))
                del actual
        transpose = host_N.T.tocsr()
        need(all((np.array_equal(getattr(host_T, k).view('u1'), getattr(transpose, k).view('u1')) for k in ('data', 'indices', 'indptr'))), 'The second factor must contain the actual full stored CSR transpose.')
        self.targets = {'N': target_N, 'T': target_T}
        self.host_norms = {k: float(np.asarray(abs(v).sum(axis=1)).max()) for k, v in (('N', host_N), ('T', host_T))}
        need(all((np.isfinite(v) and v > 0 for v in self.host_norms.values())), 'Every true target norm must be finite.')
        self.api = api or NativeAPI(dll)
        self.allocator = PoolAllocator(cp, resource_callback, self.device, self.stream)
        started = time.perf_counter()
        try:
            for key, target in self.targets.items():
                state = {name: C.c_void_p() for name in ('handle', 'config', 'data', 'matrix')}
                state.update(dense={}, config_actual_readbacks=[])
                self.states[key] = state
                self._call('cudssCreate', C.byref(state['handle']))
                self._call('cudssSetStream', state['handle'], C.c_void_p(self.stream))
                self._call('cudssSetDeviceMemHandler', state['handle'], C.byref(self.allocator.handler))
                self._call('cudssConfigCreate', C.byref(state['config']))
                for param, value in ((9, C.c_double(0.0)), (4, C.c_int(0)), (5, C.c_int(3)), (6, C.c_double(0.0)), (11, C.c_int(0)), (15, C.c_int(0))):
                    self._call('cudssConfigSet', state['config'], param, C.byref(value), C.sizeof(value))
                    actual = type(value)()
                    written = C.c_size_t()
                    self._call('cudssConfigGet', state['config'], param, C.byref(actual), C.sizeof(actual), C.byref(written))
                    need(written.value == C.sizeof(actual) and actual.value == value.value, 'Exact native no-perturbation/device-only settings must roundtrip.')
                    state['config_actual_readbacks'].append(dict(parameter_index=param, value=actual.value, actual_bytes_written=written.value, expected_bytes=C.sizeof(actual)))
                self._call('cudssDataCreate', state['handle'], C.byref(state['data']))
                self._call('cudssMatrixCreateCsr', C.byref(state['matrix']), self.shape[0], self.shape[1], target.nnz, C.c_void_p(target.indptr.data.ptr), None, C.c_void_p(target.indices.data.ptr), C.c_void_p(target.data.data.ptr), 10, 10, 1, 0, 0, 0)
                self._dense(state, 32)
                self._execute(key, 3, 32)
                estimates = np.zeros(16, dtype='i8')
                written = C.c_size_t()
                self._call('cudssDataGet', state['handle'], state['data'], 13, C.c_void_p(estimates.ctypes.data), estimates.nbytes, C.byref(written))
                need(written.value == estimates.nbytes and np.all(estimates[:4] >= 0), 'Actual full32 memory estimates must be returned.')
                state['estimates'] = estimates.tolist()
                self.log('native_full32_analysis_complete', orientation=key, estimates=state['estimates'])
            N, T = (self.states[k]['estimates'] for k in ('N', 'T'))
            joint_device = max(N[1] + T[0], T[1] + N[0])
            joint_host = max(N[3] + T[2], T[3] + N[2])
            need(type(future_shared_GPU_reserve_bytes) is int and future_shared_GPU_reserve_bytes >= 0, 'Explicit nonnegative future shared workspace reserve is required.')
            preallocate(cp, resource_callback, 'native_A_and_true_AT_joint_numeric_reserve', GPU_bytes=int(joint_device) + future_shared_GPU_reserve_bytes, CPU_bytes=int(joint_host), both_full32_estimates=[N, T], future_shared_GPU_workspace_planning_reserve_bytes=future_shared_GPU_reserve_bytes, planning_reserve_not_a_measured_ANN_peak=True)
            for key in ('N', 'T'):
                t = time.perf_counter()
                self._execute(key, 4, 32)
                self.factorization_records.append(dict(orientation=key, seconds=time.perf_counter() - t, general_full_matrix=True, pivot_epsilon_actual_roundtrip_zero=True, DATA_INFO_zero=True))
            self.analysis = dict(backend='native_cudss0_8_full_general_A_and_true_AT_FP64', no_shift_drop_symmetry_reinterpretation_or_pivot_replacement=True, independent_transpose_factor=True, actual_full32_analysis_joint_reserve=dict(device=joint_device, host=joint_host), actual_memory_estimates={k: s['estimates'] for k, s in self.states.items()}, actual_config_readbacks={k: s['config_actual_readbacks'] for k, s in self.states.items()}, actual_full_target_CSR_byte_identity=matrix_byte_checks, fixed_IR_N_STEPS=3, fixed_IR_TOL=0.0, actual_solve_mode_normal_roundtrip_zero=True, factorization_records=self.factorization_records, setup_seconds=time.perf_counter() - started, actual_native_phase_INFO_records=self.phase_records, actual_native_runtime_does_not_qualify_science_or_budget=True)
        except BaseException:
            self.close()
            raise

    def _call(self, name, *args):
        try:
            result = self.api.call(name, *args)
        except BaseException:
            self.allocator.raise_pending()
            raise
        self.allocator.raise_pending()
        return result

    def _dense(self, state, columns):
        if columns in state['dense']:
            return state['dense'][columns]
        n = self.shape[0]
        preallocate(self.cp, self.resource_callback, 'native_complete_dense_' + str(columns), GPU_bytes=2 * n * columns * 8)
        pair = dict(B=self.cp.zeros((n, columns), dtype='f8', order='F'), X=self.cp.zeros((n, columns), dtype='f8', order='F'), b=C.c_void_p(), x=C.c_void_p())
        state['dense'][columns] = pair
        for array, descriptor in (('B', 'b'), ('X', 'x')):
            self._call('cudssMatrixCreateDn', C.byref(pair[descriptor]), n, columns, n, C.c_void_p(pair[array].data.ptr), 1, 0)
        return pair

    def _execute(self, key, phase, columns):
        state = self.states[key]
        pair = self._dense(state, columns)
        self._call('cudssExecute', state['handle'], phase, state['config'], state['data'], state['matrix'], pair['x'], pair['b'])
        self.cp.cuda.get_current_stream().synchronize()
        info = C.c_int()
        written = C.c_size_t()
        self._call('cudssDataGet', state['handle'], state['data'], 0, C.byref(info), C.sizeof(info), C.byref(written))
        need(written.value == C.sizeof(info) and info.value == 0, 'Actual native device DATA_INFO must be zero after sync.')
        actual_IR_steps = None
        if phase == 1008:
            count = C.c_int()
            written = C.c_size_t()
            self._call('cudssDataGet', state['handle'], state['data'], 24, C.byref(count), C.sizeof(count), C.byref(written))
            need(written.value == C.sizeof(count) and 0 <= count.value <= 3, 'Actual native finite IR step count must be returned.')
            actual_IR_steps = count.value
        self.resource_callback()
        self.phase_records.append(dict(orientation=key, phase=int(phase), columns=int(columns), actual_DATA_INFO=info.value, actual_DATA_INFO_bytes_written=C.sizeof(info), synchronized_before_INFO_readback=True, actual_IR_steps=actual_IR_steps))
        return actual_IR_steps

    def solve_device(self, rhs, trans='N'):
        need(not self.closed and threading.get_ident() == self.owner_thread, 'Native factor cannot cross owning threads or close.')
        need(int(self.cp.cuda.runtime.getDevice()) == self.device and int(self.cp.cuda.get_current_stream().ptr) == self.stream, 'Native factor cannot change owning CUDA device/stream.')
        need(trans in self.states and isinstance(rhs, self.cp.ndarray) and (rhs.dtype == np.dtype('f8')), 'Only exact original device FP64 N/T RHS is allowed.')
        vector = rhs.ndim == 1
        b = rhs.reshape(-1, 1) if vector else rhs
        need(b.ndim == 2 and b.shape[0] == self.shape[0] and (b.shape[1] in (1, 32)) and bool(self.cp.all(self.cp.isfinite(b))), 'Every complete native RHS must preserve all finite one/32 columns.')
        started = time.perf_counter()
        pair = self._dense(self.states[trans], b.shape[1])
        pair['B'][...] = b
        pair['X'].fill(0)
        actual_IR_steps = self._execute(trans, 1008, b.shape[1])
        x = pair['X'].copy(order='F')
        image = self.targets[trans] @ x
        r = b - image
        bn, rn = (self.cp.linalg.norm(b, axis=0), self.cp.linalg.norm(r, axis=0))
        denominator = self.host_norms[trans] * self.cp.max(abs(x), axis=0) + self.cp.max(abs(b), axis=0)
        need(all((bool(self.cp.all(self.cp.isfinite(v))) for v in (x, image, r, bn, rn, denominator))), 'All native original-target residuals/norms/denominators must be finite.')
        rel = rn / self.cp.maximum(bn, 1e-300)
        eta = self.cp.max(abs(r), axis=0) / self.cp.maximum(denominator, 1e-300)
        need(bool(self.cp.all(self.cp.isfinite(rel))) and bool(self.cp.all(self.cp.isfinite(eta))) and (float(rel.max()) <= 1e-10) and (float(eta.max()) <= 128 * np.finfo('f8').eps), 'Native original full N/T residual AND eta failed.')
        record = dict(trans=trans, columns=b.shape[1], relative=self.cp.asnumpy(rel).tolist(), eta=self.cp.asnumpy(eta).tolist(), original_general_matrix=True, DATA_INFO_zero=True, owned_output=True, pivot_epsilon_zero=True, fixed_IR_N_STEPS=3, fixed_IR_TOL=0.0, actual_IR_steps=actual_IR_steps, seconds=time.perf_counter() - started)
        self.calls.append(record)
        self.resource_callback()
        return x[:, 0] if vector else x

    def solve(self, rhs):
        return self.cp.asnumpy(self.solve_device(self.cp.asarray(rhs)))

    @contextmanager
    def local_threads(self, threads):
        yield

    def plan_evidence(self):
        return dict(no_SpSM_plan=True, actual_native_solve_records=self.calls, actual_native_allocation_records=self.allocator.records, actual_native_live_requested_bytes=sum((v[1] for v in self.allocator.live.values())))

    def close(self):
        if self.closed:
            return
        errors = []
        if hasattr(self, 'cp'):
            self.cp.cuda.get_current_stream().synchronize()
        for state in self.states.values():
            for pair in state['dense'].values():
                for name in ('b', 'x'):
                    if pair[name].value:
                        try:
                            self._call('cudssMatrixDestroy', pair[name])
                            pair[name] = C.c_void_p()
                        except BaseException as error:
                            errors.append(error)
            for name, fn, args in (('matrix', 'cudssMatrixDestroy', ()), ('data', 'cudssDataDestroy', (state['handle'],)), ('config', 'cudssConfigDestroy', ()), ('handle', 'cudssDestroy', ())):
                if state[name].value:
                    try:
                        self._call(fn, *args, state[name])
                        state[name] = C.c_void_p()
                    except BaseException as error:
                        errors.append(error)
            state['dense'].clear()
        self.targets.clear()
        self.closed = True
        if hasattr(self, 'allocator') and self.allocator.live:
            errors.append(RuntimeError('Native allocations remain after DataDestroy.'))
        if errors:
            raise errors[0]
