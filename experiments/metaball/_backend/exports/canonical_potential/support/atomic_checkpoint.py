from __future__ import annotations
import json
import math
import os
from _neb_paths import Path
import time
from typing import Callable, Mapping
import numpy as np
FIELDS = ('W', 'b', 'mW', 'vW', 'mb', 'vb')

def _replace(source, target):
    deadline = time.monotonic() + 2.0
    while True:
        try:
            os.replace(source, target)
            return
        except PermissionError:
            if time.monotonic() >= deadline:
                raise
            time.sleep(0.025)

def _atomic_json(path, value, validate=None):
    path = Path(path)
    temporary = path.with_name(path.name + f'.{os.getpid()}.{time.time_ns()}.tmp')
    encoded = json.dumps(value, indent=2, ensure_ascii=True, allow_nan=False).encode('utf-8')
    with temporary.open('xb') as stream:
        stream.write(encoded)
        stream.flush()
        os.fsync(stream.fileno())
    reread = json.loads(temporary.read_text(encoding='utf-8'))
    if reread != value:
        raise ValueError('Written JSON differs from requested complete state')
    if validate:
        validate(reread)
    _replace(temporary, path)

def _finite_json(value):
    json.dumps(value, ensure_ascii=True, allow_nan=False)

def _check_auxiliary_schema(arrays, schema):
    if not schema:
        raise ValueError('Applicable auxiliary state requires its full declared array schema')
    for name, declared in schema.items():
        if name not in arrays:
            raise ValueError('Missing declared auxiliary state: ' + name)
        actual = arrays[name]
        if actual['shape'] != list(declared['shape']) or actual['dtype'] != np.dtype(declared['dtype']).str:
            raise ValueError('Auxiliary shape/dtype differs from declared same-epoch state: ' + name)

def inspect_npz(path, *, epoch, model=False, K=None, layers=6):
    path = Path(path)
    result = {}
    with np.load(path, allow_pickle=False) as archive:
        if int(archive['epoch']) != epoch:
            raise ValueError('Wrong checkpoint epoch')
        for name in archive.files:
            value = archive[name]
            if value.dtype.hasobject:
                raise ValueError('Object array is not a valid safe restart payload: ' + name)
            if value.dtype.kind in 'fc' and (not np.isfinite(value).all()):
                raise ValueError('Nonfinite restart array: ' + name)
            result[name] = dict(shape=list(value.shape), dtype=value.dtype.str)
        if model:
            required = {f'{field}_{layer}' for field in FIELDS for layer in range(layers)}
            if not required.issubset(result):
                raise ValueError('Model is missing full parameter or Adam state')
            if K is None or K < 1:
                raise ValueError('Explicit positive complete mode count is required')
            for name in required:
                value = archive[name]
                if value.dtype != np.dtype('float32') or value.ndim < 1 or value.shape[0] != K:
                    raise ValueError('Incomplete mode count or nonhistorical model dtype: ' + name)
            for layer in range(layers):
                for param, moments in (('W', ('mW', 'vW')), ('b', ('mb', 'vb'))):
                    shape = archive[f'{param}_{layer}'].shape
                    if any((archive[f'{field}_{layer}'].shape != shape for field in moments)):
                        raise ValueError('Adam moment shape differs from its parameter')
    return result

def _timing(times, extra_io_seconds):
    required = ('normal_training_seconds', 'state_recovery_seconds', 'discarded_attempt_seconds')
    if any((key not in times for key in required)):
        raise ValueError('Separate actual normal/recovery/discarded timing is required')
    value = {key: float(times[key]) for key in required}
    if any((not math.isfinite(v) or v < 0 for v in value.values())):
        raise ValueError('Elapsed times must be finite nonnegative actual measurements')
    active = times.get('active_stage', 'normal_training')
    key = {'normal_training': 'normal_training_seconds', 'state_recovery': 'state_recovery_seconds'}.get(active)
    if key is None:
        raise ValueError('Only normal_training or state_recovery can publish a training bundle')
    value[key] += extra_io_seconds
    value.update(active_stage=active, total_active_seconds=sum((value[k] for k in required)), bundle_IO_included_as_of_seconds=extra_io_seconds, timing_as_of='Before final CURRENT atomic commit; full I/O remains in surrounding actual stage timer')
    return value

def verify_bundle(pointer, *, expected_epoch=None):
    root = Path(pointer['bundle_directory']).resolve()
    manifest_path = Path(pointer['complete_manifest']).resolve()
    if manifest_path != root / 'complete_manifest.json':
        raise ValueError('Manifest is not inside the exact declared bundle')
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    if manifest['status'] != 'complete_verified_same_epoch_restart_bundle':
        raise ValueError('Not a complete published bundle')
    epoch, K = (manifest['epoch'], manifest['K'])
    if pointer['epoch'] != epoch or (expected_epoch is not None and expected_epoch != epoch):
        raise ValueError('Published epoch differs from the requested state')
    bindings = manifest['bindings']
    for key in ('model', 'auxiliary_state'):
        item = manifest[key]
        if item.get('status') == 'not_applicable':
            if key != 'auxiliary_state':
                raise ValueError('The model is always applicable')
            continue
        if item['file'] != ('model.npz' if key == 'model' else 'auxiliary.npz'):
            raise ValueError('Unexpected or unsafe restart file name')
        path = root / item['file']
        actual = inspect_npz(path, epoch=epoch, model=key == 'model', K=K, layers=manifest['model_layers'])
        if actual != item['arrays']:
            raise ValueError('Restart payload array state changed: ' + key)
        if key == 'auxiliary_state':
            _check_auxiliary_schema(actual, item['declared_full_array_schema'])
    if manifest['counts']['optimizer_steps'] != epoch:
        raise ValueError('Optimizer step and bundle epoch must match')
    if manifest['sampler_state']['next_epoch'] != epoch + 1:
        raise ValueError('Sampler next epoch differs from optimizer state')
    _finite_json(manifest)
    return manifest

def publish_bundle(root, *, epoch: int, K: int, model_writer: Callable[[Path], None], sampler_state: Mapping, counts: Mapping, timing: Mapping, bindings: Mapping, method_state: Mapping, auxiliary_writer: Callable[[Path], None] | None=None, auxiliary_na_reason: str | None=None, auxiliary_expected_arrays: Mapping | None=None, layers: int=6):
    tic = time.perf_counter()
    root = Path(root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    if not isinstance(epoch, int) or epoch < 0 or (not isinstance(K, int)) or (K < 1):
        raise ValueError('Explicit complete integer epoch and mode count are required')
    if counts.get('optimizer_steps') != epoch or sampler_state.get('next_epoch') != epoch + 1:
        raise ValueError('Model epoch, Adam step and sampler state must refer to the same commit')
    if counts.get('validations') != (0 if bindings.get('scheduled_validations') == 0 else epoch // 25):
        raise ValueError('Save only after original scheduled validation commits')
    if counts.get('dual_updates', 0) < 0:
        raise ValueError('Invalid method-specific dual counter')
    if auxiliary_writer is None and (not auxiliary_na_reason):
        raise ValueError('Multipliers/auxiliary state require a writer or explicit N/A')
    if auxiliary_writer is not None and (not auxiliary_expected_arrays):
        raise ValueError('Declare every method-specific multiplier/scalar shape and dtype')
    for value in (sampler_state, counts, bindings, method_state):
        _finite_json(value)
    target = root / f'e{epoch:04d}'
    if target.exists():
        raise FileExistsError('Historical complete restart bundles are immutable')
    temporary = root / f'.e{epoch:04d}.{os.getpid()}.{time.time_ns()}.tmp'
    temporary.mkdir(exist_ok=False)
    model_writer(temporary / 'model.npz')
    model = temporary / 'model.npz'
    with model.open('r+b') as stream:
        stream.flush()
        os.fsync(stream.fileno())
    model_arrays = inspect_npz(model, epoch=epoch, model=True, K=K, layers=layers)
    aux = dict(status='not_applicable', reason=auxiliary_na_reason)
    if auxiliary_writer is not None:
        auxiliary_writer(temporary / 'auxiliary.npz')
        path = temporary / 'auxiliary.npz'
        with path.open('r+b') as stream:
            stream.flush()
            os.fsync(stream.fileno())
        aux_arrays = inspect_npz(path, epoch=epoch)
        _check_auxiliary_schema(aux_arrays, auxiliary_expected_arrays)
        aux = dict(status='applicable', file='auxiliary.npz', arrays=aux_arrays, declared_full_array_schema=dict(auxiliary_expected_arrays))
    manifest = dict(schema_version=1, status='complete_verified_same_epoch_restart_bundle', epoch=epoch, K=K, model_layers=layers, model=dict(file='model.npz', arrays=model_arrays), auxiliary_state=aux, sampler_state=dict(sampler_state), counts=dict(counts), bindings=dict(bindings), method_state=dict(method_state), actual_timing=_timing(timing, time.perf_counter() - tic), publication_order='write_all/fsync/verify_arrays/manifest/fsync/rename_directory/verify/CURRENT_atomic', unix=time.time())
    _atomic_json(temporary / 'complete_manifest.json', manifest)
    _replace(temporary, target)
    pointer = dict(epoch=epoch, K=K, bundle_directory=str(target), complete_manifest=str(target / 'complete_manifest.json'), status=manifest['status'], unix=time.time())

    def validate(p):
        verify_bundle(p, expected_epoch=epoch)
    _atomic_json(root / 'CURRENT.json', pointer, validate=validate)
    return dict(pointer=pointer, actual_bundle_IO_seconds=time.perf_counter() - tic, surrounding_actual_stage_timer_must_include_final_CURRENT_commit=True)
