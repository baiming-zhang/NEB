"""Network-only backend checks; DNS and CPU numerical paths are independent."""
from pathlib import Path
import ast
import os
import sys
import importlib.util
from types import SimpleNamespace

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
VENDOR = ROOT / 'experiments/metaball/_vendor'


def test_production_networks_do_not_use_torch_autograd():
    for path in (ROOT / 'experiments').rglob('*.py'):
        tree = ast.parse(path.read_text(encoding='utf-8-sig'))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                assert node.module not in {'torch.nn', 'torch.optim', 'torch.func'}, path
                if node.module == 'torch':
                    assert not ({a.name for a in node.names} & {'nn', 'optim', 'autograd'}), path
            if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name):
                if node.value.id == 'torch':
                    assert node.attr not in {'nn', 'optim', 'autograd', 'func'}, path


def native_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def test_cuda_analytic_network_forward_reverse_and_adam():
    if os.environ.get('NEB_RUN_CUDA_TESTS') != '1':
        pytest.skip('Set NEB_RUN_CUDA_TESTS=1 for the bounded native CUDA check')
    cp = pytest.importorskip('cupy')
    cp.get_default_memory_pool().set_limit(size=256 * 2**20)
    sys.path.insert(0, str(VENDOR))
    core = native_module(VENDOR / 'ICLR2027_campaign/cases/metaball/native_cupy_core.py',
                         'network_backend_core')
    cfg = SimpleNamespace(hidden=8, hidden_layers=2,
                          siren_first_omega0=2.3, siren_hidden_omega0=1.7)
    model = core.Model(cfg, K=2, seed=31)
    rng = np.random.default_rng(82)
    f = rng.normal(size=(5, 15)).astype('f4')
    j = rng.normal(size=(5, 15, 3)).astype('f4')
    e = rng.uniform(.3, .8, size=(5, 1)).astype('f4')
    ej = rng.normal(size=(5, 1, 3)).astype('f4')
    data = SimpleNamespace(features=core.Jet1(cp.asarray(f), cp.asarray(j)),
                           envelope=core.Jet1(cp.asarray(e), cp.asarray(ej)))
    value, tape = model.forward(data, 0, 2, True)
    # Independent NumPy forward and exact first spatial derivative.
    expected, derivatives = [], []
    for mode in range(2):
        x, dx = f.copy(), j.copy()
        for layer in range(cfg.hidden_layers):
            w, b = cp.asnumpy(model.W[layer][mode]), cp.asnumpy(model.b[layer][mode])
            omega = cfg.siren_first_omega0 if layer == 0 else cfg.siren_hidden_omega0
            pre = omega * (x @ w.T + b)
            dx = np.cos(pre)[..., None] * omega * np.einsum('oi,nid->nod', w, dx)
            x = np.sin(pre)
        w, b = cp.asnumpy(model.W[-1][mode]), cp.asnumpy(model.b[-1][mode])
        raw = x @ w.T + b
        expected.append(e * raw)
        derivatives.append(e[..., None] * np.einsum('oi,nid->nod', w, dx) + raw[..., None] * ej)
    np.testing.assert_allclose(cp.asnumpy(value.value), expected, rtol=2e-5, atol=2e-6)
    np.testing.assert_allclose(cp.asnumpy(value.grad), derivatives, rtol=2e-5, atol=2e-6)
    av = cp.asarray(rng.normal(size=value.value.shape).astype('f4'))
    ag = cp.asarray(rng.normal(size=value.grad.shape).astype('f4'))
    gw, gb = model.backward(data, 0, 2, tape, core.Jet1(av, ag))
    # Directional finite differences cover the joint value/derivative reverse.
    dw = [cp.asarray(rng.normal(size=w.shape).astype('f4')) for w in model.W]
    db = [cp.asarray(rng.normal(size=b.shape).astype('f4')) for b in model.b]
    exact = float(sum(cp.sum(a * b) for a, b in zip(gw + gb, dw + db)))
    parameters = model.W + model.b
    directions = dw + db
    original = [p.copy() for p in parameters]
    def loss(sign):
        for p, before, delta in zip(parameters, original, directions):
            p[...] = before + sign * 1e-3 * delta
        y, _ = model.forward(data, 0, 2)
        return float(cp.sum(y.value * av) + cp.sum(y.grad * ag))
    difference = (loss(1) - loss(-1)) / .002
    for p, before in zip(parameters, original):
        p[...] = before
    np.testing.assert_allclose(difference, exact, rtol=3e-3, atol=3e-3)
    model.adam(gw, gb, 1e-4, clip=1e6)
    assert model.epoch == 1
    for p, before, grad in zip(parameters, original, gw + gb):
        expected_step = cp.asnumpy(before) - 1e-4 * cp.asnumpy(grad) / (np.abs(cp.asnumpy(grad)) + 1e-8)
        np.testing.assert_allclose(cp.asnumpy(p), expected_step, rtol=2e-6, atol=2e-7)
    cp.cuda.Stream.null.synchronize()
    cp.get_default_memory_pool().free_all_blocks()
