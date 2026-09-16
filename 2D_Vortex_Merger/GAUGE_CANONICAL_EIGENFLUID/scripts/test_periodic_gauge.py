import json
import sys
from pathlib import Path
import cupy as cp
import numpy as np
from verify_analytic import ROOT,CASE,gauge_checks
sys.path.insert(0,str(CASE/'src'))
from eigenfluid.gen.vortex_merger2d.hardcoded_backend import HardcodedModel,cfg_namespace,peel_step
from evaluate_neural import evaluate_fields

def models():
    initial=ROOT/'outputs/neural_k64_e2000/inputs/initial_epoch0000.npz'
    meta=json.loads(initial.with_suffix('.json').read_text())
    cfg=cfg_namespace(dict(meta['config'],modes=4))
    with np.load(initial) as f: arrays={key:f[key] for key in f.files}
    shifted={key:value.copy() for key,value in arrays.items()}
    for i in range(4): shifted[f'm{i:03d}.output.bias']+=.1*(i+1)
    return cfg,HardcodedModel(cfg,arrays),HardcodedModel(cfg,shifted)

def test_actual_backend_constant_gauge_before_sort_and_mgs():
    cfg,model,shifted=models()
    points=np.random.default_rng(5).uniform(0,2*np.pi,(1024,2)).astype(np.float32)
    original=peel_step(cfg,model,points,0.,0.,False,full_diagnostics=True)
    changed=peel_step(cfg,shifted,points,0.,0.,False,full_diagnostics=True)
    np.testing.assert_allclose(original['loss'],changed['loss'],rtol=1e-5)
    assert original['peeled_rank']==changed['peeled_rank']==4
    before=evaluate_fields(model,cfg,points)
    after=evaluate_fields(shifted,cfg,points)
    for left,right in zip(before[1:],after[1:]): np.testing.assert_allclose(left,right,atol=1e-8)

def test_actual_neural_periodicity_in_value_and_jets():
    cfg,model,_=models()
    edge=np.linspace(0,2*np.pi,64,endpoint=False)
    p=np.concatenate([np.column_stack((edge,np.zeros(64))),np.column_stack((edge,np.full(64,2*np.pi)))])
    for field in evaluate_fields(model,cfg,p): np.testing.assert_allclose(field[:64],field[64:],atol=2e-5)

def test_analytic_gauge_and_fixed_boundary_control():
    result=gauge_checks()
    assert result['k64_fft_eigen_residual_max']<1e-9
    assert not result['dirichlet_control']['constant_shift_admissible']
