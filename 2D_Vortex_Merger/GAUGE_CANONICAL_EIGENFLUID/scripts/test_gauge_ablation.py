import json
import sys
import hashlib
from pathlib import Path
import numpy as np
import cupy as cp
from verify_analytic import ROOT,CASE
sys.path.insert(0,str(CASE/'src'))
from eigenfluid.gen.vortex_merger2d.hardcoded_backend import HardcodedModel,cfg_namespace,peel_step
from train_gauge_ablation import peel_step as ablated

def model(enabled):
    p=ROOT/'outputs/neural_k64_e2000/inputs/initial_epoch0000.npz'
    meta=json.loads(p.with_suffix('.json').read_text())
    cfg=cfg_namespace(dict(meta['config'],modes=4,gauge_fixing_enabled=enabled))
    with np.load(p) as f: arrays={k:f[k] for k in f.files}
    return cfg,HardcodedModel(cfg,arrays)

def test_enabled_toggle_is_exact_original_forward_and_training_update():
    cfg,a=model(True); _,b=model(True)
    points=np.random.default_rng(88).uniform(0,2*np.pi,(256,2)).astype(np.float32)
    d1=peel_step(cfg,a,points,0.,0.,True)
    d2=ablated(cfg,b,points,0.,0.,True)
    assert d1['loss']==d2['loss']
    for x,y in zip(a.master_weight_stacks+a.master_bias_stacks,b.master_weight_stacks+b.master_bias_stacks):
        cp.testing.assert_array_equal(x,y)

def test_disabled_gauge_changes_objective_and_bias_gradient():
    cfg,a=model(False); _,b=model(False)
    points=np.random.default_rng(88).uniform(0,2*np.pi,(256,2)).astype(np.float32)
    d1=peel_step(cfg,a,points,0.,0.,True)
    d2=ablated(cfg,b,points,0.,0.,True)
    assert abs(d1['loss']-d2['loss'])>1
    assert float(cp.max(abs(a.master_bias_stacks[-1]-b.master_bias_stacks[-1])).get())>1e-5

def test_pair_has_identical_initial_weights_and_validation():
    protocol=json.loads((ROOT/'outputs/gauge_ablation/protocol.json').read_text())
    assert protocol['initial_weights_sha256']==protocol['raw_initial_weights_sha256']
    assert protocol['validation_sha256']==protocol['raw_validation_sha256']
