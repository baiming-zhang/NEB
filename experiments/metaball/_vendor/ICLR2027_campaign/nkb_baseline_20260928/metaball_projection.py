
import os
for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'): os.environ[key] = '1'
import argparse
import ast
import hashlib
import sys
import time
import numpy as np
import cupy as cp
from common import CAMPAIGN, OUT, guard, read, write
from core import Model
from domains import Domain

BASE = CAMPAIGN/'cases/metaball'
REF = BASE/'results/laplacian1024_full_ns_reconstruction_v1'
GT = BASE/'results/ns1024_rom_rollout_v1/ns_dt0.0025'


def spectral_class():
    path = BASE/'ns1024_rom_rollout_v1/core.py'
    node = next(x for x in ast.parse(path.read_text()).body if isinstance(x, ast.ClassDef) and x.name == 'Spectral')
    namespace = dict(cp=cp, np=np, DNS=__import__('pathlib').Path(read(REF/'plan.json')['DNS']),
                     relative=lambda a,b: float(cp.linalg.norm(a-b)/cp.maximum(cp.linalg.norm(b),1e-30)))
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), 'exec'), namespace)
    return namespace['Spectral']


def load_model(run):
    cfg = read(run/'config.json')
    assert read(run/'training_complete.json')['epoch'] == 2000


    m = Model(cp, cfg['input_dim'], cfg['dim'], cfg['K'], cfg['seed'],
              cfg['width'], cfg['hidden_layers'], dtype='float64')
    m.load(run/'checkpoint_e2000.npz')
    return m
