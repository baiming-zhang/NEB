import os, sys, time, argparse, traceback, json, gc
from _neb_paths import Path
for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[key] = '1'
ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
CASE = ROOT / 'flow/cases/metaball'
sys.path.insert(0, str(ROOT / 'flow/nkb'))
from common import guard, read, write
from metaball_projection import REF, GT
sys.path.insert(0, str(CASE / 'potential_scalar'))
import runtime as rt
cp, np = (rt.cp, rt.np)
OUT = Path('_data/metaball/potential_capacity_20261005')
