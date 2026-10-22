
import os,time,json,sys,math,gc,shutil,argparse,traceback,importlib.util
from _neb_paths import Path
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[key]='1'
HERE=Path(__file__).resolve().parent;CASE=HERE.parent
sys.path.insert(0,str(CASE/'fixed_geometry_k256_20260921'))
from common import read,write,journal
sys.path.insert(0,str(CASE/'velocity'))
import core,fused,fast_path
fused.install();fast_path.install()
cp,np=core.cp,core.np
from types import SimpleNamespace
