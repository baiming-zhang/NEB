"""Portable NKB run paths and the public total-memory resource guard."""
from pathlib import Path
import os,sys
HERE=Path(__file__).resolve().parent
WORK=HERE.parent/'_backend';CAMPAIGN=WORK/'flow'
OUT=Path(os.environ.get('NEB_NKB_OUTPUT_ROOT','outputs/nkb'))
sys.path.insert(0,str(HERE.parents[1]/'periodic'))
from runtime import guard,read,write
