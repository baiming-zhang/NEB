
import ctypes as ct
import json
import os
from _neb_paths import Path
import threading
import time
import psutil

HERE = Path(__file__).resolve().parent
CAMPAIGN = HERE.parent
WORK = CAMPAIGN.parent
OUT = Path('_data/nkb')


def read(path): return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def write(path, data):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(data, indent=2, allow_nan=False), encoding='utf-8')
    deadline = time.monotonic()+10
    while True:
        try:
            tmp.replace(path); break
        except PermissionError:
            if time.monotonic() >= deadline: raise
            time.sleep(.05)


def guard(cp,run,pool_gib=4):
    from train import resource
    stop=threading.Event()
    resource(cp,run)
    return stop,{}
