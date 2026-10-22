import csv, gc, os, time
from original_potential_k32_core import *

def fingerprint(model):
    h = None
    for name in ('W', 'b', 'mW', 'vW', 'mb', 'vb'):
        for a in getattr(model, name):
            h.update(name.encode())
            h.update(cp.asnumpy(a).tobytes())
    return None
