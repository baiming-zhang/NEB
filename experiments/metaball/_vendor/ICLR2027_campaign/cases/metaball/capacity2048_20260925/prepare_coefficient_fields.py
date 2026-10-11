from __future__ import annotations
import os,json,struct
import numpy as np
from pathlib import Path
def write(path, value):
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, indent=2))
    temp.replace(path)

def header(path):
    with path.open('rb') as stream:
        version = np.lib.format.read_magic(stream)
        if version == (1, 0):
            shape, order, dtype = np.lib.format.read_array_header_1_0(stream)
        else:
            shape, order, dtype = np.lib.format.read_array_header_2_0(stream)
        if order:
            raise ValueError('Require C-order storage')
        return (shape, dtype, stream.tell())
