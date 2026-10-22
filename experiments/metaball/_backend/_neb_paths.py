"""Explicit scientific data root shared by isolated vendor modules."""
import os
from pathlib import Path as SystemPath

VENDOR=SystemPath(__file__).resolve().parent
DATA=SystemPath(os.environ.get('NEB_DATA_DIR','data/metaball')).resolve()
class Path(type(SystemPath())):
    def __new__(cls,*parts):
        p=SystemPath(*parts)
        value=str(p).replace('\\','/')
        if value.startswith('_data/'):
            p=DATA/value[len('_data/'):]
        return super().__new__(cls,p)
    def __truediv__(self,other):
        p=super().__truediv__(other)
        try:rel=SystemPath(p).relative_to(VENDOR)
        except ValueError:return p
        if 'results' in rel.parts or 'qualification' in rel.parts or 'audit' in rel.parts:
            return Path(DATA/rel)
        return p
