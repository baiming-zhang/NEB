"""Portable public inputs and the paper's exact reported experiment scope."""
import json
from pathlib import Path
import re
import ast
import importlib.util
import sys
from types import SimpleNamespace

ROOT=Path(__file__).resolve().parents[1]


def test_checkpoint_case_and_configuration_coverage():
    entries=json.loads((ROOT/'checkpoints/index.json').read_text())['checkpoints']
    byid={e['id']:e for e in entries}
    assert len(byid)==len(entries)
    for method in ('potential','velocity','nkb'):
        for seed in range(3):assert f'periodic-{method}-k1024-seed{seed}' in byid
    for k in (32,64,128,256,512):assert f'metaball-potential-k{k}-seed20260823' in byid
    for method in ('velocity','nkb'):
        for k in (32,64,128,256,512,1024):assert f'metaball-{method}-k{k}-seed20260823' in byid
    for k in (32,64,128,256):assert f'metaball-potential-fg-k{k}-seed20260823' in byid
    for e in entries:
        metadata=json.loads((ROOT/e['metadata']).read_text())
        config=json.loads((ROOT/e['config']).read_text())
        assert metadata['config']==config
        assert metadata['epoch']==e['epoch']
        for a in [e,*e.get('auxiliary',[])]:
            assert not Path(a['path']).is_absolute()
            assert (ROOT/a['path']).resolve().is_relative_to(ROOT)
            assert a['bytes']>0
        if e['method']=='Potential-AL':assert len(e['auxiliary'])==1
    assert byid['metaball-potential-k1024-timing25']['epoch']==25


def test_published_metadata_is_portable():
    for directory in ('checkpoints','data','docs'):
        for p in (ROOT/directory).rglob('*'):
            if p.suffix not in ('.json','.csv','.md'):continue
            text=p.read_text(encoding='utf-8-sig')
            assert not re.search(r'\b[a-f0-9]{64}\b',text),p
            assert not re.search(r'[A-Za-z]:[\\/]',text),p
    readme=(ROOT/'README.md').read_text(encoding='utf-8')
    for path in re.findall(r'\]\(([^)]+)\)',readme):
        if not path.startswith(('http:','https:','#')):assert (ROOT/path).is_file(),path


def test_static_configuration_dependencies_exist():
    names={p.name for p in (ROOT/'experiments').rglob('*.json')}
    for p in (ROOT/'experiments').rglob('*.py'):
        tree=ast.parse(p.read_text(encoding='utf-8'))
        for n in ast.walk(tree):
            if isinstance(n,ast.Constant) and isinstance(n.value,str) and 'configs/' in n.value and n.value.endswith('.json'):
                assert Path(n.value).name in names,(p,n.value)


def test_periodic_evaluator_uses_only_packaged_runtime(monkeypatch):
    # Import orchestration without native execution, with no experiment workspace.
    monkeypatch.setitem(sys.modules,'cupy',SimpleNamespace())
    p=ROOT/'experiments/periodic/paper_rollout.py'
    spec=importlib.util.spec_from_file_location('isolated_paper_rollout',p)
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    assert m.WORK.resolve()==(ROOT/'experiments/2d/_vendor').resolve()
    assert m.guard.__module__=='neb_periodic_runtime'
