from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def test_2d_analytic_kernels_and_projection():
    subprocess.run([sys.executable, "-B", "experiments/2d/check_cpu.py"], cwd=ROOT, check=True)


def test_public_help_without_cuda_execution():
    code = """
import importlib.abc
from pathlib import Path
import runpy
import sys
class RejectNativeImport(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'cupy', 'cupyx', 'torch'}:
            raise AssertionError('Native import during CPU help: ' + fullname)
sys.meta_path.insert(0, RejectNativeImport())
kind, entry = sys.argv[1:]
sys.argv = [entry, '--help']
if kind == 'script':
    sys.path.insert(0, str(Path(entry).resolve().parent))
try:
    if kind == 'script':
        runpy.run_path(entry, run_name='__main__')
    else:
        runpy.run_module(entry, run_name='__main__')
except SystemExit as exc:
    assert exc.code in (None, 0), exc.code
else:
    raise AssertionError('Help parser did not exit')
print('CPU_HELP_WITHOUT_NATIVE_IMPORT_PASSED')
"""
    scripts = [
        "experiments/periodic/run.py", "experiments/periodic/plot.py",
        "experiments/multihole/run.py", "experiments/multihole/evaluate.py",
        "experiments/2d/nkb/run.py", "experiments/metaball/run.py",
        "experiments/metaball/prepare.py", "experiments/metaball/train.py",
        "experiments/metaball/evaluate.py", "experiments/metaball/operators.py",
        "experiments/metaball/capacity.py", "experiments/metaball/heldout.py",
        "experiments/metaball/heldout_samples.py", "experiments/gauge/run.py",
        "experiments/gauge/particular.py", "experiments/gauge/inputs.py",
    ]
    modules = [
        "experiments.multihole.reference.run",
        "experiments.multihole.reference.sparse_controls",
        "experiments.multihole.learned_rom.run",
        "experiments.multihole.learned_rom.compare",
        "experiments.metaball.heldout_geometry4",
        "experiments.metaball.heldout_samples",
    ]
    for kind, entries in [("script", scripts), ("module", modules)]:
        for entry in entries:
            result = subprocess.run(
                [sys.executable, "-B", "-c", code, kind, entry], cwd=ROOT,
                check=True, capture_output=True, text=True,
            )
            assert "CPU_HELP_WITHOUT_NATIVE_IMPORT_PASSED" in result.stdout
