
import ast
import copy
from _neb_paths import Path
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
SOURCE=ROOT/'flow/nkb/common.py'
MINIMUM_GPUFREE=500000000
PREALLOCATION_RESERVE=256*2**20

def guard_tree():
    original=next(n for n in ast.parse(SOURCE.read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='guard')
    selected=copy.deepcopy(original);tests=[]
    for n in ast.walk(selected):
        if isinstance(n,ast.Compare) and isinstance(n.left,ast.Subscript) and isinstance(n.left.slice,ast.Constant) and n.left.slice.value=='gpu_free':tests.append(n)
    if len(tests)!=1 or not isinstance(tests[0].ops[0],ast.Gt):raise RuntimeError('Original exact resource condition changed')
    tests[0].comparators[0]=ast.Constant(value=MINIMUM_GPUFREE)
    return original,selected

def install(cm):
    original=cm.guard
    _,selected=guard_tree();namespace=dict(original.__globals__)
    exec(compile(ast.fix_missing_locations(ast.Module(body=[selected],type_ignores=[])),str(SOURCE),'exec'),namespace)
    cm.guard=namespace['guard']
    return dict(original_guard_source=str(SOURCE),minimum_GPUfree_bytes=MINIMUM_GPUFREE,
                user_unit='decimal0.5GB',pool_RSS_host_and_polling_unchanged=True)

def self_check():
    old,new=guard_tree();restored=copy.deepcopy(new)
    original_test=next(n for n in ast.walk(old) if isinstance(n,ast.Compare) and isinstance(n.left,ast.Subscript) and isinstance(n.left.slice,ast.Constant) and n.left.slice.value=='gpu_free')
    target=next(n for n in ast.walk(restored) if isinstance(n,ast.Compare) and isinstance(n.left,ast.Subscript) and isinstance(n.left.slice,ast.Constant) and n.left.slice.value=='gpu_free')
    target.comparators[0]=copy.deepcopy(original_test.comparators[0])
    assert ast.dump(old,include_attributes=False)==ast.dump(restored,include_attributes=False)
    valid=next(n for n in new.body if isinstance(n,ast.FunctionDef) and n.name=='valid')
    namespace={};exec(compile(ast.fix_missing_locations(ast.Module(body=[valid],type_ignores=[])),str(SOURCE),'exec'),namespace)
    r=dict(rss=8*2**30-1,host_free=9*2**30+1,gpu_free=MINIMUM_GPUFREE+1)
    assert namespace['valid'](r)
    assert not namespace['valid'](dict(r,gpu_free=MINIMUM_GPUFREE))
    assert not namespace['valid'](dict(r,rss=8*2**30))
    assert not namespace['valid'](dict(r,host_free=9*2**30))
    return dict(original_guard_AST_identical_except_only_GPUfree_constant=True,
        actual_decimal_threshold_and_unchanged_host_RSS_boundary_controls=True,
        minimum_GPUfree_bytes=MINIMUM_GPUFREE,preallocation_reserve_bytes=PREALLOCATION_RESERVE,GPU_calls=0)
