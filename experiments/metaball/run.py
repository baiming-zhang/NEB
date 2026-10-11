"""Portable full-domain Metaball evaluation and autonomous reduced dynamics."""
from __future__ import annotations
import argparse, ast, csv, hashlib, importlib.util, json, os, time
from pathlib import Path
for name in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):
    os.environ.setdefault(name,'1')
import numpy as np
import psutil

HERE=Path(__file__).resolve().parent
VENDOR=HERE/'_vendor'
POINTS=1599754
def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module
def guard():
    vm=psutil.virtual_memory()
    if psutil.Process().memory_info().rss>=8*2**30 or vm.available<=9*2**30 or vm.used>=62_500_000_000:
        raise RuntimeError('CPU RSS/host reserve exceeded')
def publish(out,values):
    out=Path(out);out.parent.mkdir(parents=True,exist_ok=True)
    with out.open('x',encoding='utf-8') as f:
        json.dump(values,f,indent=2,allow_nan=False);f.flush();os.fsync(f.fileno())
def array_reader(path,shape=None):
    f=Path(path).open('rb');version=np.lib.format.read_magic(f)
    actual,order,dtype=np.lib.format._read_array_header(f,version)
    if order or dtype.hasobject or (shape is not None and tuple(actual)!=tuple(shape)):
        f.close();raise ValueError('Complete C-order numeric NPY shape required')
    offset=f.tell()
    if Path(path).stat().st_size!=offset+int(np.prod(actual))*dtype.itemsize:
        f.close();raise ValueError('Truncated or appended array')
    return f,actual,dtype
def blocks(path,K,points=POINTS,block=2048):
    f,shape,dt=array_reader(path,(points,3,K))
    try:
        for lo in range(0,points,block):
            guard();n=min(block,points-lo)
            B=np.frombuffer(f.read(n*3*K*dt.itemsize),dtype=dt).reshape(n*3,K).astype('f8')
            if not np.isfinite(B).all():raise FloatingPointError('Nonfinite full basis')
            yield lo,B
    finally:f.close()
def paired20(args):
    # Execute the exact original two-pass projection node. Only the stream
    # providers are local paths, so no historical catalog/coordination is used.
    original=VENDOR/'exports/projected_potential_scaleup_20261008/postprocessing/metaball20_fullmass_CPU_v2.py'
    tree=ast.parse(original.read_text('utf-8'))
    node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='project')
    def basis(spec,K,points,block):yield from blocks(spec,K,points,block)
    def targets(paths,points,block):
        streams=[array_reader(p,(points,3)) for p in paths]
        try:
            for lo in range(0,points,block):
                guard();n=min(block,points-lo)
                data=np.column_stack([np.frombuffer(f.read(n*3*dt.itemsize),dtype=dt).astype('f8') for f,_,dt in streams])
                if not np.isfinite(data).all():raise FloatingPointError('Nonfinite target')
                yield lo,data
        finally:
            for f,_,_ in streams:f.close()
    ns=dict(np=np,POINTS=POINTS,WEIGHT=1/128**3,BLOCK=2048,basis=basis,targets=targets)
    exec(compile(ast.Module(body=[node],type_ignores=[]),str(original),'exec'),ns)
    if len(args.targets)!=20:raise ValueError('Exactly twenty paired target fields required')
    out=Path(args.output);out.mkdir(parents=True,exist_ok=False)
    with np.load(args.mass,allow_pickle=False) as z:
        key=args.mass_key;M=z[key].astype('f8')
    start=time.perf_counter();coverage=[]
    result=ns['project'](M,args.basis,args.targets,args.K,observe=lambda phase,end:coverage.append([phase,end]))
    np.savez(out/'projection.npz',**result)
    with (out/'errors.csv').open('x',newline='',encoding='utf-8') as f:
        w=csv.writer(f);w.writerow(['case','relative_L2'])
        w.writerows(enumerate(result['direct_relative_L2']))
    e=result['direct_relative_L2']
    publish(out/'complete.json',dict(K=args.K,points=POINTS,cases=20,passes=2,
        mean=float(np.mean(e)),sample_SD=float(np.std(e,ddof=1)),ddof=1,
        actual_seconds=time.perf_counter()-start,final_coverage=coverage[-1],
        normal_residual_max=float(np.max(result['normal_residuals'])),
        Gram_direct_difference_max=float(np.max(np.abs(result['direct_relative_L2']-result['Gram_identity_errors'])))))
def fullmass(args):
    start=time.perf_counter();M=np.zeros((args.K,args.K),dtype='f8');covered=0
    for lo,B in blocks(args.basis,args.K):
        if lo!=covered:raise ValueError('Point order gap')
        M+=(1/128**3)*(B.T@B);covered+=len(B)//3
    values=np.linalg.eigvalsh((M+M.T)*.5)
    if covered!=POINTS or values[0]<=1e-12*values[-1]:raise ValueError('Fullmass basis is rank deficient')
    out=Path(args.output);out.parent.mkdir(parents=True,exist_ok=True)
    if out.exists():raise FileExistsError(out)
    np.savez(out,M=M,eigenvalues=values)
    print(json.dumps(dict(K=args.K,points=covered,rank=args.K,actual_seconds=time.perf_counter()-start)))
def rom(args):
    module=load('metaball_coefficient_rom',VENDOR/'ICLR2027_campaign/cases/metaball/capacity2048_20260925/coefficient_rom.py')
    with np.load(args.operators,allow_pickle=False) as z:M,A=z['M'].copy(),z['A'].copy()
    Q=np.load(args.quadratic,mmap_mode='r',allow_pickle=False)
    c=np.load(args.initial,allow_pickle=False).astype('f8')
    solver=module.CoefficientROM(M,A,Q)
    start=time.perf_counter();rows=[c.copy()]
    for _ in range(args.steps):
        guard();c=solver.step(c,args.dt)
        if not np.isfinite(c).all():raise FloatingPointError('Nonfinite autonomous ROM')
        rows.append(c.copy())
    out=Path(args.output)
    if out.exists():raise FileExistsError(out)
    np.save(out,np.asarray(rows),allow_pickle=False)
    print(json.dumps(dict(K=solver.K,steps=args.steps,dt=args.dt,online_seconds=time.perf_counter()-start,reference_forcing=False)))
def checkpoint(args):
    import zipfile
    rows=[]
    with zipfile.ZipFile(args.checkpoint) as z:
        for name in z.namelist():
            with z.open(name) as f:
                v=np.lib.format.read_magic(f);shape,order,dt=np.lib.format._read_array_header(f,v)
                row=dict(key=name[:-4],shape=list(shape),dtype=dt.str,fortran_order=order)
                if name in ('epoch.npy','step.npy','config_hash.npy','protocol_sha256.npy') and int(np.prod(shape))*dt.itemsize<=512:
                    row['value']=np.frombuffer(f.read(),dtype=dt).tolist()
                rows.append(row)
    print(json.dumps(rows,indent=2))
def main():
    p=argparse.ArgumentParser(__doc__);s=p.add_subparsers(dest='command',required=True)
    a=s.add_parser('paired20');a.add_argument('--basis',required=True);a.add_argument('--mass',required=True);a.add_argument('--mass-key',default='M');a.add_argument('--targets',nargs=20,required=True);a.add_argument('--K',type=int,required=True);a.add_argument('--output',required=True);a.set_defaults(func=paired20)
    a=s.add_parser('fullmass');a.add_argument('--basis',required=True);a.add_argument('--K',type=int,required=True);a.add_argument('--output',required=True);a.set_defaults(func=fullmass)
    a=s.add_parser('rom');a.add_argument('--operators',required=True);a.add_argument('--quadratic',required=True);a.add_argument('--initial',required=True);a.add_argument('--steps',type=int,default=1856);a.add_argument('--dt',type=float,default=.000625);a.add_argument('--output',required=True);a.set_defaults(func=rom)
    a=s.add_parser('inspect-checkpoint');a.add_argument('--checkpoint',required=True);a.set_defaults(func=checkpoint)
    args=p.parse_args();args.func(args)
if __name__=='__main__':main()
