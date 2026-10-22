
from _neb_paths import Path
import json
import shutil
import time
import numpy as np
from prepare_coefficient_fields import header


class RowTensor:
    def __init__(self,run,k,cp,cache_bytes=6*2**30):
        self.shape=(k,k*(k+1)//2);self.nbytes=8*self.shape[0]*self.shape[1]
        self.cp=cp;self.rows=16;self.cache={};self.files=[]
        self.disk_read_seconds=0.;self.disk_bytes_read=0
        used=0
        for path in sorted(Path(run).glob('quadratic_rows_*.npy')):
            start=int(path.stem.rsplit('_',1)[1]);shape,dtype,offset=header(path)
            assert shape[1]==self.shape[1] and dtype==np.dtype('f8')
            assert path.stat().st_size==offset+8*shape[0]*shape[1]
            self.files.append((start,start+shape[0],path,offset))
        assert self.files and self.files[0][0]==0 and self.files[-1][1]==k
        assert all(a[1]==b[0] for a,b in zip(self.files,self.files[1:]))
        for first in range(0,k,self.rows):
            count=min(self.rows,k-first);size=8*count*self.shape[1]
            if used+size>cache_bytes:break
            host=self._read(first,first+count)
            self.cache[first]=cp.asarray(host,blocking=True)
            used+=size
        self.cache_bytes=used
    def _read(self,first,last):
        tick=time.perf_counter()
        output=np.empty((last-first,self.shape[1]),dtype='f8')
        for start,end,path,offset in self.files:
            lo,hi=max(start,first),min(end,last)
            if lo>=hi:continue
            with path.open('rb') as stream:
                stream.seek(offset+(lo-start)*self.shape[1]*8)
                view=memoryview(output[lo-first:hi-first]).cast('B')
                if stream.readinto(view)!=len(view):raise IOError('Truncated packed tensor')
        self.disk_read_seconds+=time.perf_counter()-tick
        self.disk_bytes_read+=output.nbytes
        return output
    def __getitem__(self,key):
        assert isinstance(key,slice) and key.step is None
        first,last=key.start,min(key.stop,self.shape[0])
        if first in self.cache and len(self.cache[first])==last-first:return self.cache[first]
        return self._read(first,last)


def load(run,k,cp):
    if k<=1024:
        tensor=cp.empty((k,k*(k+1)//2),dtype='f8')
        storage=RowTensor(run,k,cp,cache_bytes=0)
        for first in range(0,k,16):tensor[first:first+16]=cp.asarray(storage._read(first,min(k,first+16)),blocking=True)
        return tensor,tensor.nbytes,dict(backend='GPU resident FP64',tensor_bytes=tensor.nbytes,gpu_tensor_bytes=tensor.nbytes)

    run=Path(run)
    cache=Path(__file__).resolve().parents[4]/f'_runtime_cache/case3_reduced_dynamics/coefficient_k{k}_tensor'
    sources=sorted(run.glob('quadratic_rows_*.npy'))
    total=sum(path.stat().st_size for path in sources)
    cache.mkdir(parents=True,exist_ok=True)
    contract=json.loads((run/'quadratic_complete.json').read_text())['contract']
    receipt=cache/'source.json'
    existing=json.loads(receipt.read_text()) if receipt.exists() else None
    if existing is not None and existing!=contract:raise ValueError('SSD tensor source mismatch')
    missing=sum(src.stat().st_size for src in sources if not (cache/src.name).exists())
    if shutil.disk_usage(cache).free<missing+5*2**30:raise RuntimeError('Insufficient SSD tensor headroom')
    for source in sources:
        target=cache/source.name
        if target.exists():
            assert target.stat().st_size==source.stat().st_size
            continue
        temp=target.with_suffix('.copying')
        shutil.copyfile(source,temp);temp.replace(target)
    receipt.write_text(json.dumps(contract,indent=2))
    tensor=RowTensor(cache,k,cp)
    return tensor,8*tensor.shape[1]*tensor.rows,dict(backend='FP64 coefficient tensor, GPU cache and bounded host staging',
        tensor_bytes=tensor.nbytes,gpu_tensor_bytes=tensor.cache_bytes,staging_rows=tensor.rows,
        remainder='Packed coefficient tensor files on local SSD via OS cache; all transfer time included online',
        ssd_cache=str(cache))
