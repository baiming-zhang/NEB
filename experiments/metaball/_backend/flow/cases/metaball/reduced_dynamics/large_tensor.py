




import gc
import json
from _neb_paths import Path
import time
import numpy as np
from prepare_coefficient_fields import header,write


class Reader:
    def __init__(self,run):
        self.ready=json.loads((run/'derived_fields_complete.json').read_text())
        self.k,self.n=self.ready['K'],self.ready['points']
        self.bpath=Path(self.ready['original_fields']);self.dpath=Path(self.ready['derived_fields'])
        _,_,self.bo=header(self.bpath);_,_,self.do=header(self.dpath)
        self.bs,self.ds=self.bpath.open('rb'),self.dpath.open('rb')
    def read(self,first,count,cp):
        self.bs.seek(self.bo+first*3*self.k*4);self.ds.seek(self.do+first*6*self.k*8)
        b=np.fromfile(self.bs,dtype='f4',count=count*3*self.k).reshape(count,3,self.k)
        d=np.fromfile(self.ds,dtype='f8',count=count*6*self.k).reshape(count,6,self.k)
        b=cp.asarray(np.ascontiguousarray(b.transpose(1,2,0)),dtype='f8',blocking=True)
        d=cp.asarray(np.ascontiguousarray(d.transpose(1,2,0)),blocking=True)
        return b,d[:3],d[3:]


def add(raw,b,p,w,cp,first_output,mode_block=8):
    k,points=b.shape[1:];outputs=raw.shape[1]
    p=p[:,first_output:first_output+outputs]
    for j in range(0,k,mode_block):
        end=min(k,j+mode_block);target=raw[j:end].reshape(-1,k)
        for a,bb,c,sign in ((0,1,2,1),(0,2,1,-1),(1,2,0,1),(1,0,2,-1),(2,0,1,1),(2,1,0,-1)):
            product=(b[bb,j:end,None,:]*p[a,None,:,:]).reshape(-1,points)
            target+=sign*(product@w[c].T)


def save_checkpoint(path,raw,cp,**metadata):

    manifest=path.with_suffix('.json')
    old=json.loads(manifest.read_text()) if manifest.exists() else {}
    slot=1-int(old.get('slot',1));data=path.with_name(path.stem+f'_{slot}.npy')
    with data.open('wb') as stream:
        np.lib.format.write_array_header_1_0(stream,dict(shape=raw.shape,fortran_order=False,descr='<f8'))
        for j in range(0,len(raw),8):cp.asnumpy(raw[j:j+8]).tofile(stream)
    write(manifest,dict(slot=slot,file=str(data),**metadata))


def restore_checkpoint(path,raw,cp):
    meta=json.loads(path.with_suffix('.json').read_text())
    shape,dtype,offset=header(Path(meta['file']))
    assert shape==raw.shape and dtype==np.dtype('f8')
    with Path(meta['file']).open('rb') as stream:
        stream.seek(offset)
        for j in range(0,len(raw),8):
            count=min(8,len(raw)-j)*raw.shape[1]*raw.shape[2]
            host=np.fromfile(stream,dtype='f8',count=count).reshape(raw[j:j+8].shape)
            raw[j:j+8]=cp.asarray(host,blocking=True)
    return meta


def run(run,cp,progress):
    reader=Reader(run);k,n=reader.k,reader.n
    marker=run/'quadratic_complete.json'
    if marker.exists():
        receipt=json.loads(marker.read_text());assert receipt['contract']==reader.ready['contract']
        return
    outputs=min(k,(8*2**30)//(8*k*k))
    point_block=1024;pairs=k*(k+1)//2
    sync=cp.cuda.get_current_stream().synchronize
    timing=run/'assembly_timing.json'
    if not timing.exists():
        raw=cp.zeros((k,outputs,k),dtype='f8')
        b,p,w=reader.read(0,17,cp)

        add(raw,b,p,w,cp,0)
        from assemble_triads import triad_blocks
        read=lambda first,last:(p[:,:,first:last].transpose(2,0,1),b[:,:,first:last].transpose(2,0,1),w[:,:,first:last].transpose(2,0,1))
        for first,expected in triad_blocks(read,17,k,cp,point_block=17,mode_block=1,stop_mode=1):
            difference=float(cp.linalg.norm(raw[0,:]-expected[0,:outputs,:])/cp.linalg.norm(expected[0,:outputs,:]))
            assert difference<2e-12
        del b,p,w,expected
        raw.fill(0);sync();start=time.perf_counter()
        b,p,w=reader.read(0,point_block,cp);add(raw,b,p,w,cp,0);sync()
        seconds=time.perf_counter()-start
        estimate=seconds*n/point_block*k/outputs
        write(timing,dict(K=k,point_block=point_block,output_block=outputs,tile_seconds=seconds,
            full_assembly_estimate_seconds=estimate,estimated_only=True,actual_slab_check_relative=difference,
            excludes='checkpoint writes, tensor packing, prior preparation, subsequent integration/output'))
        progress(run,'assembly_timing_complete',K=k,estimated_assembly_hours=estimate/3600,tile_seconds=seconds)
        del b,p,w,raw;gc.collect();cp.get_default_memory_pool().free_all_blocks()
    total_seconds=0.
    jj,kk=cp.triu_indices(k)
    for out in range(0,k,outputs):
        count=min(outputs,k-out);packed=run/f'quadratic_rows_{out:04d}.npy'
        receipt=run/f'quadratic_rows_{out:04d}.json'
        if receipt.exists():
            r=json.loads(receipt.read_text());assert r['contract']==reader.ready['contract']
            total_seconds+=r['seconds'];continue
        raw=cp.zeros((k,count,k),dtype='f8')
        checkpoint=run/'tensor_slab_checkpoint'
        first_point=0;previous=0.
        if checkpoint.with_suffix('.json').exists():
            saved=json.loads(checkpoint.with_suffix('.json').read_text())
            if saved['first_output']==out:
                assert saved['contract']==reader.ready['contract']
                saved=restore_checkpoint(checkpoint,raw,cp)
                first_point=saved['points'];previous=saved['seconds']
        begin=time.perf_counter();last_save=begin;last_report=begin
        for first in range(first_point,n,point_block):
            if (run/'user_requested_stop.json').exists():raise RuntimeError('User stopped this coefficient task')
            points=min(point_block,n-first)
            b,p,w=reader.read(first,points,cp);add(raw,b,p,w,cp,out);sync();del b,p,w
            now=time.perf_counter();done=first+points
            if now-last_report>=30 or done==n:
                elapsed=previous+now-begin
                fraction=(out+count*done/n)/k
                progress(run,'full_quadrature_assembly',K=k,output_first=out,output_count=count,
                    points=done,total_points=n,fraction=fraction,seconds=total_seconds+elapsed,
                    estimated_remaining_seconds=(total_seconds+elapsed)*(1/fraction-1))
                last_report=now
            if now-last_save>=600 or done==n:
                save_checkpoint(checkpoint,raw,cp,first_output=out,points=done,
                    seconds=previous+time.perf_counter()-begin,contract=reader.ready['contract'])
                last_save=time.perf_counter()
        temp=packed.with_suffix('.tmp.npy')
        with temp.open('wb') as stream:
            np.lib.format.write_array_header_1_0(stream,dict(shape=(count,pairs),fortran_order=False,descr='<f8'))
            for i in range(count):
                values=raw[jj,i,kk]+cp.where(jj==kk,0.,raw[kk,i,jj])
                cp.asnumpy(values/128**3).tofile(stream)
        temp.replace(packed)
        seconds=previous+time.perf_counter()-begin
        write(receipt,dict(first_output=out,rows=count,seconds=seconds,contract=reader.ready['contract']))
        total_seconds+=seconds
        del raw,values;gc.collect();cp.get_default_memory_pool().free_all_blocks()
    write(marker,dict(K=k,points=n,contract=reader.ready['contract'],seconds=total_seconds,
        precision='float64',packed_shape=[k,pairs],output_block=outputs,all_nodes=True,all_modes=True,
        storage='quadratic_rows_*.npy',source='Original full-grid rotational generator, FP64 output slabs'))
    progress(run,'quadratic_complete',K=k,seconds=total_seconds)
