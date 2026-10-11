
import json
from _neb_paths import Path
import shutil
import time
import numpy as np
from streamed_fields import StreamedFields


def ssd_copy(source, destination, contract, progress):
    source, destination = Path(source), Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    marker = destination.with_suffix('.json')
    expected = dict(source=str(source), source_bytes=source.stat().st_size, **contract)
    if destination.exists() and marker.exists():
        saved = json.loads(marker.read_text())
        if saved['contract'] == expected and destination.stat().st_size == expected['source_bytes']:
            return destination
        raise ValueError('SSD cache source contract mismatch')
    if shutil.disk_usage(destination.parent).free < expected['source_bytes'] + 2**30:
        raise RuntimeError('Insufficient SSD space for evaluation field cache')
    temp = destination.with_suffix('.npy.tmp')
    tick = time.perf_counter()
    with source.open('rb') as src, temp.open('wb') as dst:
        shutil.copyfileobj(src, dst, length=16*2**20)
    if temp.stat().st_size != expected['source_bytes']:
        raise IOError('Incomplete SSD copy')
    temp.replace(destination)
    marker.write_text(json.dumps(dict(contract=expected, copy_seconds=time.perf_counter()-tick), indent=2))
    progress('ssd_field_cache_ready', bytes=expected['source_bytes'], seconds=time.perf_counter()-tick)
    return destination


class TieredFields(StreamedFields):
    def __init__(self, path, xp, *, expected_points, expected_modes, gpu_bytes=2*2**30,
                 host_bytes=4*2**30, **kw):


        kw.setdefault('block_rows', max(4096, (128*2**20)//(4*expected_modes)))
        super().__init__(path, xp, expected_points=expected_points, expected_modes=expected_modes, **kw)
        self.gpu_blocks, self.host_blocks = {}, {}
        used_gpu = used_host = 0
        cached_rows = 0
        tick = self._start()
        with self.path.open('rb') as stream:
            stream.seek(self.offset)
            for off in range(0, 3*self.N, self.block_rows):
                count = min(self.block_rows, 3*self.N-off)
                size = count*self.K*4
                target = ('gpu' if used_gpu+size <= gpu_bytes else
                          'host' if used_host+size <= host_bytes else None)
                if target is None:
                    break
                host = np.fromfile(stream, dtype='f4', count=count*self.K).reshape(count, self.K)
                if target == 'gpu':
                    self.gpu_blocks[off] = xp.asarray(host, dtype='f4')
                    used_gpu += size
                else:
                    if hasattr(xp, 'cuda'):
                        memory = xp.cuda.alloc_pinned_memory(host.nbytes)
                        cached = np.frombuffer(memory, dtype='f4', count=host.size).reshape(host.shape)
                        cached[:] = host
                        self.host_blocks[off] = cached
                    else:
                        self.host_blocks[off] = host
                    used_host += size
                cached_rows += count
                del host
        self.storage = dict(gpu_bytes=used_gpu, host_bytes=used_host,
                            streamed_bytes=(3*self.N-cached_rows)*self.K*4, source=str(self.path),
                            block_rows=self.block_rows, gpu_precision_promotion=True,
                            pinned_host=hasattr(xp, 'cuda'))
        self._finish('tier_cache_setup', tick)

    def blocks(self):


        with self.path.open('rb') as stream:
            for off in range(0, 3*self.N, self.block_rows):
                if off in self.gpu_blocks:
                    block = self.gpu_blocks[off].astype('f8')
                elif off in self.host_blocks:
                    block = self.xp.asarray(self.host_blocks[off], dtype='f4').astype('f8')
                else:
                    count = min(self.block_rows, 3*self.N-off)
                    stream.seek(self.offset+off*self.K*4)
                    host = np.fromfile(stream, dtype='f4', count=count*self.K)
                    if host.size != count*self.K:
                        raise IOError('SSD field cache truncated')



                    kwargs = {'blocking': True} if hasattr(self.xp, 'cuda') else {}
                    block = self.xp.asarray(host.reshape(count,self.K), dtype='f4', **kwargs).astype('f8')
                    del host
                yield off, block
                del block
