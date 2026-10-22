
from stage_common import *
import numpy as np,cupy as cp
CASE=ROOT/'flow/cases/metaball';sys.path.insert(0,str(CASE));sys.path.insert(0,str(CASE/'vendor'))
from skfem import MeshTet
from piola_reference_fields import Reference32,PAIRS
from piola_radial_geometry import RadialMap
class Geometry:
    locate=Reference32.locate
    def __init__(self,p,t):
        self.mesh=MeshTet(p.T,t.T);self.p=cp.asarray(p);self.t=cp.asarray(t);self.v=self.p[self.t]
        self.inv=cp.linalg.inv((self.v[:,1:]-self.v[:,:1]).transpose(0,2,1));self.grad=cp.concatenate((-self.inv.sum(1)[:,None,:],self.inv),1)
        self.edges=cp.asarray(self.mesh.t2e.T);self.orient=cp.stack([1-2*(self.t[:,a]>self.t[:,b]) for a,b in PAIRS],1)
        self.curls=cp.stack([2*cp.cross(self.grad[:,a],self.grad[:,b]) for a,b in PAIRS],1)
        lo=cp.clip(cp.floor((self.v.min(1)-1e-10)*20).astype('i4'),0,19);hi=cp.clip(cp.floor((self.v.max(1)+1e-10)*20).astype('i4'),0,19)
        dims=hi-lo+1;counts=cp.prod(dims,1);ends=cp.cumsum(counts);total=int(ends[-1]);cells=cp.repeat(cp.arange(len(t)),counts)
        off=cp.arange(total)-cp.repeat(ends-counts,counts);d=dims[cells]
        xyz=lo[cells]+cp.stack((off%d[:,0],off//d[:,0]%d[:,1],off//(d[:,0]*d[:,1])),1)
        bins=xyz[:,0]+20*xyz[:,1]+400*xyz[:,2];order=cp.argsort(bins);bins=bins[order];cells=cells[order]
        cnt=cp.bincount(bins,minlength=8000);base=cp.cumsum(cnt)-cnt;slot=cp.arange(total)-base[bins]
        self.table=cp.full((8000,int(cnt.max())),-1,dtype='i4');self.table[bins,slot]=cells
    def query(self,mapping,x):
        positions=[];jac=[];det=[];ids=[]
        for off in range(0,len(x),2048):
            y,J,d,check=mapping.map(cp.asarray(x[off:off+2048]),inverse=True);cell=self.locate(y)
            assert bool(cp.all(cell>=0)) and check['inverse_point_max']<1e-10 and check['inverse_J_max']<1e-8
            positions.append(cp.asnumpy(y));jac.append(cp.asnumpy(J));det.append(cp.asnumpy(d));ids.append(cp.asnumpy(cell))
        return dict(x=np.concatenate(positions),J=np.concatenate(jac),det=np.concatenate(det),ids=np.concatenate(ids))
    def fields(self,allco,query):
        outa=[];outu=[]
        for off in range(0,len(query['ids']),2048):
            ids=cp.asarray(query['ids'][off:off+2048]);x=cp.asarray(query['x'][off:off+2048]);J=cp.asarray(query['J'][off:off+2048]);det=cp.asarray(query['det'][off:off+2048])
            bar=cp.einsum('nij,nj->ni',self.inv[ids],x-self.v[ids,0]);bar=cp.concatenate((1-bar.sum(1)[:,None],bar),1);g=self.grad[ids]
            phi=cp.stack([bar[:,a,None]*g[:,b]-bar[:,b,None]*g[:,a] for a,b in PAIRS],1)
            co=allco[self.edges[ids]]*self.orient[ids,:,None]
            a=cp.einsum('nei,nek->nik',phi,co);u=cp.einsum('nei,nek->nik',self.curls[ids],co)
            outa.append(cp.asnumpy(cp.einsum('nij,njk->nik',cp.linalg.inv(J).transpose(0,2,1),a)))
            outu.append(cp.asnumpy(cp.einsum('nij,njk->nik',J,u)/det[:,None,None]))
        return np.concatenate(outa),np.concatenate(outu)
