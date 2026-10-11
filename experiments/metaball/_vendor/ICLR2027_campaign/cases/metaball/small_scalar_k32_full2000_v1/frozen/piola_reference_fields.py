
import sys,zipfile
from _neb_paths import Path
from native_cupy_core import cp,np,ROOT
sys.path.insert(0,str(ROOT/'vendor'))
from skfem import MeshTet
PAIRS=[(0,1),(1,2),(0,2),(0,3),(1,3),(2,3)]

class Reference32:
    def __init__(self,path):
        with np.load(path) as z:
            self.mesh=MeshTet(z['p'],z['t']);self.free=cp.asarray(z['free_edges']);self.lam=cp.asarray(z['eigenvalues'][:32]);self.guard=float(z['eigenvalues'][32])
        mesh=self.mesh;self.p=cp.asarray(mesh.p.T);self.t=cp.asarray(mesh.t.T);self.v=self.p[self.t]
        self.inv=cp.linalg.inv((self.v[:,1:]-self.v[:,:1]).transpose(0,2,1))
        self.grad=cp.concatenate((-self.inv.sum(1)[:,None,:],self.inv),axis=1)
        self.volume=cp.abs(cp.linalg.det((self.v[:,1:]-self.v[:,:1]).transpose(0,2,1)))/6
        self.edges=cp.asarray(mesh.t2e.T);self.orient=cp.stack([1-2*(self.t[:,a]>self.t[:,b]) for a,b in PAIRS],1)
        with zipfile.ZipFile(path) as z,z.open('potential.npy') as f:
            version=np.lib.format.read_magic(f);shape,fort,dtype=np.lib.format._read_array_header(f,version);assert not fort
            prefix=np.empty((shape[0],32),dtype=dtype)
            for i in range(0,len(prefix),128):
                n=min(128,len(prefix)-i);prefix[i:i+n]=np.frombuffer(f.read(n*shape[1]*dtype.itemsize),dtype=dtype).reshape(n,shape[1])[:,:32]
        self.co=cp.zeros((mesh.edges.shape[1],32));self.co[self.free]=cp.asarray(prefix);del prefix
        local=self.co[self.edges]*self.orient[:,:,None]
        curls=cp.stack([2*cp.cross(self.grad[:,a],self.grad[:,b]) for a,b in PAIRS],1)
        self.u=cp.einsum('nea,nek->nak',curls,local)/cp.sqrt(self.lam)[None,None,:]
        lo=cp.clip(cp.floor((self.v.min(1)-1e-10)*20).astype('i4'),0,19);hi=cp.clip(cp.floor((self.v.max(1)+1e-10)*20).astype('i4'),0,19)
        dims=hi-lo+1;counts=cp.prod(dims,1);ends=cp.cumsum(counts);total=int(ends[-1]);cells=cp.repeat(cp.arange(len(self.t)),counts)
        off=cp.arange(total)-cp.repeat(ends-counts,counts);d=dims[cells]
        xyz=lo[cells]+cp.stack((off%d[:,0],off//d[:,0]%d[:,1],off//(d[:,0]*d[:,1])),1)
        bins=xyz[:,0]+20*xyz[:,1]+400*xyz[:,2];order=cp.argsort(bins);bins=bins[order];cells=cells[order]
        cnt=cp.bincount(bins,minlength=8000);base=cp.cumsum(cnt)-cnt;slot=cp.arange(total)-base[bins]
        self.table=cp.full((8000,int(cnt.max())),-1,dtype='i4');self.table[bins,slot]=cells

    def locate(self,x):
        result=[]
        for i in range(0,len(x),256):
            q=x[i:i+256];b=cp.clip(cp.floor(q*20).astype('i4'),0,19);ids=self.table[b[:,0]+20*b[:,1]+400*b[:,2]]
            safe=cp.maximum(ids,0);bar=cp.einsum('ncij,ncj->nci',self.inv[safe],q[:,None]-self.v[safe,0])
            inside=(ids>=0)&cp.all(bar>=-1e-10,2)&(bar.sum(2)<=1+1e-10);ok=inside.any(1)
            result.append(cp.where(ok,cp.min(cp.where(inside,ids,len(self.t)),1),-1))
        return cp.concatenate(result)

    def velocity(self,x):
        ids=self.locate(x);value=self.u[cp.maximum(ids,0)].copy();value[ids<0]=cp.nan
        return value,ids

    def lifted_curl_check(self,ids,J):

        g=cp.einsum('nij,nvj->nvi',cp.linalg.inv(J).transpose(0,2,1),self.grad[ids])
        local=self.co[self.edges[ids]]*self.orient[ids,:,None]
        curls=cp.stack([2*cp.cross(g[:,a],g[:,b]) for a,b in PAIRS],1)
        return cp.einsum('nea,nek->nak',curls,local)/cp.sqrt(self.lam)[None,None,:]

    def potential(self,x,ids):
        b=cp.einsum('nij,nj->ni',self.inv[ids],x-self.v[ids,0]);b=cp.concatenate((1-b.sum(1)[:,None],b),1)
        g=self.grad[ids];w=cp.stack([b[:,a,None]*g[:,bb]-b[:,bb,None]*g[:,a] for a,bb in PAIRS],1)
        local=self.co[self.edges[ids]]*self.orient[ids,:,None]
        return cp.einsum('nei,nek->nik',w,local)
