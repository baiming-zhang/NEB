





from dataclasses import dataclass
from native_cupy_core import cp,np,features1,jet_geometry,INDEX,curl_adjoint
PAIRS=[(0,0),(0,1),(0,2),(1,1),(1,2),(2,2)]
HI={p:i for i,p in enumerate(PAIRS)}
def hid(a,b):return HI[tuple(sorted((a,b)))]
@dataclass
class J2:
    value:object
    grad:object
    hess:object
def scaled(x,s):return J2(x.value*s,x.grad*s,x.hess*s)
def plus(x,y):return J2(x.value+y.value,x.grad+y.grad,x.hess+y.hess)
def zeros(x):return J2(cp.zeros_like(x.value),cp.zeros_like(x.grad),cp.zeros_like(x.hess))
def pack(x):
    axis=x.value.ndim-2
    return cp.concatenate([x.value,*[x.grad[...,j] for j in range(3)],*[x.hess[...,j] for j in range(6)]],axis=axis)
def unpack(x,n):
    q=x.reshape(*x.shape[:-2],10,n,x.shape[-1])
    return J2(q[...,0,:,:],cp.moveaxis(q[...,1:4,:,:],-3,-1),cp.moveaxis(q[...,4:,:,:],-3,-1))
def linear(x,W,b):
    z=unpack(pack(x)@W.transpose(0,2,1),x.value.shape[-2]);z.value+=b[:,None,:];return z
def linear_reverse(x,a,W):
    xp=pack(x);ap=pack(a);gw=ap.transpose(0,2,1)@xp
    return unpack(ap@W,x.value.shape[-2]),gw,a.value.sum(axis=1)
def sine(x):
    s=cp.sin(x.value);c=cp.cos(x.value)
    h=cp.stack([c*x.hess[...,k]-s*x.grad[...,i]*x.grad[...,j] for k,(i,j) in enumerate(PAIRS)],-1)
    return J2(s,c[...,None]*x.grad,h)
def sine_reverse(x,a):
    s=cp.sin(x.value);c=cp.cos(x.value)
    av=c*a.value-s*cp.sum(a.grad*x.grad,-1);ag=c[...,None]*a.grad
    for k,(i,j) in enumerate(PAIRS):
        av+=a.hess[...,k]*(-s*x.hess[...,k]-c*x.grad[...,i]*x.grad[...,j])
        ag[...,i]-=s*a.hess[...,k]*x.grad[...,j];ag[...,j]-=s*a.hess[...,k]*x.grad[...,i]
    return J2(av,ag,c[...,None]*a.hess)
def sine_tangent(x,d):
    s=cp.sin(x.value);c=cp.cos(x.value)
    h=cp.stack([c*d.hess[...,k]-s*d.value*x.hess[...,k]-c*d.value*x.grad[...,i]*x.grad[...,j]
       -s*(d.grad[...,i]*x.grad[...,j]+x.grad[...,i]*d.grad[...,j]) for k,(i,j) in enumerate(PAIRS)],-1)
    return J2(c*d.value,c[...,None]*d.grad-s[...,None]*d.value[...,None]*x.grad,h)
def product(e,x):
    v=e.value*x.value;g=e.value[...,None]*x.grad+e.grad*x.value[...,None]
    h=cp.stack([e.value*x.hess[...,k]+e.hess[...,k]*x.value+e.grad[...,i]*x.grad[...,j]+e.grad[...,j]*x.grad[...,i] for k,(i,j) in enumerate(PAIRS)],-1)
    return J2(v,g,h)
def product_reverse(e,a):
    av=e.value*a.value+cp.sum(e.grad*a.grad,-1)+cp.sum(e.hess*a.hess,-1);ag=e.value[...,None]*a.grad
    for k,(i,j) in enumerate(PAIRS):
        ag[...,i]+=e.grad[...,j]*a.hess[...,k];ag[...,j]+=e.grad[...,i]*a.hess[...,k]
    return J2(av,ag,e.value[...,None]*a.hess)
class FieldData:
    def __init__(self,points,geometry,c,dtype='f4'):
        self.x=cp.asarray(points,dtype=dtype);self.n=len(points);g=cp.asarray(geometry,dtype=dtype)
        f=features1(self.x,g,c);self.features=J2(f.value,f.grad,cp.zeros((*f.value.shape,6),dtype=dtype))

        _,e,_=jet_geometry(self.x.astype('f8'),cp.asarray(geometry,dtype='f8'),c)
        value=e[:,0,None];grad=cp.stack([e[:,INDEX[tuple(int(k==i) for k in range(3))]] for i in range(3)],-1)[:,None,:]
        hess=cp.stack([e[:,INDEX[tuple(int(k==i)+int(k==j) for k in range(3))]]*(2 if i==j else 1) for i,j in PAIRS],-1)[:,None,:]
        self.envelope=J2(value.astype(dtype),grad.astype(dtype),hess.astype(dtype))
def forward(model,data,start=0,stop=32,cache=False):
    x=data.features;hist=[];dtype=x.value.dtype
    for j in range(len(model.W)):
        W=model.W[j][start:stop].astype(dtype);b=model.b[j][start:stop].astype(dtype);pre=linear(x,W,b)
        if j<len(model.W)-1:
            omega=model.c.siren_first_omega0 if j==0 else model.c.siren_hidden_omega0;pre=scaled(pre,omega)
            if cache:hist.append((x,pre,omega,W))
            x=sine(pre)
        else:
            if cache:hist.append((x,None,1,W))
            x=pre
    return product(data.envelope,x),hist
def backward(model,data,hist,adj):
    a=product_reverse(data.envelope,adj);gw=[];gb=[]
    for source,pre,omega,W in hist[::-1]:
        if pre is not None:a=scaled(sine_reverse(pre,a),omega)
        a,w,b=linear_reverse(source,a,W);gw.append(w);gb.append(b)
    return gw[::-1],gb[::-1]
def jvp(model,data,dW,db,start,stop):
    x=data.features;d=zeros(x);dtype=x.value.dtype
    for j in range(len(model.W)):
        W=model.W[j][start:stop].astype(dtype);b=model.b[j][start:stop].astype(dtype)
        pre=linear(x,W,b);dt=plus(linear(d,W,cp.zeros_like(b)),linear(x,dW[j],db[j]))
        if j<len(model.W)-1:
            omega=model.c.siren_first_omega0 if j==0 else model.c.siren_hidden_omega0;pre=scaled(pre,omega);dt=scaled(dt,omega)
            x=sine(pre);d=sine_tangent(pre,dt)
        else:x=pre;d=dt
    return product(data.envelope,d)
def velocity(x):
    g=x.grad
    return cp.stack((g[...,2,1]-g[...,1,2],g[...,0,2]-g[...,2,0],g[...,1,0]-g[...,0,1]),-1)
def vorticity(x):
    h=x.hess
    return cp.stack([sum(h[...,b,hid(a,b)]-h[...,a,hid(b,b)] for b in range(3)) for a in range(3)],-1)
def velocity_grad(x):
    h=x.hess
    return cp.stack([cp.stack([h[...,(a+2)%3,hid(j,(a+1)%3)]-h[...,(a+1)%3,hid(j,(a+2)%3)] for j in range(3)],-1) for a in range(3)],-2)
def lift_reverse(gu,gw):
    ag=curl_adjoint(gu);ah=cp.zeros((*gu.shape,6),dtype=gu.dtype)
    for a in range(3):
        for b in range(3):ah[...,b,hid(a,b)]+=gw[...,a];ah[...,a,hid(b,b)]-=gw[...,a]
    return J2(cp.zeros_like(gu),ag,ah)
def trace_objective(U,O,adjoints=False):

    n=U.shape[0]//3;k=U.shape[1];M=U.T@U/n;K=O.T@O/n;e=cp.linalg.eigvalsh(M)
    if not bool(cp.all(cp.isfinite(e))) or float(e[-1])<=0 or float(e[0]/e[-1])<1e-12:
        raise FloatingPointError('Physical mass zero/rank deficient/condition exceeds1e12; no jitter or rescue')
    L=cp.linalg.cholesky(M);inv=cp.linalg.solve(L.T,cp.linalg.solve(L,cp.eye(k)));inv=(inv+inv.T)/2
    F=cp.trace(inv@K)/k;result=dict(loss=F,M=M,K=K,mass_eigenvalues=e,mass_condition=e[-1]/e[0])
    if adjoints:
        cross=inv@K@inv;cross=(cross+cross.T)/2
        result.update(GU=-2/(n*k)*(U@cross),GO=2/(n*k)*(O@inv))
    return result
def make_batches(points,geometry,c,dtype='f4',chunk=1024):return [FieldData(points[i:i+chunk],geometry,c,dtype) for i in range(0,len(points),chunk)]
def fields(model,batches,block=4):
    n=sum(d.n for d in batches);U=cp.empty((3*n,32));O=cp.empty_like(U);offset=0
    for data in batches:
        for j in range(0,32,block):
            f,_=forward(model,data,j,j+block);U[3*offset:3*(offset+data.n),j:j+block]=velocity(f).reshape(block,-1).T
            O[3*offset:3*(offset+data.n),j:j+block]=vorticity(f).reshape(block,-1).T
        offset+=data.n
    return U,O
def parameter_gradient(model,batches,gu,go,block=4):
    GW=[cp.zeros(w.shape,dtype='f8') for w in model.W];Gb=[cp.zeros(b.shape,dtype='f8') for b in model.b];offset=0
    for data in batches:
        for j in range(0,32,block):
            f,hist=forward(model,data,j,j+block,True);dtype=data.x.dtype
            ag=gu[3*offset:3*(offset+data.n),j:j+block].T.reshape(block,data.n,3).astype(dtype)
            ao=go[3*offset:3*(offset+data.n),j:j+block].T.reshape(block,data.n,3).astype(dtype)
            w,b=backward(model,data,hist,lift_reverse(ag,ao))
            for k in range(len(w)):GW[k][j:j+block]+=w[k];Gb[k][j:j+block]+=b[k]
        offset+=data.n
    return GW,Gb
