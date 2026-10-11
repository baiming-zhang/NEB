





import sys, time, math, ast
from _neb_paths import Path
from contextlib import contextmanager
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent))
from native_cupy_core import cp,np,Model,fluid_points,curl_adjoint,write,sha
from velocity_trace_jet2 import J2,FieldData,forward,backward,jvp,velocity,velocity_grad,hid
from cupyx.scipy.linalg import solve_triangular
WORK=HERE.parents[3]
OLD=WORK/'3d_metaball_reproduction_latest/cases/3D_Metaball/scripts/train_validate.py'

names={'geometry_from_token','overlap_graph_connected','sphere_directions','spheres_inside_rounded_cube'}
tree=ast.parse(OLD.read_text(encoding='utf-8-sig'))
module=ast.Module(body=[x for x in tree.body if isinstance(x,ast.FunctionDef) and x.name in names],type_ignores=[])
ns={'np':np,'math':math,'Config':object};exec(compile(module,str(OLD),'exec'),ns)
ns['SPHERE_DIRECTIONS']=ns['sphere_directions']()
geometry_from_token=ns['geometry_from_token']

@contextmanager
def timed(record,name):
    cp.cuda.Stream.null.synchronize();t=time.perf_counter()
    try:yield
    finally:
        cp.cuda.Stream.null.synchronize();record[name]=record.get(name,0)+time.perf_counter()-t

def lift(gu,gd):
    h=cp.zeros((*gu.shape,6),dtype=gu.dtype)
    for a in range(3):
        for j in range(3):
            h[...,(a+2)%3,hid(j,(a+1)%3)]+=gd[...,a,j]
            h[...,(a+1)%3,hid(j,(a+2)%3)]-=gd[...,a,j]
    return J2(cp.zeros_like(gu),curl_adjoint(gu),h)

def grams(U,D):
    k=U.shape[1];n=U.shape[0]//3
    M=cp.zeros((k,k),dtype='f8');S=cp.zeros_like(M)
    for X,target in ((U,M),(D,S)):
        for start in range(0,len(X),3072):
            b=X[start:start+3072].astype('f8');target+=b.T@b/n
    return M,S

def coefficient_peel(M,S):
    k=len(M);mass=cp.diag(M);raw=cp.diag(S)/mass
    order=cp.argsort(raw,kind='stable');ms=M[order[:,None],order];ss=S[order[:,None],order]
    scale=cp.sqrt(cp.diag(ms));H=ms/scale[:,None]/scale[None,:]
    L=cp.linalg.cholesky(H)
    T=solve_triangular(L.T,cp.eye(k),lower=False)/scale[:,None]

    H2=T.T@ms@T;L2=cp.linalg.cholesky((H2+H2.T)*.5)
    T=T@solve_triangular(L2.T,cp.eye(k),lower=False)
    ortho=T.T@ms@T;error=cp.max(cp.abs(ortho-cp.eye(k)))
    norm=1/cp.diag(T);relative=norm*norm/cp.diag(ms)
    F=T.T@ss@T;loss=cp.diag(F)
    if not bool(cp.all(cp.isfinite(loss)&(loss>0)&(relative>=1e-4))) or float(error)>1e-5:
        raise FloatingPointError('Velocity rank/relative mass/orthogonality gate failed before Adam')


    C=-2*(T*loss[None,:]+T@cp.triu(F,1))/norm[None,:]
    B=2*T/norm[None,:]
    Cu=cp.empty_like(C);Cd=cp.empty_like(B);transform=cp.empty_like(T)
    Cu[order[:,None],order]=C;Cd[order[:,None],order]=B;transform[order]=T
    return dict(losses=loss,order=order,T=transform,Cu=Cu,Cd=Cd,relative_mass=relative,
                gram_max=error,raw=raw)

def fields(model,data,k,block):
    U=cp.empty((3*data.n,k),dtype='f4');D=cp.empty((9*data.n,k),dtype='f4')
    for start in range(0,k,block):
        stop=min(k,start+block);f,_=forward(model,data,start,stop)
        U[:,start:stop]=velocity(f).reshape(stop-start,-1).T
        D[:,start:stop]=velocity_grad(f).reshape(stop-start,-1).T
    return U,D

def compute(model,c,geo,points,training=True):
    tt={};k=model.K;n=len(points);chunk=c.point_chunk;block=c.mode_batch
    U=cp.empty((3*n,k),dtype='f4');D=cp.empty((9*n,k),dtype='f4');datas=[]
    with timed(tt,'geometry_preparation'):
        for off in range(0,n,chunk):datas.append((off,FieldData(points[off:off+chunk],geo,c)))
    with timed(tt,'neural_forward_analytic_derivatives'):
        for off,data in datas:
            u,d=fields(model,data,k,block);U[3*off:3*(off+data.n)]=u;D[9*off:9*(off+data.n)]=d
        del u,d
    with timed(tt,'velocity_sort_orthogonalization_objective'):
        M,S=grams(U,D);obj=coefficient_peel(M,S)
    if training:
        GW=[cp.zeros_like(w) for w in model.W];Gb=[cp.zeros_like(b) for b in model.b]
        with timed(tt,'neural_recompute_and_analytic_backward'):
            for off,data in datas:
                gu=(U[3*off:3*(off+data.n)].astype('f8')@obj['Cu']/n).astype('f4')
                gd=(D[9*off:9*(off+data.n)].astype('f8')@obj['Cd']/n).astype('f4')
                for start in range(0,k,block):
                    stop=min(k,start+block);f,hist=forward(model,data,start,stop,True)
                    au=gu[:,start:stop].T.reshape(stop-start,data.n,3)
                    ad=gd[:,start:stop].T.reshape(stop-start,data.n,3,3)
                    gw,gb=backward(model,data,hist,lift(au,ad))
                    for j in range(len(gw)):GW[j][start:stop]+=gw[j];Gb[j][start:stop]+=gb[j]
                    del f,hist,gw,gb
        obj['grads']=(GW,Gb)
    obj['timings']=tt
    return obj

def literal_peel(U,D):

    n=U.shape[0]//3;k=U.shape[1];order=cp.argsort(cp.sum(D*D,0)/cp.sum(U*U,0),kind='stable')
    Q=cp.empty_like(U);V=cp.empty_like(D);GU=cp.empty_like(U);GD=cp.empty_like(D);loss=[]
    for r,index in enumerate(cp.asnumpy(order)):
        u=U[:,index].copy();d=D[:,index].copy();q=Q[:,:r];v=V[:,:r]
        for _ in range(2):
            a=q.T@u/n;u-=q@a;d-=v@a
        m=cp.dot(u,u)/n;s=cp.dot(d,d)/n;loss.append(s/m)
        gu=-2*s*u/(n*m*m);gd=2*d/(n*m)
        for _ in range(2):gu-=q@(q.T@gu+v.T@gd)/n
        GU[:,index]=gu;GD[:,index]=gd;Q[:,r]=u/cp.sqrt(m);V[:,r]=d/cp.sqrt(m)
    return cp.asarray(loss),GU,GD,Q,V,order
