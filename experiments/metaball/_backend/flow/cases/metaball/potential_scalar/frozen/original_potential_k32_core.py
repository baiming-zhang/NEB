




from canonical_poisson_residual_fit import *
from equivalent_harmonic_lift import lift_fields,raw_lift_jets
from native_cupy_verify import jvp

class NeuralPoints:
    def __init__(self,points,geo,c,dtype='f4'):
        self.x=cp.asarray(points,dtype=dtype);self.n=len(self.x)
        self.features=features1(self.x,cp.asarray(geo,dtype=dtype),c)
        self.envelope=envelope1(self.x,cp.asarray(geo,dtype=dtype),c)[0]
    def part(self,start,stop):
        return SimpleNamespace(n=stop-start,x=self.x[start:stop],
            features=Jet1(self.features.value[start:stop],self.features.grad[start:stop]),
            envelope=Jet1(self.envelope.value[start:stop],self.envelope.grad[start:stop]))

def network_fields(model,data,which='aud',chunk=2048):
    result={k:cp.empty(((data.n if k=='d' else 3*data.n),model.K),dtype='f8') for k in which}
    for first in range(0,data.n,chunk):
        stop=min(first+chunk,data.n);p=data.part(first,stop)
        for mode in range(0,model.K,4):
            f,_=model.forward(p,mode,mode+4)
            if 'a' in result:result['a'][3*first:3*stop,mode:mode+4]=f.value.reshape(4,-1).T
            if 'u' in result:result['u'][3*first:3*stop,mode:mode+4]=curl1(f).reshape(4,-1).T
            if 'd' in result:result['d'][first:stop,mode:mode+4]=sum(f.grad[...,j,j] for j in range(3)).T
    return result

def network_backward(model,data,value=None,curl=None,div=None,grads=None,chunk=2048):
    if grads is None:grads=([cp.zeros(w.shape,dtype='f8') for w in model.W],[cp.zeros(b.shape,dtype='f8') for b in model.b])
    gw,gb=grads
    for first in range(0,data.n,chunk):
        stop=min(first+chunk,data.n);p=data.part(first,stop);n=stop-first
        for mode in range(0,model.K,4):
            _,cache=model.forward(p,mode,mode+4,True)
            av=cp.zeros((4,n,3),dtype=data.x.dtype);ag=cp.zeros((4,n,3,3),dtype=data.x.dtype)
            if value is not None:av[:]=value[3*first:3*stop,mode:mode+4].T.reshape(4,n,3)
            if curl is not None:ag+=curl_adjoint(curl[3*first:3*stop,mode:mode+4].T.reshape(4,n,3).astype(data.x.dtype))
            if div is not None:
                for j in range(3):ag[...,j,j]+=div[first:stop,mode:mode+4].T
            dw,db=model.backward(p,mode,mode+4,cache,Jet1(av,ag))
            for j in range(len(gw)):gw[j][mode:mode+4]+=dw[j];gb[j][mode:mode+4]+=db[j]
    return grads

class Context:
    def __init__(self,c,geo,dtype='f4'):
        self.c,self.geo=c,cp.asarray(geo,dtype='f8')
        path=ROOT/'results/canonical_poisson_residual_fit_v1/attempt_001/samples.npz'
        points=np.load(path)['fit'];self.fit=NeuralPoints(points,self.geo,c,dtype)
        self.scalar=ResidualFitPoisson(c,self.geo,cp.asarray(points,dtype='f8'))
        self.hcoeff=cp.asarray(np.load(ROOT/'results/equivalent_harmonic_lift_v1/attempt_001/new_coefficients.npy'))
        assert self.hcoeff.shape==(self.scalar.n+10,1)
    def solve(self,rhs,transpose=False):
        self.scalar.deadline=time.perf_counter()+90
        answer=self.scalar.solve(rhs,transpose=transpose)
        record=self.scalar.solve_records[-1]
        if not record['converged'] or not bool(cp.all(cp.isfinite(answer))):
            raise FloatingPointError('Actual scalar solve unresolved/nonfinite: '+json.dumps(record))
        return answer

class CanonicalBatch:
    def __init__(self,context,points,dtype='f4'):
        self.context=context;self.data=NeuralPoints(points,context.geo,context.c,dtype);self.n=self.data.n
        self.x=cp.asarray(points,dtype='f8');self.G=context.scalar.operator(self.x,'grad')
        _,dq,_,_=lift_fields(self.x,context.geo,context.c)
        self.h=dq.reshape(-1,1)-self.G@context.hcoeff;self.hmass=cp.sum(self.h*self.h)
        if not math.isfinite(float(self.hmass)) or float(self.hmass)<=0:raise FloatingPointError('Degenerate harmonic removal vector')
    def Q(self,a):return a-self.h@((self.h.T@a)/self.hmass)
    def forward(self,a,coeff):return self.Q(a-self.G@coeff)
    def reverse(self,adj):
        ga=self.Q(adj);gd=-self.context.solve(self.G.T@ga,transpose=True)
        return ga,gd

def potential_objective(p,u,n):



    identity=SimpleNamespace(n=n,project=lambda a:a)
    return peel(p,u,identity,True)

def objective_gradients(model,context,batch,state):
    ga,gd=batch.reverse(state['objective']['GA'])
    grad=network_backward(model,batch.data,value=ga,curl=state['objective']['GU'])
    return network_backward(model,context.fit,div=gd,grads=grad),ga,gd

def compute(model,context,batch,gradients=True):
    context.scalar.solve_records=[]
    f=network_fields(model,batch.data,'au');d=network_fields(model,context.fit,'d')['d']
    coeff=context.solve(d);p=batch.forward(f['a'],coeff);obj=potential_objective(p,f['u'],batch.n)
    state=dict(a=f['a'],u=f['u'],d=d,coeff=coeff,p=p,objective=obj)
    if gradients:state['grads'],state['ga'],state['gd']=objective_gradients(model,context,batch,state)
    return state

def metrics(state,context):
    obj=state['objective']
    return dict(mean_rayleigh=float(cp.mean(obj['losses'])),min_rayleigh=float(cp.min(obj['losses'])),max_rayleigh=float(cp.max(obj['losses'])),
        gram_max=float(obj['gram_max']),order=cp.asnumpy(obj['order']).tolist(),
        scalar_max_iterations=max(x['iterations'] for x in context.scalar.solve_records),
        scalar_max_normal_residual=max(max(x['relative_normal_residuals']) for x in context.scalar.solve_records))

def parameter_tangent(model,data,dw,db,which='aud',chunk=2048):
    result={k:cp.empty(((data.n if k=='d' else 3*data.n),32),dtype='f8') for k in which}
    for first in range(0,data.n,chunk):
        stop=min(first+chunk,data.n);p=data.part(first,stop)
        for mode in range(0,32,4):
            f=jvp(model,p,[w[mode:mode+4] for w in dw],[b[mode:mode+4] for b in db],mode,mode+4)
            if 'a' in result:result['a'][3*first:3*stop,mode:mode+4]=f.value.reshape(4,-1).T
            if 'u' in result:result['u'][3*first:3*stop,mode:mode+4]=curl1(f).reshape(4,-1).T
            if 'd' in result:result['d'][first:stop,mode:mode+4]=sum(f.grad[...,j,j] for j in range(3)).T
    return result

def scalar_hessian(context,x,coeff):

    base=context.scalar.base;row,col,d,f,gf,lf,g,hh=base.entries(x)
    e,de,le,he,*_=geometry2(x,context.geo,context.c)
    hk=g[:,None,None]*cp.eye(3)[None]/base.radius**2+hh[:,None,None]*d[:,:,None]*d[:,None,:]/base.radius**4
    h=e[row,None,None]*hk+de[row,:,None]*gf[:,None,:]+gf[:,:,None]*de[row,None,:]+he[row]*f[:,None,None]
    rr=(9*row[:,None]+cp.arange(9)).reshape(-1);cc=cp.repeat(col,9)
    H=csr_matrix((h.reshape(-1)/base.center_e[cc],(rr,cc)),shape=(9*len(x),base.n))
    result=H@coeff[:base.n]
    xyz,ej,_=jet_geometry(x,context.geo,context.c)
    y=[2*a-jconst(cp.ones(len(x))) for a in xyz];pieces=[]
    for powers in POWERS:
        p=jconst(cp.ones(len(x)))
        for k in range(3):
            for _ in range(powers[k]):p=jmul(p,y[k])
        a=jmul(ej,p)
        pieces.append(cp.stack([a[:,INDEX[tuple(int(k==i)+int(k==j) for k in range(3))]]*(2 if i==j else 1) for i in range(3) for j in range(3)],axis=1).reshape(-1))
    return result+cp.stack(pieces,axis=1)@coeff[base.n:]
