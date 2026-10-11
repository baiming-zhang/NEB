




from native_cupy_core import cp,np

class RadialMap:
    def __init__(self,mesh,geo,c):
        self.geo=cp.asarray(geo,dtype='f8');self.c=c;self.center=self.geo[1,:3]
        p=cp.asarray(mesh.p.T);t=cp.asarray(mesh.t.T)
        bid=cp.asarray(mesh.boundary_facets());faces=cp.asarray(mesh.facets.T)[bid]
        cells=cp.asarray(mesh.f2t[0])[bid];v=p[faces];q=v-self.center
        n=cp.cross(v[:,1]-v[:,0],v[:,2]-v[:,0]);n/=cp.linalg.norm(n,axis=1)[:,None]
        n*=cp.where(cp.sum(n*(v.mean(1)-p[t[cells]].mean(1)),1)>0,1,-1)[:,None]

        fo,fi,go,gi=self.level_grad(v.mean(1))
        outer=cp.abs(fo)/cp.maximum(cp.linalg.norm(go,axis=1),1e-30)<cp.abs(fi)/cp.maximum(cp.linalg.norm(gi,axis=1),1e-30)
        self.bid=bid;self.faces=faces;self.vertices=v;self.n=n;self.outer=outer;self.cells=cells
        self.surfaces=[];self.stats=[]
        for isouter in (False,True):
            ids=cp.where(outer==isouter)[0];vv=v[ids];qq=q[ids];nn=n[ids]
            plane=cp.sum(nn*qq[:,0],axis=1);margin=plane if isouter else -plane
            assert float(margin.min())>1e-8,('non_star_facet',isouter,float(margin.min()))
            cone=cp.stack([cp.cross(qq[:,0],qq[:,1]),cp.cross(qq[:,1],qq[:,2]),cp.cross(qq[:,2],qq[:,0])],1)
            cone*=cp.sign(cp.sum(cone[:,0]*qq[:,2],axis=1))[:,None,None]
            cone/=cp.linalg.norm(cone,axis=2)[:,:,None]
            self.surfaces.append(dict(ids=ids,normal=nn,plane=plane,cone=cone.reshape(-1,3)))
            self.stats.append(dict(wall='outer' if isouter else 'inner',faces=len(ids),oriented_plane_margin_min=float(margin.min()),
                vertex_radius_max=float(cp.linalg.norm(qq,axis=2).max())))

        topo=[]
        for flag in (False,True):
            ff=cp.asnumpy(faces[outer==flag]);edges=np.sort(np.concatenate((ff[:,[0,1]],ff[:,[1,2]],ff[:,[0,2]])),axis=1)
            ee,counts=np.unique(edges,axis=0,return_counts=True);nodes=np.unique(ff)
            adjacency={int(k):[] for k in nodes}
            for a,b in ee:adjacency[int(a)].append(int(b));adjacency[int(b)].append(int(a))
            reached=set();todo=[int(nodes[0])]
            while todo:
                a=todo.pop()
                if a in reached:continue
                reached.add(a);todo.extend(b for b in adjacency[a] if b not in reached)
            chi=len(nodes)-len(ee)+len(ff)
            assert len(reached)==len(nodes) and np.all(counts==2) and chi==2
            topo.append(dict(vertices=len(nodes),edges=len(ee),faces=len(ff),euler=chi,connected=True,closed_edge_incidence2=True))
        self.poly_inner_lo=self.stats[0]['oriented_plane_margin_min'];self.poly_inner_hi=self.stats[0]['vertex_radius_max']
        self.poly_outer_lo=self.stats[1]['oriented_plane_margin_min'];self.poly_outer_hi=self.stats[1]['vertex_radius_max']
        assert self.poly_outer_lo>self.poly_inner_hi+1e-6,'polyhedral radial shell separation'
        dist=cp.linalg.norm(self.geo[:,:3]-self.center,axis=1);R=float(self.geo[1,3]);radii=self.geo[:,3]
        margin=R*R+radii*radii-dist*dist


        assert bool(cp.all(margin>1e-8)),('inner_uniqueness_margin',cp.asnumpy(margin).tolist())
        bounds=dist+radii;top=bounds.max();inner_hi=top+c.metaball_softmin_tau*cp.log(cp.exp((bounds-top)/c.metaball_softmin_tau).sum())
        self.true_inner_lo=R;self.true_inner_hi=float(inner_hi)
        self.true_outer_lo=float(c.outer_half_extent-cp.linalg.norm(self.center-.5))
        self.true_outer_hi=float(cp.linalg.norm(c.outer_half_extent+cp.abs(self.center-.5)))
        assert self.true_outer_lo>self.true_inner_hi+1e-6,'true walls separation'
        derivative_lower=margin/(2*inner_hi*(inner_hi+dist))
        self.certificate=dict(center=cp.asnumpy(self.center).tolist(),polyhedral_walls=self.stats,topology=topo,
            polyhedral_shell_separation_lower=self.poly_outer_lo-self.poly_inner_hi,
            true_inner_radius_interval=[R,float(inner_hi)],true_outer_radius_interval=[self.true_outer_lo,self.true_outer_hi],
            true_shell_separation_lower=self.true_outer_lo-self.true_inner_hi,
            inner_uniqueness_ball_margins=cp.asnumpy(margin).tolist(),inner_radial_derivative_lower=float(derivative_lower.min()),
            argument='Closed connected sphere meshes and strictly outward radial face orientation imply each boundary radial projection is a covering of S2, hence degree-one homeomorphism. True outer wall is convex with center inside. At each inner zero all ball distances>=radii; R^2+Ri^2-Di^2>0 implies strictly positive radial derivative, giving a unique zero. Positive separated radial intervals make the affine radial interpolation bijective and detJ=k*(rho/r)^2>0 away from cone seams.',
            evidence_scope='Topology and radial face margins are exhaustive finite-mesh checks; true-wall uniqueness uses analytic parameter inequalities. Floating-point evaluation is not an interval-arithmetic global certificate. Sampled matrix bounds are separately labeled and are not global bounds.',
            nearest_projection='No nearest-normal projection or tubular-neighborhood assumption is used; projection is unique along rays about the verified center.')

        k_lo=(self.true_outer_lo-self.true_inner_hi)/(self.poly_outer_hi-self.poly_inner_lo)
        k_hi=(self.true_outer_hi-self.true_inner_lo)/(self.poly_outer_lo-self.poly_inner_hi)
        self.certificate['conditional_global_det_interval']=[k_lo*(self.true_inner_lo/self.poly_outer_hi)**2,k_hi*(self.true_outer_hi/self.poly_inner_lo)**2]

    def level_grad(self,x):
        c=self.c;q=(x-.5)/c.outer_half_extent
        fo=1-cp.sum(q**c.outer_superellipsoid_power,axis=-1)
        go=-c.outer_superellipsoid_power/c.outer_half_extent*q**(c.outer_superellipsoid_power-1)
        delta=x[...,None,:]-self.geo[:,:3];length=cp.linalg.norm(delta,axis=-1);dist=length-self.geo[:,3]
        low=dist.min(-1);w=cp.exp(-(dist-low[...,None])/c.metaball_softmin_tau);fi=low-c.metaball_softmin_tau*cp.log(w.sum(-1));w/=w.sum(-1)[...,None]
        gi=cp.sum(w[...,None]*delta/cp.maximum(length[...,None],1e-30),axis=-2)
        return fo,fi,go,gi

    def true_ends(self,d):
        result=[]
        for inner in (True,False):
            lo=cp.full(len(d),self.true_inner_lo if inner else self.true_outer_lo);hi=cp.full(len(d),self.true_inner_hi if inner else self.true_outer_hi)
            for _ in range(45):
                mid=(lo+hi)/2;fo,fi,_,_=self.level_grad(self.center+mid[:,None]*d)
                below=fi<0 if inner else fo>0;lo=cp.where(below,mid,lo);hi=cp.where(below,hi,mid)
            r=(lo+hi)/2;fo,fi,go,gi=self.level_grad(self.center+r[:,None]*d);g=gi if inner else go
            rd=cp.sum(g*d,1);assert bool(cp.all(rd>0 if inner else rd<0))
            deriv=-r[:,None]*g/rd[:,None]
            result.append((r,deriv))
        return result

    def mesh_ends(self,d):
        result=[]
        for surf in self.surfaces:
            picks=[];counts=[]
            for i in range(0,len(d),128):
                test=(d[i:i+128]@surf['cone'].T).reshape(-1,len(surf['ids']),3)
                hit=cp.all(test>=-2e-12,axis=2);cnt=hit.sum(1)
                assert bool(cp.all(cnt>0)),('radial_ray_uncovered',int(cp.sum(cnt==0)))
                picks.append(cp.argmax(hit,axis=1));counts.append(cnt)
            pick=cp.concatenate(picks);count=cp.concatenate(counts);n=surf['normal'][pick];den=cp.sum(n*d,1)
            r=surf['plane'][pick]/den;deriv=-r[:,None]*n/den[:,None]
            result.append((r,deriv,surf['ids'][pick],count))
        return result

    def ends(self,d):return self.mesh_ends(d),self.true_ends(d)

    def radial(self,d,r,source,target):
        a,da=source[0][:2];b,db=source[1][:2];A,dA=target[0][:2];B,dB=target[1][:2]
        k=(B-A)/(b-a);s=(r-a)/(b-a);rho=A+s*(B-A)
        tang=(1-s[:,None])*dA+s[:,None]*dB-k[:,None]*((1-s[:,None])*da+s[:,None]*db)
        tang-=d*cp.sum(tang*d,1)[:,None]
        gr=k[:,None]*d+tang/r[:,None];P=cp.eye(3)[None]-d[:,:,None]*d[:,None,:]
        J=(rho/r)[:,None,None]*P+d[:,:,None]*gr[:,None,:]
        det=k*(rho/r)**2
        return self.center+rho[:,None]*d,J,det,s

    def map(self,x,inverse=False):
        q=x-self.center;r=cp.linalg.norm(q,axis=1);d=q/r[:,None];mesh,true=self.ends(d)
        source,target=(true,mesh) if inverse else (mesh,true)
        y,J,det,s=self.radial(d,r,source,target)

        other,Jback,detback,_=self.radial(d,cp.linalg.norm(y-self.center,axis=1),target,source)
        validation=dict(inverse_point_max=float(cp.max(cp.linalg.norm(other-x,axis=1))),
            inverse_J_max=float(cp.max(cp.abs(Jback@J-cp.eye(3)))),det_formula_max=float(cp.max(cp.abs(cp.linalg.det(J)-det))),
            radial_fraction_min=float(s.min()),radial_fraction_max=float(s.max()),
            inner_ray_ties=int(cp.sum(mesh[0][3]>1)),outer_ray_ties=int(cp.sum(mesh[1][3]>1)))
        return y,(Jback if inverse else J),(detback if inverse else det),validation
