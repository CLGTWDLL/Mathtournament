"""Coverage backbone with posterior-localized source insertions.

No hidden environment fields enter this module. The route is a feasible
completion skeleton, not a claimed lower bound or a global optimal route.
"""
import numpy as np

from q3_policy import AutonomousPolicy, Cover, Proposal, GRID


# Whole 25 m squares intersecting the target disk: a conservative outer cover.
CELLS = GRID[np.linalg.norm(np.maximum(np.abs(GRID)-12.5, 0), axis=1) <= 1800]
SIGNS = np.array([[-1,-1],[-1,1],[1,-1],[1,1]])


def covered(q):
    return np.linalg.norm(np.abs(CELLS-q)+12.5, axis=1) <= 999.9


def length(start, points):
    if not len(points): return 0.
    return float(np.linalg.norm(np.diff(np.vstack((start,points)),axis=0),axis=1).sum())


def ring(phase=0., sign=1):
    angles=phase+sign*np.arange(6)*np.pi/3
    return [1250*np.array([np.cos(t),np.sin(t)]) for t in angles]


def union_cover(nodes):
    result=np.zeros(len(CELLS),bool)
    for q in nodes: result |= covered(q)
    return result


def shift_with_certificate(origin, target, required):
    """Farthest feasible point on a specified segment, with whole-cell proof."""
    if not required.any(): return np.asarray(target,float).copy()
    corners=(CELLS[required,None,:]+12.5*SIGNS).reshape(-1,2)
    d=origin-corners
    if np.max(np.linalg.norm(d,axis=1)) > 999.900001:
        raise ArithmeticError('Starting backbone is not a coverage certificate')
    v=np.asarray(target)-origin; travel=np.linalg.norm(v)
    if travel < 1e-9: return origin.copy()
    v/=travel; projection=d@v
    upper=-projection+np.sqrt(np.maximum(0,projection**2+999.9**2-np.sum(d*d,axis=1)))
    amount=max(0.,min(travel,float(upper.min())-1e-4))
    return origin+amount*v


class ExitCover(Cover):
    """The same finite clearance continuation, including its actual exit leg."""
    def __init__(self,history,owner):
        super().__init__(history); self.owner=owner

    def costs(self,start,active,g):
        end=self.owner.exit_point
        if end is None: return super().costs(start,active,g)
        active=active.copy(); n=len(g); alive=active.any(axis=1)
        costs=np.where(alive,0.,np.linalg.norm(start-end)/5)
        distances=np.repeat(np.linalg.norm(self.c-start,axis=1)[None,:],n,axis=0)
        for _ in range(len(self.c)):
            ids=np.flatnonzero(alive)
            if not len(ids): break
            if not active[ids].any(axis=1).all():
                raise ArithmeticError('Scenario escaped the clearance cover')
            choices=np.argmin(np.where(active[ids],distances[ids],np.inf),axis=1)
            costs[ids]+=distances[ids,choices]/5+3
            success=np.linalg.norm(g[ids]-self.c[choices],axis=1)<=20
            costs[ids[success]]+=2+np.linalg.norm(self.c[choices[success]]-end,axis=1)/5
            active[ids,choices]=False; distances[ids]=self.distance[choices]
            alive[ids[success]]=False
        if alive.any(): raise ArithmeticError('Clearance cover exhausted')
        return costs


class BackbonePolicy(AutonomousPolicy):
    def __init__(self,mode='backbone',seed=9101,particles=48,scan_scale=.5,**kwargs):
        super().__init__(mode='joint',seed=seed,particles=particles,scan_scale=scan_scale)
        if mode not in ('backbone','backbone_fixed','backbone_sweep'):
            raise ValueError('Unknown backbone mode')
        self.variant=mode
        self.remaining=np.ones((20,len(CELLS)),bool)
        self.nodes=None; self.batch=np.zeros(2); self.batch_reason='backbone_scan'
        self.active_source=None; self.exit_point=None; self.planned=[]
        self.deleted_nodes=0; self.shifted_nodes=0; self.replaced_distance=0.
        self.batch_count=0; self.last_optional_stop=None

    def cover(self,i):
        key=(i,len(self.history[i]))
        if key not in self.covers:
            self.covers={k:v for k,v in self.covers.items() if k[0]!=i}
            self.covers[key]=ExitCover(self.history[i],self)
        return self.covers[key]

    def unresolved_area(self):
        unknown=self.unknown()
        return self.remaining[unknown].any(axis=0) if len(unknown) else np.zeros(len(CELLS),bool)

    def means(self):
        result={}
        for i in self.known():
            if self.numeric_ok:
                p=self.bank.spatial[i]; result[int(i)]=p.weights@p.points
            else: result[int(i)]=self.cover(i).c.mean(axis=0)
        return result

    def insertion_plan(self,nodes,means=None):
        """Open route; each known source inserted once, by least extra distance."""
        tasks=[('station',j,np.asarray(q)) for j,q in enumerate(nodes)]
        jobs=dict(self.means() if means is None else means)
        while jobs:
            best=None
            for i,q in jobs.items():
                for k in range(len(tasks)+1):
                    a=self.position if k==0 else tasks[k-1][2]
                    delta=np.linalg.norm(q-a)
                    if k<len(tasks): delta+=np.linalg.norm(q-tasks[k][2])-np.linalg.norm(a-tasks[k][2])
                    if best is None or delta<best[0]: best=(delta,i,k,q)
            _,i,k,q=best; tasks.insert(k,('source',i,q)); del jobs[i]
        return tasks

    def initialize_nodes(self):
        # Rotation/order choice uses only posterior means after the origin batch.
        candidates=[ring()]
        if self.variant=='backbone':
            candidates=[ring(t,sign) for t in np.arange(12)*np.pi/6 for sign in (1,-1)]
        means=self.means()
        self.nodes=min(candidates,key=lambda nodes:length(self.position,[a[2] for a in self.insertion_plan(nodes,means)]))
        assert not np.any(self.unresolved_area() & ~union_cover(self.nodes))

    def optimize_nodes(self):
        if not len(self.unknown()): self.nodes=[]; return
        needed=self.unresolved_area()
        if np.any(needed & ~union_cover(self.nodes)):
            raise ArithmeticError('Lost remaining-area coverage invariant')
        if self.variant!='backbone': return
        before=length(self.position,self.nodes)
        # Delete only a node whose entire outstanding coverage is replaceable.
        while self.nodes:
            choices=[]
            for j in range(len(self.nodes)):
                rest=self.nodes[:j]+self.nodes[j+1:]
                if not np.any(needed & ~union_cover(rest)):
                    choices.append((length(self.position,rest),j,rest))
            if not choices: break
            _,_,self.nodes=min(choices,key=lambda x:x[0]); self.deleted_nodes+=1
        means=self.means()
        # One conservative coordinate pass; choose by the entire proxy route,
        # not by the distance to a source alone. The required region is exact
        # for the current whole-cell outer approximation.
        for j in range(len(self.nodes)):
            others=self.nodes[:j]+self.nodes[j+1:]
            required=needed & ~union_cover(others)
            origin=self.nodes[j].copy()
            prev=self.position if j==0 else self.nodes[j-1]
            targets=[prev]+list(means.values())
            if j+1<len(self.nodes): targets.append((prev+self.nodes[j+1])/2)
            best=origin; best_length=length(self.position,[a[2] for a in self.insertion_plan(self.nodes,means)])
            for target in targets:
                q=shift_with_certificate(origin,target,required)
                trial=self.nodes.copy();trial[j]=q
                value=length(self.position,[a[2] for a in self.insertion_plan(trial,means)])
                if value+1<best_length:
                    best,best_length=q,value
            if np.linalg.norm(best-origin)>1:
                self.nodes[j]=best;self.shifted_nodes+=1
        self.replaced_distance+=max(0.,before-length(self.position,self.nodes))
        assert not np.any(needed & ~union_cover(self.nodes))

    def emit(self,proposal):
        result=super().emit(proposal)
        result['backbone']=[q.tolist() for q in (self.nodes or [])]
        result['exit_point']=None if self.exit_point is None else self.exit_point.tolist()
        result['planned_tasks']=[dict(kind=k,index=int(i),position=q.tolist()) for k,i,q in self.planned]
        return result

    def batch_action(self):
        todo=[i for i in self.unknown() if not self.repeated(i,self.batch)]
        if not todo: self.batch=None;return None
        i=next((i for i in todo if i+1==self.channel),todo[0])
        return self.emit(Proposal('measure',i+1,self.batch,5+(i+1!=self.channel)+np.linalg.norm(self.batch-self.position)/5,self.batch_reason))

    def optional_batch(self):
        if self.variant!='backbone' or self.extra_scans>=200 or not len(self.unknown()): return False
        if self.last_optional_stop is not None and np.linalg.norm(self.position-self.last_optional_stop)<150: return False
        area=self.unresolved_area(); gain=int(np.sum(area & covered(self.position)))
        # Cheap screening only: at least 1% of the target's outer cells is new,
        # or the stop can finish an entire unresolved channel certificate.
        finish=any(not np.any(self.remaining[i] & ~covered(self.position)) for i in self.unknown())
        if gain<.01*len(CELLS) and not finish: return False
        if all(self.repeated(i,self.position) for i in self.unknown()): return False
        self.batch=self.position.copy();self.batch_reason='opportunistic_scan'
        self.last_optional_stop=self.position.copy();self.batch_count+=1
        return True

    def local_to_exit(self,i):
        proposal=self.local_proposal(i)
        # These early returns do not invoke ExitCover.costs.
        if self.exit_point is not None and (proposal.reason in ('near_clear','certified_clear') or not self.numeric_ok):
            proposal.expected_s+=np.linalg.norm(proposal.position-self.exit_point)/5
        return proposal

    def next_action(self):
        if self.pending is not None: raise RuntimeError('Accept feedback before replanning')
        if self.complete(): return None
        if self.batch is not None:
            result=self.batch_action()
            if result is not None:return result
        if self.nodes is None:self.initialize_nodes()
        if self.optional_batch():return self.batch_action()
        if self.active_source is not None and not self.cleared[self.active_source]:
            return self.emit(self.local_to_exit(self.active_source))
        self.active_source=None
        self.optimize_nodes()
        if self.variant=='backbone_sweep' and len(self.unknown()):
            self.planned=[('station',j,q) for j,q in enumerate(self.nodes)]
        else:self.planned=self.insertion_plan(self.nodes)
        if not self.planned: raise ArithmeticError('No route without completion proof')
        kind,i,q=self.planned[0]
        if kind=='station':
            # Coverage is committed only after the complete unknown-channel
            # batch finishes; keep the station in the invariant until then.
            self.batch=q.copy();self.batch_reason='backbone_scan'
            return self.batch_action()
        self.active_source=i
        self.exit_point=self.planned[1][2].copy() if len(self.planned)>1 else None
        return self.emit(self.local_to_exit(i))

    def accept_feedback(self,result):
        p=self.pending
        super().accept_feedback(result)
        if result['kind']=='no_signal':self.remaining[p.channel-1]&=~covered(p.position)
        # Independent whole-cell certificate works even after numerical failure.
        for c in self.unknown():
            if not self.remaining[c].any():
                self.absent[c]=True;self.absence_reason[int(c+1)]='geometry_empty'
        if self.nodes is not None:
            # Once all unknown channels were measured here, this station's
            # service is complete even if some returned positive bearings.
            self.nodes=[q for q in self.nodes if not (
                np.linalg.norm(q-self.position)<.01 and
                all(self.repeated(i,q) for i in self.unknown()))]
