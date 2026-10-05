"""Autonomous discovery, localization and clearance using public feedback.

Policies: sequential (finish each discovery, resume fixed scan), discover_first,
joint (reselect source after each action, opportunistic scans at actual stops).
Optimization uses posterior scenarios, never environment truth. Completion uses
positive/clear feedback and geometric/count certificates, never small probability.
"""
from dataclasses import dataclass
import numpy as np

from q3_belief import BeliefBank, Observation, possible_cells, EPS
from q3_environment import fixed_stations
from q2_analytic import RayPrior


# The 25 m lattice remains the discovery/backbone certificate grid.  Clearance
# uses a separate 28 m lattice below, so changing the fallback search does not
# silently weaken or otherwise alter the coverage-station proof.
axis = np.arange(-1787.5,1800,25.)
_x,_y = np.meshgrid(axis,axis)
GRID = np.column_stack((_x.ravel(),_y.ravel()))

CLEAR_CELL_SIZE = 28.
CLEAR_CELL_HALF = CLEAR_CELL_SIZE/2
_clear_cells_per_axis = int(np.ceil(3600/CLEAR_CELL_SIZE))
_clear_axis = (np.arange(_clear_cells_per_axis)-(_clear_cells_per_axis-1)/2)*CLEAR_CELL_SIZE
_clear_x,_clear_y = np.meshgrid(_clear_axis,_clear_axis)
CLEAR_GRID = np.column_stack((_clear_x.ravel(),_clear_y.ravel()))
STATIONS = fixed_stations()


@dataclass
class Proposal:
    action: str
    channel: int
    position: np.ndarray
    expected_s: float
    reason: str


class Cover:
    def __init__(self,history):
        self.c = CLEAR_GRID[possible_cells(CLEAR_GRID,CLEAR_CELL_HALF,history)]
        if not len(self.c):
            raise ValueError('Confirmed source lost its continuous geometric support')
        self.distance = np.linalg.norm(self.c[:,None,:]-self.c[None,:,:],axis=2)
        self.lo,self.hi = np.full(len(self.c),1000.),np.full(len(self.c),1500.)
        for o in history:
            delta=np.abs(self.c-[o.x,o.y])
            if o.kind == 'direction':
                self.lo=np.maximum(self.lo,np.linalg.norm(np.maximum(delta-CLEAR_CELL_HALF,0),axis=1))
            elif o.kind == 'no_signal':
                self.hi=np.minimum(self.hi,np.linalg.norm(delta+CLEAR_CELL_HALF,axis=1))

    def remaining_after_clear(self,q):
        return np.linalg.norm(np.abs(self.c-q)+CLEAR_CELL_HALF,axis=1) > 20

    def costs(self,start,active,g):
        active=active.copy(); n=len(g)
        cost=np.zeros(n)
        alive=active.any(axis=1)
        distances=np.repeat(np.linalg.norm(self.c-start,axis=1)[None,:],n,axis=0)
        for _ in range(len(self.c)):
            ids=np.flatnonzero(alive)
            if not len(ids):break
            if not active[ids].any(axis=1).all():
                raise ArithmeticError('Scenario escaped the continuous clearance cover')
            choices=np.argmin(np.where(active[ids],distances[ids],np.inf),axis=1)
            cost[ids]+=distances[ids,choices]/5+3
            success=np.linalg.norm(g[ids]-self.c[choices],axis=1) <= 20
            cost[ids[success]]+=2
            active[ids,choices]=False
            distances[ids]=self.distance[choices]
            alive[ids[success]]=False
        if alive.any():raise ArithmeticError('Clearance cover exhausted')
        return cost

    def measurement_masks(self,q,g,radius,noise):
        delta=g-q; d=np.linalg.norm(delta,axis=1)
        near=d<=5; direction=(d>5)&(d<=radius); negative=d>radius
        mask=np.ones((len(g),len(self.c)),bool)
        rel=self.c-q
        dmin=np.linalg.norm(np.maximum(np.abs(rel)-CLEAR_CELL_HALF,0),axis=1)
        dmax=np.linalg.norm(np.abs(rel)+CLEAR_CELL_HALF,axis=1)
        ids=np.flatnonzero(direction)
        if len(ids):
            angle=np.deg2rad(np.round(np.rad2deg(np.arctan2(delta[ids,1],delta[ids,0]))+noise[ids],2))
            for theta,sign in [(angle-EPS,1),(angle+EPS,-1)]:
                normals=np.column_stack((-sign*np.sin(theta),sign*np.cos(theta)))
                mask[ids]&=normals@rel.T+CLEAR_CELL_HALF*np.abs(normals).sum(axis=1)[:,None]>=-1e-9
            mask[ids]&=(dmax>5)&(np.maximum(self.lo,dmin)<=self.hi+1e-9)
        if negative.any():
            mask[negative]&=self.lo<=np.minimum(self.hi,dmax)+1e-9
        mask[near]=False
        return mask,near


class AutonomousPolicy:
    def __init__(self,mode='joint',seed=9101,particles=64,scan_scale=1.,extra_scan_budget=100):
        if mode not in ('joint','sequential','discover_first'):
            raise ValueError('Unknown policy')
        self.mode,self.rng,self.particles=mode,np.random.default_rng(seed),particles
        self.scan_scale,self.scan_budget=scan_scale,extra_scan_budget
        self.bank=BeliefBank()
        self.history=[[] for _ in range(20)]
        self.confirmed=np.zeros(20,bool); self.cleared=np.zeros(20,bool)
        self.absent=np.zeros(20,bool); self.station_seen=np.zeros((20,8),bool)
        self.absence_reason={}
        self.local_measurements=np.zeros(20,int); self.free_clears=np.zeros(20,int)
        self.position=np.zeros(2);self.channel=1
        self.focus=None;self.pending=None;self.steps=0
        self.numeric_ok=True;self.numeric_failures=[]
        self.covers={};self.extra_scans=0
        self.frontier_scans=0
        self.tail_batch=None
        self.last_measure_channel=None;self.last_target=None;self.target_switches=0

    def unknown(self):
        return np.flatnonzero(~(self.confirmed|self.absent))

    def known(self):
        return np.flatnonzero(self.confirmed & ~self.cleared)

    def complete(self):
        return bool(np.all(self.cleared|self.absent))

    def cover(self,i):
        key=(i,len(self.history[i]))
        if key not in self.covers:
            self.covers={k:v for k,v in self.covers.items() if k[0]!=i}
            self.covers[key]=Cover(self.history[i])
        return self.covers[key]

    def repeated(self,i,q):
        return any(o.kind in ('direction','near','no_signal') and
                   np.linalg.norm(np.array([o.x,o.y])-q)<.1 for o in self.history[i])

    def emit(self,proposal):
        q=np.round(np.asarray(proposal.position,float),6)
        self.pending=Proposal(proposal.action,int(proposal.channel),q,proposal.expected_s,proposal.reason)
        return dict(action=proposal.action,channel=int(proposal.channel),position=q.tolist(),
                    reason=proposal.reason,estimated_local_finish_s=float(proposal.expected_s))

    def safe_clear(self,i):
        cover=self.cover(i)
        q=cover.c[np.argmin(np.linalg.norm(cover.c-self.position,axis=1))]
        return Proposal('clear',i+1,q,np.linalg.norm(q-self.position)/5+5,'finite_cover')

    def local_proposal(self,i):
        history=self.history[i]
        near=next((o for o in reversed(history) if o.kind=='near'),None)
        if near is not None:
            q=np.array([near.x,near.y])
            return Proposal('clear',i+1,q,np.linalg.norm(q-self.position)/5+5,'near_clear')
        if not self.numeric_ok:
            return self.safe_clear(i)
        post=self.bank.spatial[i]
        mean=post.weights@post.points
        for q in (self.position,mean):
            if len(post.centers) and np.max(np.linalg.norm(np.abs(post.centers-q)+post.half[:,None],axis=1)) <= 19.999:
                return Proposal('clear',i+1,q,np.linalg.norm(q-self.position)/5+5,'certified_clear')
        cover=self.cover(i)
        ids=self.rng.choice(len(post.points),size=self.particles,p=post.weights)
        g=post.points[ids]
        radius=self.rng.uniform(post.lower[ids],post.upper[ids])
        noise=self.rng.uniform(-1,1,len(g))
        active=np.ones((len(g),len(cover.c)),bool)
        nearest=cover.c[np.argmin(np.linalg.norm(cover.c-self.position,axis=1))]
        clear_points=[nearest]
        if self.free_clears[i]<2:
            clear_points.extend([mean,self.position])
        best=None;best_samples=None
        for q in clear_points:
            success=np.linalg.norm(g-q,axis=1)<=20
            if not success.any() and np.linalg.norm(q-nearest)>.1:continue
            mask=active & cover.remaining_after_clear(q)
            mask[success]=False
            costs=cover.costs(q,mask,g)+np.linalg.norm(q-self.position)/5+3+2*success
            value=float(costs.mean())
            if best is None or value<best.expected_s:
                best=Proposal('clear',i+1,q,value,'posterior_clear' if np.linalg.norm(q-nearest)>.1 else 'finite_cover')
                best_samples=costs
        if self.local_measurements[i]>=5 or self.steps>=500:
            return best
        direction=mean-self.position
        direction/=max(np.linalg.norm(direction),1e-9)
        normal=np.array([-direction[1],direction[0]])
        projection=(g-mean)@direction
        anchors=[mean]
        for a in (.25,.75):
            target=np.quantile(projection,a)
            nearest_ids=np.argsort(np.abs(projection-target))[:max(4,len(g)//8)]
            anchors.append(g[nearest_ids].mean(axis=0))
        candidates=[anchor+b*normal for anchor in anchors for b in (-30.,0.,30.)]
        candidates.extend(self.position+f*(mean-self.position)+b*normal for f,b in ((.35,40),(.35,-40),(.65,40),(.65,-40)))
        first=next(o for o in history if o.kind=='direction')
        if self.local_measurements[i]==0:
            try:
                uv,_,_=RayPrior(dict(first=[first.x,first.y],bearing=first.angle)).select(.25)
                theta=np.deg2rad(first.angle); rotation=np.array([[np.cos(theta),-np.sin(theta)],[np.sin(theta),np.cos(theta)]])
                candidates.extend(np.array([first.x,first.y])+rotation@np.array([uv[0],sign*uv[1]]) for sign in (-1,1))
            except ValueError:
                pass
        for q in candidates:
            q=np.round(q,6)
            if self.repeated(i,q):continue
            mask,is_near=cover.measurement_masks(q,g,radius,noise)
            costs=cover.costs(q,mask,g)+np.linalg.norm(q-self.position)/5+5+(i+1!=self.channel)+5*is_near
            gain=best_samples-costs
            # One-step posterior scenario evaluation, no future truth inputs.
            if gain.mean()-2*gain.std(ddof=1)/np.sqrt(len(g))>1.:
                best=Proposal('measure',i+1,q,float(costs.mean()),'localize')
                best_samples=costs
        return best

    def scan_at_stop(self):
        unknown=self.unknown()
        if not self.numeric_ok or self.extra_scans>=self.scan_budget or not len(unknown):return None
        best=None;best_net=0.
        remaining_sites=np.flatnonzero(np.any(~self.station_seen[unknown,1:],axis=0))+1
        route=0.;position=self.position.copy()
        sites=list(remaining_sites)
        while sites:
            j=min(sites,key=lambda k:np.linalg.norm(STATIONS[k]-position))
            route+=np.linalg.norm(STATIONS[j]-position);position=STATIONS[j];sites.remove(j)
        for i in unknown:
            if self.repeated(i,self.position):continue
            predicted=self.bank.predict(i+1,self.position)['receive']
            remaining=int((~self.station_seen[i,1:]).sum())
            # Heuristic saved future discovery service, sharing route expense
            # over unresolved channels. Not an exact value-of-information.
            saved=self.scan_scale*predicted*(6*remaining+route/5/len(unknown))
            cost=5+int(i+1!=self.channel)
            net=saved-cost
            if net>best_net:
                best_net=net
                best=Proposal('measure',i+1,self.position.copy(),cost,'opportunistic_scan')
        return best

    def discovery_proposal(self):
        unknown=self.unknown()
        if not len(unknown):return None
        if self.mode!='joint' or not self.numeric_ok:
            for j,q in enumerate(STATIONS):
                sequence=range(20) if j%2==0 else range(19,-1,-1)
                for i in sequence:
                    if i in unknown and not self.station_seen[i,j]:
                        return Proposal('measure',i+1,q,np.linalg.norm(q-self.position)/5+5+(i+1!=self.channel),'coverage_scan')
        best=None;best_score=np.inf
        for j,q in enumerate(STATIONS):
            todo=[i for i in unknown if not self.station_seen[i,j]]
            if not todo:continue
            probabilities=[self.bank.predict(i+1,q)['receive'] for i in todo]
            score=(np.linalg.norm(q-self.position)/5+6*len(todo))/(sum(probabilities)+len(todo)/7)
            if score<best_score:
                best_score=score
                i=max(todo,key=lambda k:(k+1==self.channel,self.bank.predict(k+1,q)['receive']))
                best=Proposal('measure',i+1,q,score,'coverage_scan')
        if best is not None and self.confirmed.sum()>=10 and self.bank.exists[unknown].sum()<=.5:
            # A conservative common reception point resolves every unknown
            # channel in one batch. Use it only if closer than the regular next
            # site, with little remaining posterior source probability.
            corners=[]
            signs=np.array([[-1,-1],[-1,1],[1,-1],[1,1]])
            for i in unknown:
                p=self.bank.spatial[i]
                corners.append((p.centers[:,None,:]+p.half[:,None,None]*signs).reshape(-1,2))
            corners=np.vstack(corners)
            center=(corners.min(axis=0)+corners.max(axis=0))/2
            delta=center-corners
            if np.max(np.linalg.norm(delta,axis=1))<=999.5:
                v=self.position-center;distance=np.linalg.norm(v)
                if distance>1e-9:
                    v/=distance
                    projection=delta@v
                    upper=-projection+np.sqrt(np.maximum(0,projection**2+999.5**2-np.sum(delta**2,axis=1)))
                    center=center+min(distance,float(upper.min()))*v
                if np.linalg.norm(center-self.position)+5 < np.linalg.norm(best.position-self.position):
                    self.tail_batch=np.round(center,6)
                    i=next((i for i in unknown if i+1==self.channel),unknown[0])
                    best=Proposal('measure',i+1,self.tail_batch,np.linalg.norm(center-self.position)/5+6*len(unknown),'posterior_certification')
        return best

    def next_action(self):
        if self.pending is not None:raise RuntimeError('Accept current feedback first')
        if self.complete():return None
        if self.tail_batch is not None:
            todo=[i for i in self.unknown() if not self.repeated(i,self.tail_batch)]
            if todo and self.numeric_ok:
                i=next((i for i in todo if i+1==self.channel),todo[0])
                return self.emit(Proposal('measure',i+1,self.tail_batch,6.,'posterior_certification'))
            self.tail_batch=None
        known=self.known()
        if self.mode=='discover_first' and len(self.unknown()):
            return self.emit(self.discovery_proposal())
        if self.mode=='joint':
            # Take an immediately guaranteed clearance before optional scanning.
            for i in known:
                if self.numeric_ok:
                    p=self.bank.spatial[i]
                    if len(p.centers) and np.max(np.linalg.norm(np.abs(p.centers-self.position)+p.half[:,None],axis=1))<=19.999:
                        return self.emit(Proposal('clear',i+1,self.position.copy(),5.,'certified_clear'))
            scan=self.scan_at_stop()
            if scan is not None:return self.emit(scan)
        if len(known):
            if self.mode in ('sequential','discover_first'):
                if self.focus not in known:
                    self.focus=min(known,key=lambda i:np.linalg.norm(self.cover(i).c-self.position,axis=1).min())
                proposal=self.local_proposal(self.focus)
            else:
                # Restrict detailed scenario evaluation to three nearest jobs.
                candidates=sorted(known,key=lambda i:np.linalg.norm(self.cover(i).c-self.position,axis=1).min())[:3]
                proposal=min((self.local_proposal(i) for i in candidates),key=lambda p:p.expected_s)
            return self.emit(proposal)
        proposal=self.discovery_proposal()
        if proposal is None:raise RuntimeError('No action without a completion certificate')
        return self.emit(proposal)

    def accept_feedback(self,result):
        if self.pending is None:raise RuntimeError('No pending action')
        if result.get('accepted') is not True:raise ValueError('Unaccepted command must not update state')
        p=self.pending;i=p.channel-1;q=p.position
        kind=result['kind'];angle=result.get('angle')
        valid=('direction','near','no_signal') if p.action=='measure' else ('success','no_target_in_range')
        if kind not in valid:raise ValueError('Feedback does not match command')
        if kind=='direction' and (angle is None or not np.isfinite(angle)):raise ValueError('Missing bearing')
        o=Observation(kind,*q,angle)
        self.history[i].append(o)
        self.position=q.copy();self.steps+=1
        if p.action=='measure':
            self.channel=p.channel
            if self.confirmed[i]:self.local_measurements[i]+=1
            self.station_seen[i]|=np.linalg.norm(STATIONS-q,axis=1)<.01
            if p.reason=='opportunistic_scan':self.extra_scans+=1
            if p.reason=='posterior_certification':self.frontier_scans+=1
        else:
            if p.reason=='posterior_clear':self.free_clears[i]+=1
        if kind in ('direction','near','success'):self.confirmed[i]=True
        if kind=='success':self.cleared[i]=True
        if self.confirmed.sum()>16:raise ValueError('Observed more than sixteen sources')
        if self.numeric_ok:
            try:
                self.bank.update(p.channel,q,kind,angle)
            except (ArithmeticError,ValueError) as exc:
                self.numeric_ok=False
                self.numeric_failures.append(dict(step=self.steps,error=str(exc)))
        for c in self.unknown():
            geometric=self.numeric_ok and self.bank.spatial[c].certified_empty
            seven=bool(self.station_seen[c,1:].all())
            if geometric or seven or self.confirmed.sum()==16:
                self.absent[c]=True
                self.absence_reason[int(c+1)]=('geometry_empty' if geometric else
                                              'seven_negative_stations' if seven else 'count_16')
        if p.reason in ('localize','posterior_clear','finite_cover','near_clear','certified_clear'):
            if self.last_target is not None and self.last_target!=p.channel:self.target_switches+=1
            self.last_target=p.channel
        self.pending=None
