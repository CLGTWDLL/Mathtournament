"""Replace an adjacent coverage/source pair with a shared measurement site.

All proposals use public histories. Rejected proposals preserve the baseline
policy's random stream. Geometry certificates never depend on posterior mass.
"""
import copy
import numpy as np
from q3_backbone import BackbonePolicy,covered,union_cover,shift_with_certificate
from q3_policy import Proposal


class JointSitePolicy(BackbonePolicy):
    def __init__(self,mode='joint_site',seed=9101,**kwargs):
        super().__init__(mode='backbone_fixed',seed=seed,**kwargs)
        if mode not in ('joint_site','joint_site_fixed','joint_site_guard'):raise ValueError('Unknown joint-site mode')
        self.joint_mode=mode;self.joint_rng=np.random.default_rng(seed+73001)
        self.transaction=None;self.joint_trials=0;self.joint_accepted=0
        self.joint_history=[];self.joint_metadata=None

    def peek_local(self,i,start,exit_point,channel):
        state=(self.position,self.exit_point,self.channel,copy.deepcopy(self.rng.bit_generator.state))
        try:
            self.position=np.asarray(start);self.exit_point=exit_point;self.channel=channel
            return self.local_to_exit(i)
        finally:
            self.position,self.exit_point,self.channel,rng=state
            self.rng.bit_generator.state=rng

    def samples_cost(self,i,proposal,start,end,channel,g,radius,noise):
        old=self.exit_point;self.exit_point=end
        try:
            cover=self.cover(i);q=proposal.position
            travel=np.linalg.norm(q-start)/5
            if proposal.action=='measure':
                masks,near=cover.measurement_masks(q,g,radius,noise)
                return travel+5+(i+1!=channel)+5*near+cover.costs(q,masks,g)
            success=np.linalg.norm(g-q,axis=1)<=20
            masks=np.repeat(cover.remaining_after_clear(q)[None,:],len(g),axis=0)
            masks[success]=False
            return travel+3+2*success+cover.costs(q,masks,g)
        finally:self.exit_point=old

    def batch_fee_and_receiver(self,channel):
        todo=[int(i+1) for i in self.unknown()]
        cost=0
        while todo:
            c=channel if channel in todo else todo[0]
            cost+=5+(c!=channel);channel=c;todo.remove(c)
        return cost,channel

    def propose_joint(self):
        if not self.numeric_ok or not len(self.unknown()) or len(self.planned)<2:return None
        pair=self.planned[:2]
        if {a[0] for a in pair}!={'source','station'}:return None
        source_first=pair[0][0]=='source'
        source=pair[0] if source_first else pair[1]
        station=pair[1] if source_first else pair[0]
        i=int(source[1]);b=station[2]
        if station[1]!=0 or self.local_measurements[i]>=5:return None
        if any(o.kind=='near' for o in self.history[i]):return None
        end=self.planned[2][2].copy() if len(self.planned)>2 else None
        start=self.position.copy();channel=self.channel
        first_fee,batch_end=self.batch_fee_and_receiver(channel)
        baseline=self.peek_local(i,start if source_first else b,b if source_first else end,
                                 channel if source_first else batch_end)
        p=self.bank.spatial[i]
        ids=self.joint_rng.choice(len(p.points),self.particles,p=p.weights)
        g=p.points[ids];radius=self.joint_rng.uniform(p.lower[ids],p.upper[ids])
        noise=self.joint_rng.uniform(-1,1,len(g))
        if source_first:
            old_cost=self.samples_cost(i,baseline,start,b,channel,g,radius,noise)
            receiver=i+1 if baseline.action=='measure' else channel
            fee,_=self.batch_fee_and_receiver(receiver)
            old_cost+=fee+(np.linalg.norm(b-end)/5 if end is not None else 0)
        else:
            old_cost=self.samples_cost(i,baseline,b,end,batch_end,g,radius,noise)
            old_cost+=np.linalg.norm(start-b)/5+first_fee
        needed=self.unresolved_area() & ~union_cover(self.nodes[1:])
        candidates=[b.copy()]
        if self.joint_mode!='joint_site_fixed':
            mean=p.weights@p.points
            v=mean-start;v/=max(np.linalg.norm(v),1e-9);normal=np.array([-v[1],v[0]])
            targets=[baseline.position,mean,start]
            targets += [mean+s*normal for s in (-150.,-60.,60.,150.)]
            targets += [b+np.array(d) for d in ((200,0),(-200,0),(0,200),(0,-200))]
            candidates += [shift_with_certificate(b,target,needed) for target in targets]
        unknown=self.unknown()
        reception_before=sum(self.bank.predict(int(c+1),b)['receive'] for c in unknown)
        best=None;seen=set();self.joint_trials+=1
        for q in candidates:
            q=np.round(q,6);key=tuple(q)
            if key in seen or self.repeated(i,q):continue
            seen.add(key)
            if np.any(needed & ~covered(q)):continue
            reception_after=sum(self.bank.predict(int(c+1),q)['receive'] for c in unknown)
            if self.joint_mode=='joint_site_guard' and reception_after<reception_before-1e-8:continue
            proposal=Proposal('measure',i+1,q,0.,'localize')
            cost=self.samples_cost(i,proposal,start,end,batch_end,g,radius,noise)+first_fee
            gain=old_cost-cost
            mean_gain=float(gain.mean());se=float(gain.std(ddof=1)/np.sqrt(len(gain)))
            # A paired screening rule, not a simultaneous confidence guarantee.
            score=mean_gain-2*se
            if score>10 and (best is None or score>best['score']):
                best=dict(channel=i+1,position=q,exit=end,score=score,mean_gain=mean_gain,se=se,
                          old_expected=float(old_cost.mean()),new_expected=float(cost.mean()),
                          old_station=b.copy(),source_first=source_first,
                          expected_receptions_before=float(reception_before),
                          expected_receptions_after=float(reception_after))
        return best

    def emit(self,proposal):
        result=super().emit(proposal)
        result['joint_transaction']=self.joint_metadata
        return result

    def next_action(self):
        if self.pending is not None:raise RuntimeError('Accept feedback first')
        if self.complete():return None
        if self.batch is not None:
            action=self.batch_action()
            if action is not None:return action
        if self.transaction is not None:
            t=self.transaction;self.transaction=None
            self.active_source=t['channel']-1;self.exit_point=t['exit']
            # The batch has already discharged the replaced coverage station.
            # This measurement is of a known source, at exactly the same site.
            return self.emit(Proposal('measure',t['channel'],t['position'],t['new_expected'],'localize'))
        if self.nodes is None:self.initialize_nodes()
        if self.active_source is not None and not self.cleared[self.active_source]:
            return super().next_action()
        self.active_source=None;self.joint_metadata=None
        self.optimize_nodes();self.planned=self.insertion_plan(self.nodes)
        trial=self.propose_joint()
        if trial is None:return super().next_action()
        self.nodes[0]=trial['position'].copy()
        assert not np.any(self.unresolved_area() & ~union_cover(self.nodes))
        self.joint_accepted+=1;self.transaction=trial
        self.joint_metadata={k:(v.tolist() if isinstance(v,np.ndarray) else v) for k,v in trial.items()}
        self.joint_history.append(self.joint_metadata)
        self.batch=trial['position'].copy();self.batch_reason='backbone_scan'
        return self.batch_action()
