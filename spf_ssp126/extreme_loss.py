"""TRAIN-month auxiliary losses on actual Euler30 generated sequences.

Not the official annual 15-index score. No target/output transformations,
historical threshold refitting, projection, clipping, or smoothing.
"""
import math
import numpy as np
import torch
from . import core as C

RECIPE = dict(training_scenarios=['ssp126'],version=1,additional_steps=3000,parent_step=21000,final_step=24000,
    lr=2e-5,global_batch=24,aux_every=8,aux_ramp_updates=200,
    sampler_steps=30,backprop_last_steps=2,
    weights=dict(month_mean=.2,daily_std=.2,tails=.1,frequency=.1,persistence=.1,physical=.02),
    mean_scales=[2.,2.,2.,2.],std_scales=[2.,2.,2.,5.],
    tail_scales=[5.,5.,10.,30.,10.],temperature_softness_K=.5,rain_softness_mm_day=.2,
    spell_lengths=[2,5,10],percentiles='within TRAIN month 0.10/0.90 temperature,0.95 rain; NOT historical official thresholds',
    auxiliary_scope='month-local differentiable proxies; no annual CDD/CWD/GSL equivalence',
    gradient='Euler30 first28 stop-gradient, final2 differentiable; truncated path gradient',
    region_weighting='AR6 land region balanced using fixed existing mask; native actual latitude cosine',
    negative_precipitation_fix=False,temperature_order_fix=False,monthly_projection=False,
    normalizer='unchanged parent TRAIN scalar mean/std',ema=False,warmup=False,lr_schedule=False)

def weighted_mse(delta,weights):
    return (delta.square()*weights).mean()

class AuxLoss(torch.nn.Module):
    def __init__(self,norm,lat,mask):
        super().__init__()
        mean=torch.tensor(norm['mean'],dtype=torch.float32);std=torch.tensor(norm['std'],dtype=torch.float32)
        mean[3]*=86400;std[3]*=86400
        self.register_buffer('mean',mean[None,:,None,None,None])
        self.register_buffer('std',std[None,:,None,None,None])
        mask=np.asarray(mask)
        if mask.ndim != 2 or mask.shape[0] != len(lat):
            raise ValueError('Latitude and static AR6 region mask mismatch')
        if not np.issubdtype(mask.dtype, np.integer) or not np.any(mask >= 0):
            raise ValueError('Static AR6 mask must contain integer land-region labels')
        cosine=np.maximum(np.cos(np.deg2rad(np.asarray(lat,dtype=np.float64))),0)
        global_w=np.broadcast_to(cosine[:,None],mask.shape).copy();global_w/=global_w.mean()
        region_w=np.zeros_like(global_w);codes=np.unique(mask[mask>=0])
        for code in codes:
            w=global_w*(mask==code);assert w.sum()>0
            region_w+=w/w.sum()/len(codes)
        region_w*=region_w.size;assert np.isclose(region_w.mean(),1)
        self.register_buffer('global_w',torch.from_numpy(global_w.T.copy()).float())
        self.register_buffer('region_w',torch.from_numpy(region_w.T.copy()).float())
        self.register_buffer('mean_scale',torch.tensor(RECIPE['mean_scales'])[None,:,None,None])
        self.register_buffer('std_scale',torch.tensor(RECIPE['std_scales'])[None,:,None,None])
        self.register_buffer('tail_scale',torch.tensor(RECIPE['tail_scales'])[None,:,None,None])

    def physical(self,x):return x.float()*self.std+self.mean

    def features(self,p):
        tx=p[:,1];tn=p[:,2];pr=p[:,3];tas=p[:,0]
        wet=torch.sigmoid((pr-1.)/.2)
        # Gates occur only inside the differentiable loss; physical p is untouched.
        sdii=(pr*wet).sum(1)/wet.sum(1).clamp_min(1e-3)
        tails=torch.stack([tx.amax(1),tn.amin(1),torch.quantile(pr,.95,dim=1),
            pr.unfold(1,5,1).sum(-1).amax(1),sdii],1)
        frequency=torch.stack([torch.sigmoid((tx-298.15)/.5).mean(1),
            torch.sigmoid((tn-293.15)/.5).mean(1),torch.sigmoid((273.15-tn)/.5).mean(1),
            torch.sigmoid((273.15-tx)/.5).mean(1),wet.mean(1),
            torch.sigmoid((pr-10.)/.2).mean(1)],1)
        # Local TRAIN-month quantiles, deliberately NOT WSDI/CSDI baseline thresholds.
        warm=torch.sigmoid((tx-torch.quantile(tx,.90,dim=1)[:,None])/.5)
        cold=torch.sigmoid((torch.quantile(tn,.10,dim=1)[:,None]-tn)/.5)
        spell=[]
        for gate in (wet,1-wet,warm,cold):
            for k in RECIPE['spell_lengths']:
                spell.append(gate.unfold(1,k,1).prod(-1).mean(1))
        return tails,frequency,torch.stack(spell,1)

    def forward(self,generated,truth,condition):
        assert generated.shape==truth.shape and generated.ndim==5 and generated.shape[1]==4
        assert generated.shape[2] in (28,30,31) and condition.shape[2]==1
        p=self.physical(generated);g=self.physical(truth.detach());c=self.physical(condition.detach()).squeeze(2)
        pf=self.features(p)
        with torch.no_grad():gf=self.features(g)
        parts=dict(month_mean=weighted_mse((p.mean(2)-c)/self.mean_scale,self.global_w),
            daily_std=weighted_mse((p.std(2,correction=0)-g.std(2,correction=0))/self.std_scale,self.region_w),
            tails=weighted_mse((pf[0]-gf[0])/self.tail_scale,self.region_w),
            frequency=weighted_mse(pf[1]-gf[1],self.region_w),
            persistence=weighted_mse(pf[2]-gf[2],self.region_w))
        violation=torch.stack([torch.relu(-p[:,3])/2.,torch.relu(p[:,2]-p[:,0])/2.,torch.relu(p[:,0]-p[:,1])/2.],1)
        parts['physical']=(violation.square()*self.global_w).mean()
        total=sum(RECIPE['weights'][k]*v for k,v in parts.items())
        assert torch.isfinite(total)
        return total,parts

def generated_with_tail_grad(net,condition,days,seed):
    assert days in (28,30,31)
    gen=torch.Generator(device=condition.device).manual_seed(seed)
    z=torch.randn((1,4,days,288,192),generator=gen,device=condition.device)
    sch=C.scheduler();sch.set_timesteps(30,stage_index=0,temporal_level=0,device=condition.device)
    mode=net.training
    try:
        net.eval()
        with torch.no_grad():
            for t in sch.timesteps[:28]:
                with torch.autocast('cuda',dtype=torch.bfloat16):v=C.predict(net,z,condition,t.expand(1).to(condition.device))
                z=sch.step(v.float(),t,z).prev_sample
        z=z.detach();net.train(mode)
        for t in sch.timesteps[28:]:
            with torch.autocast('cuda',dtype=torch.bfloat16):v=C.predict(net,z,condition,t.expand(1).to(condition.device))
            z=sch.step(v.float(),t,z).prev_sample
    finally:net.train(mode)
    assert torch.isfinite(z).all()
    return z

class Objective(torch.nn.Module):
    def __init__(self,net,norm,lat,mask):
        super().__init__();self.net=net;self.aux=AuxLoss(norm,lat,mask)
    def forward(self,truth,condition,auxiliary=False,seed=0,ramp=1.):
        base=C.flow_loss(self.net,truth,condition,C.scheduler())
        if not auxiliary:return base,{"flow":base.detach()}
        generated=generated_with_tail_grad(self.net,condition,truth.shape[2],seed)
        aux,parts=self.aux(generated,truth,condition)
        # Keep base connected so the trainer can weight flow and auxiliary
        # gradients separately without accidentally over-weighting this month.
        return base+ramp*aux,dict(flow=base,aux_total=aux.detach(),**{k:v.detach() for k,v in parts.items()})
