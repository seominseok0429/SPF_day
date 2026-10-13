"""Original native-grid single-stage SPF flow and Euler sampler; no output calibration."""
import torch
from .model import scheduler

def predict(net,z,c,t):
    b=z.shape[0]
    return net(sample=[z],condition_sample=[c],timestep_ratio=t,
        latent_size=torch.full((b,),36,device=z.device,dtype=torch.int32),
        latent_time=torch.zeros(b,device=z.device,dtype=torch.int32))[0]

def flow_loss(net,clean,condition,sch):
    idx=int(torch.randint(0,1000,()).item())
    sigma=float(sch.sigmas_per_stage[0][0][idx]);t=float(sch.timesteps_per_stage[0][0][idx])
    noise=torch.randn_like(clean)
    z=sigma*noise+(1-sigma)*clean;target=noise-clean
    with torch.autocast('cuda',dtype=torch.bfloat16):
        p=predict(net,z,condition,torch.full((len(clean),),t,device=clean.device))
        loss=(p.float()-target).square().mean()
    return loss

@torch.no_grad()
def sample(net,condition,days,seed,steps=30):
    assert days in (28,30,31) and condition.shape==(1,4,1,288,192)
    gen=torch.Generator(device=condition.device).manual_seed(seed)
    z=torch.randn((1,4,days,288,192),generator=gen,device=condition.device)
    sch=scheduler();sch.set_timesteps(steps,stage_index=0,temporal_level=0,device=condition.device)
    for t in sch.timesteps:
        with torch.autocast('cuda',dtype=torch.bfloat16):
            v=predict(net,z,condition,t.expand(1).to(condition.device))
        z=sch.step(v.float(),t,z).prev_sample
    assert torch.isfinite(z).all()
    return z
