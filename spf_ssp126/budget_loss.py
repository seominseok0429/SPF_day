"""TRAIN-only rainfall allocation and temporal-dependence fine-tune; raw Euler30 outputs."""
import copy
import torch
from . import core as C
from . import extreme_loss as OLD

RECIPE=copy.deepcopy(OLD.RECIPE)
RECIPE.update(version='ssp126-budget',parent_step=24000,final_step=27000,
    training_scenarios=['ssp126'],normalizer='SSP126-only parent scalar mean/std; no new fit',
    weights=dict(month_mean=.2,daily_std=.2,budget=.9/4,tails=.9/4,frequency=.9/4,persistence=.9/4,physical=.02),
    grouped_objective='0.9*(budget+tails+frequency+persistence)/4 plus 0.2 mean,0.2 std,0.02 physical',
    budget='gridwise monthly rain mean /2mm/day; wet-day rain amount /2 and dry-day amount /0.5; average 3 terms',
    persistence='half existing month-local 2/5/10-day spell proxy, half gridwise rainfall lag1/lag7 correlation',
    temporal_correlation='demean within each TRAIN month; regularized variance floor (0.05mm/day)^2',
    data_sequence='SSP126-only homogeneous-calendar groups; 3000 additional updates',
    no_future_GT_trend_constants=True,monthly_projection=False,negative_precipitation_fix=False,
    output_postprocessing=False,independent_validation=False)

class AuxLoss(OLD.AuxLoss):
    @staticmethod
    def rain_features(pr):
        wet=torch.sigmoid((pr-1.)/.2)
        amounts=torch.stack([(pr*wet).mean(1),(pr*(1-wet)).mean(1)],1)
        z=pr-pr.mean(1,keepdim=True);correlations=[]
        for lag in (1,7):
            a=z[:,:-lag];b=z[:,lag:]
            den=((a.square().mean(1)+.05**2)*(b.square().mean(1)+.05**2)).sqrt()
            correlations.append((a*b).mean(1)/den)
        return amounts,torch.stack(correlations,1)

    def forward(self,generated,truth,condition):
        _,parts=super().forward(generated,truth,condition)
        p=self.physical(generated);g=self.physical(truth.detach());c=self.physical(condition.detach()).squeeze(2)
        amount,correlation=self.rain_features(p[:,3])
        with torch.no_grad():gt_amount,gt_correlation=self.rain_features(g[:,3])
        scales=p.new_tensor([2.,.5])[None,:,None,None]
        allocation=OLD.weighted_mse((amount-gt_amount)/scales,self.region_w)
        total=OLD.weighted_mse((p[:,3].mean(1)-c[:,3])/2.,self.global_w)
        # allocation averages two terms; keep three constituent budget terms equally weighted.
        parts['budget']=(total+2*allocation)/3
        lag=OLD.weighted_mse(correlation-gt_correlation,self.region_w)
        parts['persistence']=(parts['persistence']+lag)/2
        value=sum(RECIPE['weights'][k]*v for k,v in parts.items())
        assert torch.isfinite(value)
        return value,parts

class Objective(torch.nn.Module):
    def __init__(self,net,norm,lat,mask):
        super().__init__();self.net=net;self.aux=AuxLoss(norm,lat,mask)
    def forward(self,truth,condition,auxiliary=False,seed=0,ramp=1.):
        base=C.flow_loss(self.net,truth,condition,C.scheduler())
        if not auxiliary:return base,{'flow':base.detach()}
        generated=OLD.generated_with_tail_grad(self.net,condition,truth.shape[2],seed)
        aux,parts=self.aux(generated,truth,condition)
        return base+ramp*aux,dict(flow=base,aux_total=aux.detach(),**{k:v.detach() for k,v in parts.items()})
