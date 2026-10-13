import copy
import io
import numpy as np
import pytest
torch = pytest.importorskip("torch")
from spf_ssp126 import extreme_loss, budget_loss
from spf_ssp126.train import check_checkpoint
from spf_ssp126.model import config, UPSTREAM_COMMIT
from spf_ssp126.sample import check_condition_provenance
from test_data import norm


@pytest.mark.parametrize("module", [extreme_loss, budget_loss])
@pytest.mark.parametrize("days", [28, 30, 31])
def test_losses_perfect_and_gradient(module, days):
    torch.set_num_threads(2)
    lat = np.linspace(-70, 70, 8)
    mask = np.broadcast_to(np.arange(8)[:, None] // 4, (8, 6)).copy()
    auxiliary = module.AuxLoss(norm(), lat, mask)
    gen = torch.Generator().manual_seed(42 + days)
    p = torch.empty(1, 4, days, 6, 8)
    p[:, 0] = 280 + torch.randn((1, days, 6, 8), generator=gen)
    p[:, 1] = p[:, 0] + 4
    p[:, 2] = p[:, 0] - 4
    p[:, 3] = torch.rand((1, days, 6, 8), generator=gen) * 15
    truth = (p - auxiliary.mean) / auxiliary.std
    condition = truth.mean(2, keepdim=True)
    before = condition.clone()
    value, _ = auxiliary(truth, truth, condition)
    assert value < 1e-8
    shifted = truth.clone()
    shifted[:, 3] += 1 / float(auxiliary.std[0, 3, 0, 0, 0])
    shifted.requires_grad_()
    loss, parts = auxiliary(shifted, truth, condition)
    loss.backward()
    assert torch.isfinite(loss) and torch.isfinite(shifted.grad).all()
    assert shifted.grad.abs().sum() > 0
    assert torch.equal(condition, before)
    if module is budget_loss:
        assert parts["budget"] > 0


def checkpoint(stage, step):
    stages = ["base", "extreme", "budget"]
    return dict(stage=stage, step=step, configuration=config(), upstream_commit=UPSTREAM_COMMIT,
                training_scenarios=["ssp126"], normalization=norm(), data_receipt_sha256="new126",
                world_size=4, rngs=[{"rank": rank} for rank in range(4)],
                optimizer={"state": {0: {"step": torch.tensor(step)}}},
                lineage=[dict(stage=name, training_scenarios=["ssp126"])
                         for name in stages[:stages.index(stage) + 1]])


def test_stage_chains_continue_only_ssp126():
    check_checkpoint(checkpoint("base", 21000), "extreme", norm(), "new126")
    check_checkpoint(checkpoint("extreme", 24000), "budget", norm(), "new126")
    check_checkpoint(checkpoint("budget", 25500), "budget", norm(), "new126")
    for bad in (dict(training_scenarios=["ssp126", "ssp245"]), dict(data_receipt_sha256="old4"),
                dict(normalization=dict(norm(), scenarios=["ssp126", "ssp245"]))):
        value = checkpoint("extreme", 24000)
        value.update(bad)
        with pytest.raises(ValueError):
            check_checkpoint(value, "budget", norm(), "new126")
    value = checkpoint("extreme", 24000)
    value["lineage"][0]["training_scenarios"] = ["ssp245"]
    with pytest.raises(ValueError):
        check_checkpoint(value, "budget", norm(), "new126")


def test_unsafe_or_oracle_conditions_are_rejected():
    check_condition_provenance({"source": "Independent model", "uses_evaluation_targets": False})
    for provenance in ({}, {"source": "GT mean", "uses_evaluation_targets": True},
                       {"source": "", "uses_evaluation_targets": False}):
        with pytest.raises(ValueError):
            check_condition_provenance(provenance)


def test_checkpoint_rng_can_use_safe_torch_loader():
    state = np.random.get_state()
    value = checkpoint("base", 1000)
    value["rngs"][0]["numpy"] = dict(kind=state[0], keys=state[1].tolist(),
                                    pos=state[2], has_gauss=state[3], cached_gaussian=state[4])
    stream = io.BytesIO()
    torch.save(value, stream)
    stream.seek(0)
    restored = torch.load(stream, weights_only=True)
    assert restored["rngs"][0]["numpy"] == value["rngs"][0]["numpy"]
