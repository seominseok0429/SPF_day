import os
import pytest
torch = pytest.importorskip("torch")
from spf_ssp126.model import setup_upstream, scheduler


@pytest.mark.skipif(not os.environ.get("SPF_SOURCE"), reason="Optional pinned-upstream integration check")
def test_original_euler30_constant_velocity_endpoint():
    setup_upstream()
    clean = torch.randn(1, 4, 2, 8, 8)
    noise = torch.randn_like(clean)
    sched = scheduler()
    sched.set_timesteps(30, stage_index=0, temporal_level=0, device="cpu")
    z = noise.clone()
    for t in sched.timesteps:
        z = sched.step(noise - clean, t, z).prev_sample
    assert torch.allclose(z, clean, atol=3e-6, rtol=3e-6)
