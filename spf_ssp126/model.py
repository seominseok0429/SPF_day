"""Adapter to the pinned Apache-2.0 SPF implementation (modified forward gradient context).

No pretrained checkpoint is imported. Training begins from random weights.
"""
import contextlib
import hashlib
import inspect
import os
from pathlib import Path
import subprocess
import sys
import textwrap

UPSTREAM_URL = "https://github.com/stanfordmlgroup/spf"
UPSTREAM_COMMIT = "a7624ae25b47b474d5a31ca22b3ac6e6907795dd"
MODEL_SHA256 = "71d4a7af3eb32343e12d7d41a225a753dbb45e7da04d39faabc02e54ef8dcde6"


def setup_upstream(path=None):
    selected = path or os.environ.get("SPF_SOURCE")
    if not selected:
        raise RuntimeError("Set SPF_SOURCE to the pinned upstream SPF checkout.")
    root = Path(selected).expanduser().resolve(strict=True)
    commit = subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()
    if commit != UPSTREAM_COMMIT:
        raise RuntimeError(f"SPF checkout must be pinned to {UPSTREAM_COMMIT}")
    relative = "emulator/models/spatiotemporal_pyramid_flow/model.py"
    if hashlib.sha256((root / relative).read_bytes()).hexdigest() != MODEL_SHA256:
        raise RuntimeError("The upstream model has changed.")
    dirty = subprocess.check_output(
        ["git", "-C", str(root), "status", "--porcelain", "--untracked-files=no"], text=True
    )
    if dirty.strip():
        raise RuntimeError("Use a clean upstream SPF checkout.")
    sys.path.insert(0, str(root))
    return root


def config():
    # Identical 4-field / single-stage model geometry to the original experiment.
    return dict(height=288, width=192, cond_height=288, cond_width=192,
                num_attention_heads=16, attention_head_dim=72, in_channels=4,
                cond_in_channels=4, out_channels=4, depth=4, num_classes=0,
                patch_size=8, cond_patch_size=16, attention_bias=True,
                cross_attention_dim=1152, dropout=0.0, pos_embed_type="sincos",
                temp_pos_embed_type="rope", pos_embed_max_size=288,
                use_grad_checkpointing=True, add_temp_pos_embed=True, use_flash_attn=True)


def build_model():
    from emulator.models.spatiotemporal_pyramid_flow import model as upstream
    if hashlib.sha256(Path(upstream.__file__).read_bytes()).hexdigest() != MODEL_SHA256:
        raise RuntimeError("Imported model is not the pinned upstream implementation.")
    source = textwrap.dedent(inspect.getsource(upstream.SpatiotemporalPyramidFlow.forward))
    if source.count("with torch.no_grad():") != 1:
        raise RuntimeError("Unsupported upstream forward gradient context.")
    scope = dict(vars(upstream), contextlib=contextlib)
    # Original experiment's explicit gradient fix; no upstream file is modified.
    exec(compile(source.replace("with torch.no_grad():", "with contextlib.nullcontext():"),
                 __file__, "exec"), scope)

    class TrainableSPF(upstream.SpatiotemporalPyramidFlow):
        forward = scope["forward"]

    return TrainableSPF(**config())


def scheduler():
    from emulator.schedulers.temporal_flow import TemporalFlowMatchEulerDiscreteScheduler
    return TemporalFlowMatchEulerDiscreteScheduler(
        num_train_timesteps=1000, num_stages=1, stage_range=[0, 1], ns=[[]], shift=1.0
    )
