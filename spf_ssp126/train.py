"""Four-GPU SSP126-only training. Explicit commands only; no automatic evaluation."""
import argparse
import contextlib
import datetime
import json
import os
from pathlib import Path
import random
import shutil
import time
import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from . import core, extreme_loss, budget_loss
from .data import Bank, SCENARIO, groups, sha256
from .model import build_model, config, scheduler, setup_upstream, UPSTREAM_COMMIT

STAGES = {"base": (0, 21000, 1e-4), "extreme": (21000, 24000, 2e-5),
          "budget": (24000, 27000, 2e-5)}


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def capture_rng(rank):
    n = np.random.get_state()
    return dict(rank=rank, python=random.getstate(), torch=torch.get_rng_state(),
                cuda=torch.cuda.get_rng_state(),
                numpy=dict(kind=n[0], keys=n[1].tolist(), pos=n[2],
                           has_gauss=n[3], cached_gaussian=n[4]))


def restore_rng(state):
    random.setstate(state["python"])
    torch.set_rng_state(state["torch"].cpu())
    torch.cuda.set_rng_state(state["cuda"].cpu())
    n = state["numpy"]
    np.random.set_state((n["kind"], np.asarray(n["keys"], np.uint32),
                         n["pos"], n["has_gauss"], n["cached_gaussian"]))


def check_checkpoint(checkpoint, stage, norm, data_sha):
    start, end, _ = STAGES[stage]
    if checkpoint.get("training_scenarios") != [SCENARIO]:
        raise ValueError("Reject old/mixed-scenario checkpoints; SSP126-only lineage required.")
    if checkpoint.get("normalization") != norm or checkpoint.get("data_receipt_sha256") != data_sha:
        raise ValueError("Checkpoint normalization/data must match SSP126-only preparation.")
    if checkpoint.get("configuration") != config() or checkpoint.get("upstream_commit") != UPSTREAM_COMMIT:
        raise ValueError("Model configuration/upstream mismatch.")
    step = checkpoint["step"]
    if not start <= step < end:
        raise ValueError(f"Checkpoint step must be in [{start}, {end}).")
    expected_parent = {"extreme": "base", "budget": "extreme"}.get(stage)
    expected_stage = expected_parent if step == start else stage
    if checkpoint["stage"] != expected_stage:
        raise ValueError("Wrong training-stage lineage.")
    lineage = checkpoint.get("lineage", [])
    expected_lineage = {"base": ["base"], "extreme": ["base", "extreme"],
                        "budget": ["base", "extreme", "budget"]}[expected_stage]
    if [entry.get("stage") for entry in lineage] != expected_lineage:
        raise ValueError("Incomplete/unexpected parent-stage lineage.")
    if any(entry.get("training_scenarios") != [SCENARIO] for entry in lineage):
        raise ValueError("Other-scenario training occurred in the parent chain.")
    if len(checkpoint.get("rngs", [])) != 4 or checkpoint.get("world_size") != 4:
        raise ValueError("Four-rank optimizer/RNG continuation required.")
    steps = {int(value["step"]) for value in checkpoint["optimizer"]["state"].values()
             if "step" in value}
    if steps != {step}:
        raise ValueError("Optimizer update counters do not match the checkpoint.")


class BaseObjective(torch.nn.Module):
    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(self, truth, condition, auxiliary=False, seed=0, ramp=1.0):
        loss = core.flow_loss(self.net, truth, condition, scheduler())
        return loss, {"flow": loss.detach()}


def ensure_space(path, reserve_gib):
    if shutil.disk_usage(path).free < reserve_gib * 2 ** 30:
        raise RuntimeError("Storage reserve reached; no automatic deletion.")


def save_checkpoint(directory, net, optimizer, bank, stage, step, epoch, group,
                    rank, lineage, args):
    states = [None] * 4
    dist.all_gather_object(states, capture_rng(rank))
    if rank == 0:
        ensure_space(directory, args.reserve_gib)
        path = directory / f"step_{step:06d}.pt"
        if path.exists() or path.with_suffix(".partial").exists():
            raise FileExistsError(path)
        value = dict(model=net.state_dict(), optimizer=optimizer.state_dict(), rngs=states,
                     stage=stage, step=step, epoch=epoch, group=group, world_size=4,
                     lineage=lineage, training_scenarios=[SCENARIO],
                     normalization=bank.norm, data_receipt_sha256=bank.receipt_sha256,
                     coordinates={"lat": bank.lat.tolist(), "lon": bank.lon.tolist()},
                     configuration=config(), upstream_commit=UPSTREAM_COMMIT,
                     postprocessing=False, seed=22201, sampler_steps=30)
        torch.save(value, path.with_suffix(".partial"))
        path.with_suffix(".partial").replace(path)
        path.with_suffix(".READY").write_text(json.dumps(
            dict(step=step, sha256=sha256(path), training_scenarios=[SCENARIO]), indent=2))
    dist.barrier()


def train(args):
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "0,1,2,3":
        raise RuntimeError("Set CUDA_VISIBLE_DEVICES=0,1,2,3; GPU4 is not used.")
    if int(os.environ.get("WORLD_SIZE", "0")) != 4:
        raise RuntimeError("Launch using torchrun --nproc_per_node=4.")
    setup_upstream(args.upstream)
    bank = Bank(args.data)
    rank = int(os.environ["RANK"])
    device = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(device)
    torch.set_num_threads(4)
    start, end, lr = STAGES[args.stage]
    if args.stage != "base" and not args.resume:
        raise ValueError("Additional training requires the immediately preceding SSP126 checkpoint.")
    checkpoint = None
    if args.resume:
        resume = Path(args.resume)
        ready = json.loads(resume.with_suffix(".READY").read_text())
        if sha256(resume) != ready["sha256"]:
            raise ValueError("Checkpoint/READY hash mismatch.")
        checkpoint = torch.load(resume, map_location="cpu", weights_only=True)
        check_checkpoint(checkpoint, args.stage, bank.norm, bank.receipt_sha256)
    directory = Path(args.output).resolve()
    if not directory.parent.is_dir():
        raise ValueError("Create the output parent directory first.")
    ensure_space(directory.parent, args.reserve_gib)
    dist.init_process_group("nccl", timeout=datetime.timedelta(minutes=45))
    try:
        if rank == 0:
            directory.mkdir(exist_ok=False)
            (directory / "started.json").write_text(json.dumps(
                dict(stage=args.stage, training_scenarios=[SCENARIO], target_step=end,
                     resume_sha256=sha256(args.resume) if args.resume else None), indent=2))
        dist.barrier()
        seed_all(22201)
        net = build_model().cuda().train()
        optimizer = torch.optim.AdamW(net.parameters(), lr=lr, betas=(0.9, 0.999),
                                     eps=1e-8, weight_decay=0)
        step, epoch, skip_group, lineage = start, 1, 0, []
        if checkpoint:
            net.load_state_dict(checkpoint["model"], strict=True)
            optimizer.load_state_dict(checkpoint["optimizer"])
            for group in optimizer.param_groups:
                group["lr"] = lr
            step, epoch, skip_group = checkpoint["step"], checkpoint["epoch"], checkpoint["group"]
            lineage = checkpoint["lineage"]
            # Original rainfall-budget refinement begins a new shuffled epoch sequence.
            if args.stage == "budget" and step == start:
                epoch, skip_group = 1, 0
        if not lineage or lineage[-1]["stage"] != args.stage:
            lineage = lineage + [dict(stage=args.stage, training_scenarios=[SCENARIO],
                                      start_step=start, end_step=end)]
        if args.stage == "base":
            objective = BaseObjective(net)
        else:
            module = extreme_loss if args.stage == "extreme" else budget_loss
            objective = module.Objective(net, bank.norm, bank.lat, bank.mask).cuda()
        ddp = DDP(objective, device_ids=[device], find_unused_parameters=True, broadcast_buffers=False)
        if checkpoint:
            restore_rng(checkpoint["rngs"][rank])
        else:
            seed_all(22201 + rank)
        del checkpoint
        timer = time.monotonic()
        while step < end:
            grouped = groups(bank.rows, epoch, seed=61001 if args.stage == "budget" else 22201)
            for group_index, ids in enumerate(grouped, 1):
                if group_index <= skip_group:
                    continue
                ensure_space(directory, args.reserve_gib)
                auxiliary = args.stage != "base" and (step - start + 1) % 8 == 0
                ramp = min(1.0, (step - start + 1) / 200)
                local = ids[rank::4]
                optimizer.zero_grad(set_to_none=True)
                flow_sum = 0.0
                for j, index in enumerate(local):
                    truth, condition = bank.item(index, torch.device("cuda", device))
                    original_condition = condition.clone()
                    last = j == len(local) - 1
                    context = contextlib.nullcontext() if last else ddp.no_sync()
                    do_aux = auxiliary and last
                    with context:
                        value, parts = ddp(truth, condition, auxiliary=do_aux,
                                           seed=95241 + (step + 1) * 10000 + index, ramp=ramp)
                        scaled = value * 4 / len(ids)
                        if do_aux:
                            scaled = scaled + (1 - 4 / len(ids)) * (value - parts["flow"])
                        if not torch.isfinite(scaled):
                            raise FloatingPointError("Nonfinite loss.")
                        scaled.backward()
                    if not torch.equal(condition, original_condition):
                        raise RuntimeError("Monthly condition must not be modified.")
                    flow_sum += float(parts["flow"].detach())
                grad = net.cond_pos_embed.proj.weight.grad
                if grad is None or not torch.isfinite(grad).all():
                    raise RuntimeError("Missing/nonfinite condition-encoder gradient.")
                norm = torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0, error_if_nonfinite=True)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                step += 1
                total = torch.tensor(flow_sum, device=torch.device("cuda", device), dtype=torch.float64)
                dist.all_reduce(total)
                if rank == 0:
                    row = dict(stage=args.stage, step=step, target=end, epoch=epoch, group=group_index,
                               flow_loss=float(total) / len(ids), gradient_norm=float(norm),
                               training_scenarios=[SCENARIO], auxiliary=auxiliary,
                               elapsed_seconds=time.monotonic() - timer)
                    (directory / "progress.json").write_text(json.dumps(row, indent=2))
                    if step % 10 == 0:
                        print(json.dumps(row), flush=True)
                interval = 1000 if args.stage == "base" else 500
                if step % interval == 0 or step == end:
                    save_checkpoint(directory, net, optimizer, bank, args.stage, step,
                                    epoch, group_index, rank, lineage, args)
                if step == end:
                    break
            epoch += 1
            skip_group = 0
        if rank == 0:
            (directory / "complete.json").write_text(json.dumps(
                dict(status="complete", stage=args.stage, step=step,
                     training_scenarios=[SCENARIO], checkpoint=f"step_{step:06d}.pt"), indent=2))
    finally:
        dist.destroy_process_group()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", required=True, choices=list(STAGES))
    parser.add_argument("--data", required=True)
    parser.add_argument("--output", required=True, help="New directory; existing runs are never overwritten")
    parser.add_argument("--upstream", default=None, help="Or set SPF_SOURCE")
    parser.add_argument("--resume", default=None)
    parser.add_argument("--reserve-gib", type=int, default=20)
    args = parser.parse_args()
    if args.reserve_gib < 20:
        parser.error("At least 20 GiB storage reserve is required.")
    train(args)


if __name__ == "__main__":
    main()
