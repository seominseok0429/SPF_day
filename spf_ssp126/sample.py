"""Generate raw members from externally supplied monthly conditions; no evaluation GT loader."""
import argparse
import json
import os
from pathlib import Path
import shutil
import numpy as np
import torch
from .core import sample
from .data import SCENARIO
from .model import build_model, config, setup_upstream, UPSTREAM_COMMIT


def check_condition_provenance(provenance):
    if provenance.get("uses_evaluation_targets") is not False:
        raise ValueError("Evaluation-target monthly means are forbidden in this release.")
    if not isinstance(provenance.get("source"), str) or not provenance["source"].strip():
        raise ValueError("Document the independent monthly-condition source.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--conditions", required=True,
                        help="NPZ: monthly, days, year, month, lat, lon")
    parser.add_argument("--condition-provenance", required=True, help="JSON: source, uses_evaluation_targets=false")
    parser.add_argument("--output", required=True)
    parser.add_argument("--members", type=int, default=26)
    parser.add_argument("--seed", type=int, default=61001)
    parser.add_argument("--upstream", default=None)
    args = parser.parse_args()
    if os.environ.get("CUDA_VISIBLE_DEVICES") not in ("0", "1", "2", "3"):
        parser.error("Choose one physical GPU 0, 1, 2, or 3 using CUDA_VISIBLE_DEVICES.")
    if args.members < 1:
        parser.error("--members must be positive.")
    setup_upstream(args.upstream)
    provenance = json.loads(Path(args.condition_provenance).read_text())
    check_condition_provenance(provenance)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    norm = checkpoint["normalization"]
    lineage = checkpoint.get("lineage", [])
    if checkpoint.get("training_scenarios") != [SCENARIO] or norm.get("scenarios") != [SCENARIO]:
        raise ValueError("Only new SSP126-only checkpoints/normalization are allowed.")
    if norm.get("fit_scope") != "SSP126 train only" or not lineage:
        raise ValueError("Missing SSP126-only train-normalization lineage.")
    if any(stage.get("training_scenarios") != [SCENARIO] for stage in lineage):
        raise ValueError("Reject mixed-scenario parent chain.")
    if checkpoint.get("upstream_commit") != UPSTREAM_COMMIT or checkpoint["configuration"] != config():
        raise ValueError("Wrong SPF configuration/upstream.")
    net = build_model().cuda().eval()
    net.load_state_dict(checkpoint["model"], strict=True)
    mean = np.asarray(norm["mean"], np.float32)[None, :, None, None]
    std = np.asarray(norm["std"], np.float32)[None, :, None, None]
    with np.load(args.conditions, allow_pickle=False) as conditions:
        monthly = conditions["monthly"]
        days = conditions["days"]
        years, months = conditions["year"], conditions["month"]
        n = len(days)
        if monthly.shape != (n, 4, 192, 288) or len(years) != n or len(months) != n:
            raise ValueError("Conditions must use 4 physical fields in (month, channel, lat, lon) order.")
        if not np.isfinite(monthly).all() or not np.isin(days, [28, 30, 31]).all():
            raise ValueError("Invalid physical monthly conditions/calendar.")
        identities = [(int(y), int(m)) for y, m in zip(years, months)]
        if len(set(identities)) != n or identities != sorted(identities):
            raise ValueError("Condition months must be unique and chronologically ordered.")
        if not np.isin(months, np.arange(1, 13)).all():
            raise ValueError("Invalid month number.")
        if not np.array_equal(conditions["lat"], np.asarray(checkpoint["coordinates"]["lat"])):
            raise ValueError("Condition latitude coordinates differ from the training grid.")
        if not np.array_equal(conditions["lon"], np.asarray(checkpoint["coordinates"]["lon"])):
            raise ValueError("Condition longitude coordinates differ from the training grid.")
        for (year, month), length in zip(identities, days):
            expected = 28 if month == 2 else 30 if month in (4, 6, 9, 11) else 31
            if int(length) != expected and not ((year, month, int(length)) == (2100, 12, 30)):
                raise ValueError("Month length does not match the noleap calendar.")
        output = Path(args.output)
        if not output.parent.is_dir():
            raise ValueError("Create the output parent directory first.")
        required = int(days.sum()) * 4 * 192 * 288 * 4 * args.members
        if shutil.disk_usage(output.parent).free < required + 20 * 2 ** 30:
            raise RuntimeError("Insufficient output space; no automatic deletion.")
        output.mkdir(exist_ok=False)
        for member in range(args.members):
            directory = output / f"member_{member:02d}"
            directory.mkdir()
            for i, (year, month) in enumerate(identities):
                if shutil.disk_usage(output).free < 20 * 2 ** 30:
                    raise RuntimeError("Output storage reserve reached.")
                physical = monthly[i:i + 1]
                normalized = (physical - mean) / std
                condition = torch.from_numpy(np.ascontiguousarray(
                    normalized.transpose(1, 0, 3, 2)[None])).cuda()
                z = sample(net, condition, int(days[i]),
                           args.seed + member * 1000000 + year * 100 + month, steps=30)
                raw = z.float().cpu().numpy()[0].transpose(1, 0, 3, 2) * std + mean
                # Only invert SSP126 train normalization. No projection, GT affine,
                # precipitation clipping, field mixing, or mean/std matching.
                np.save(directory / f"{year:04d}_{month:02d}.npy", raw)
            (directory / "complete.json").write_text(json.dumps(
                dict(status="complete", member=member, months=identities,
                     days=days.tolist(), postprocessing=False), indent=2))
        (output / "complete.json").write_text(json.dumps(
            dict(status="complete", members=args.members, seed=args.seed,
                 condition_provenance=provenance, postprocessing=False), indent=2))


if __name__ == "__main__":
    main()
