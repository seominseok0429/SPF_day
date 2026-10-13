"""SSP126-only records, normalization and loaders. No evaluation data are opened."""
import hashlib
import json
from pathlib import Path
import numpy as np

SCENARIO = "ssp126"
VARIABLES = ("tas", "tasmax", "tasmin", "pr")
UNITS = ("K", "K", "K", "kg m-2 s-1")
NMONTHS = 1031
NDAYS = 31359


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(16 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def calendar_rows():
    rows, offset = [], 0
    for year in range(2015, 2101):
        for month in range(1, 13):
            if (year, month) == (2100, 12):
                continue
            days = 28 if month == 2 else 30 if month in (4, 6, 9, 11) else 31
            rows.append(dict(scenario=SCENARIO, year=year, month=month,
                             days=days, offset=offset, monthly_index=len(rows)))
            offset += days
    if len(rows) != NMONTHS or offset != NDAYS:
        raise RuntimeError("Invalid complete-month calendar.")
    return rows


def groups(rows, epoch, world=4, seed=22201):
    """No duplicate/padded months; redistribute a short tail to avoid empty ranks."""
    if rows != calendar_rows() or world != 4:
        raise ValueError("This recipe requires the SSP126 calendar and four DDP ranks.")
    rng = np.random.default_rng(seed + epoch)
    result = []
    for days in (28, 30, 31):
        ids = np.array([i for i, row in enumerate(rows) if row["days"] == days])
        rng.shuffle(ids)
        chunks = [ids[k:k + 24].tolist() for k in range(0, len(ids), 24)]
        if len(chunks[-1]) < world:
            merged = chunks[-2] + chunks[-1]
            chunks[-2:] = [merged[:-world], merged[-world:]]
        result.extend(chunks)
    rng.shuffle(result)
    flat = [i for group in result for i in group]
    if sorted(flat) != list(range(NMONTHS)) or len(flat) != len(set(flat)):
        raise RuntimeError("Month coverage error.")
    if not all(world <= len(group) <= 24 for group in result):
        raise RuntimeError("Empty-rank or oversized batch.")
    return result


def validate_metadata(receipt, norm, rows):
    if receipt.get("status") != "complete" or receipt.get("training_scenarios") != [SCENARIO]:
        raise ValueError("Only newly prepared SSP126-only data are allowed.")
    if norm.get("scenarios") != [SCENARIO] or norm.get("fit_scope") != "SSP126 train only":
        raise ValueError("Reject normalization fitted on other scenarios/evaluation data.")
    if norm.get("variables") != list(VARIABLES) or norm.get("units") != list(UNITS):
        raise ValueError("Wrong variable order or units.")
    if rows != calendar_rows():
        raise ValueError("Mixed-scenario or inconsistent training records.")
    mean, std = np.asarray(norm["mean"]), np.asarray(norm["std"])
    if mean.shape != (4,) or std.shape != (4,) or not np.isfinite(mean).all():
        raise ValueError("Invalid train normalization.")
    if not np.isfinite(std).all() or not (std > 0).all():
        raise ValueError("Nonpositive/nonfinite train standard deviation.")


def read_metadata(directory):
    directory = Path(directory)
    receipt = json.loads((directory / "complete.json").read_text())
    norm = json.loads((directory / "normalization.json").read_text())
    rows = json.loads((directory / "records.json").read_text())
    validate_metadata(receipt, norm, rows)
    for name in ("normalization.json", "records.json", "coords.npz", "region_mask.npy"):
        if sha256(directory / name) != receipt["hashes"][name]:
            raise ValueError(f"Prepared metadata changed: {name}")
    return receipt, norm, rows


class Bank:
    def __init__(self, directory):
        self.directory = Path(directory)
        self.receipt, self.norm, self.rows = read_metadata(directory)
        self.receipt_sha256 = sha256(self.directory / "complete.json")
        self.daily = np.load(self.directory / "ssp126_daily.npy", mmap_mode="r", allow_pickle=False)
        self.monthly = np.load(self.directory / "ssp126_monthly.npy", mmap_mode="r", allow_pickle=False)
        if self.daily.shape != (NDAYS, 4, 192, 288) or self.monthly.shape != (NMONTHS, 4, 192, 288):
            raise ValueError("Unexpected native-grid training data shape.")
        for name in ("ssp126_daily.npy", "ssp126_monthly.npy"):
            path = self.directory / name
            identity = self.receipt["identities"][name]
            if path.stat().st_size != identity["bytes"] or path.stat().st_mtime_ns != identity["mtime_ns"]:
                raise ValueError(f"Prepared training file changed: {name}")
        with np.load(self.directory / "coords.npz", allow_pickle=False) as coordinates:
            self.lat, self.lon = coordinates["lat"].copy(), coordinates["lon"].copy()
        self.mask = np.load(self.directory / "region_mask.npy", allow_pickle=False)
        self.mean = np.asarray(self.norm["mean"], np.float32)[None, :, None, None]
        self.std = np.asarray(self.norm["std"], np.float32)[None, :, None, None]

    def item(self, index, device):
        import torch
        row = self.rows[index]
        daily = np.asarray(self.daily[row["offset"]:row["offset"] + row["days"]])
        month = np.asarray(self.monthly[row["monthly_index"]])
        y = (daily - self.mean) / self.std
        c = (month[None] - self.mean) / self.std
        y = torch.from_numpy(np.ascontiguousarray(y.transpose(1, 0, 3, 2)[None])).to(device)
        c = torch.from_numpy(np.ascontiguousarray(c.transpose(1, 0, 3, 2)[None])).to(device)
        return y, c
