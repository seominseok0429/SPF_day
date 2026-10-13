"""Explicitly prepare one SSP126 training NetCDF; never discover/load other scenarios."""
import argparse
import json
from pathlib import Path
import shutil
import numpy as np
from .data import SCENARIO, VARIABLES, UNITS, NDAYS, NMONTHS, calendar_rows, sha256


def combine_moments(n, mean, m2, values):
    flat = np.moveaxis(values, 1, 0).reshape(4, -1)
    count = flat.shape[1]
    batch_mean = flat.mean(1)
    batch_m2 = ((flat - batch_mean[:, None]) ** 2).sum(1)
    total = n + count
    delta = batch_mean - mean
    return total, mean + delta * count / total, m2 + batch_m2 + delta ** 2 * n * count / total


def verify_source_scenario(experiment, filename):
    if experiment:
        if str(experiment).lower() != SCENARIO:
            raise ValueError("The training NetCDF must be SSP126.")
    elif not Path(filename).name.lower().endswith("train_ssp126.nc"):
        raise ValueError("No experiment_id: require an explicit *_train_ssp126.nc filename.")


def prepare(source, region_mask, output, reserve_gib=20):
    import netCDF4
    source, region_mask, output = Path(source).resolve(), Path(region_mask).resolve(), Path(output).resolve()
    if output.exists():
        raise FileExistsError("Preparation never overwrites an existing output directory.")
    if not output.parent.is_dir():
        raise ValueError("Create the output parent directory first.")
    required = NDAYS * 4 * 192 * 288 * 4 + NMONTHS * 4 * 192 * 288 * 4
    if shutil.disk_usage(output.parent).free < required + reserve_gib * 2 ** 30:
        raise RuntimeError("Insufficient free space; do not delete other data automatically.")
    with netCDF4.Dataset(source) as ds, np.load(region_mask, allow_pickle=False) as geometry:
        verify_source_scenario(getattr(ds, "experiment_id", ""), source.name)
        if ds["time"].calendar != "noleap":
            raise ValueError("This recipe expects the original noleap calendar.")
        lat, lon = np.asarray(ds["lat"][:]), np.asarray(ds["lon"][:])
        mask = geometry["mask"]
        if mask.shape != (192, 288) or not np.issubdtype(mask.dtype, np.integer):
            raise ValueError("Provide an integer AR6 mask (lat, lon), with -1 for ocean.")
        if not np.array_equal(geometry["lat"], lat) or not np.array_equal(geometry["lon"], lon):
            raise ValueError("Static region-mask coordinates must exactly match the training grid.")
        if len(np.unique(mask[mask >= 0])) != 46:
            raise ValueError("The original recipe requires all 46 AR6 land regions.")
        if lat.shape != (192,) or lon.shape != (288,):
            raise ValueError("Native grid must be 192 latitude x 288 longitude.")
        dates = netCDF4.num2date(ds["time"][:], ds["time"].units, calendar="noleap")
        rows = calendar_rows()
        for row in rows:
            lo, hi = row["offset"], row["offset"] + row["days"]
            if [(d.year, d.month, d.day) for d in dates[lo:hi]] != [
                (row["year"], row["month"], day) for day in range(1, row["days"] + 1)
            ]:
                raise ValueError("Missing/misaligned SSP126 training dates.")
        prefixes = {}
        for name, unit in zip(VARIABLES, UNITS):
            variable = ds[name]
            if variable.units != unit:
                raise ValueError(f"Wrong units for {name}: {variable.units}")
            if variable.dimensions[-3:] != ("time", "lat", "lon"):
                raise ValueError("Expected (..., time, lat, lon) dimensions.")
            leading = variable.dimensions[:-3]
            if any(len(ds.dimensions[d]) != 1 for d in leading):
                raise ValueError("Choose exactly one ensemble member before preparation.")
            prefixes[name] = (0,) * len(leading)
        output.mkdir()
        (output / "started.json").write_text(json.dumps({"status": "started", "training_scenarios": [SCENARIO]}))
        daily = np.lib.format.open_memmap(output / "ssp126_daily.npy", mode="w+", dtype="float32",
                                         shape=(NDAYS, 4, 192, 288))
        monthly = np.lib.format.open_memmap(output / "ssp126_monthly.npy", mode="w+", dtype="float32",
                                           shape=(NMONTHS, 4, 192, 288))
        n, mean, m2 = 0, np.zeros(4, np.float64), np.zeros(4, np.float64)
        for row in rows:
            lo, hi = row["offset"], row["offset"] + row["days"]
            fields = []
            for name in VARIABLES:
                value = ds[name][prefixes[name] + (slice(lo, hi), slice(None), slice(None))]
                if np.ma.isMaskedArray(value) and np.ma.getmaskarray(value).any():
                    raise ValueError("Masked training values are not supported.")
                fields.append(np.asarray(value, np.float32))
            values = np.stack(fields, axis=1)
            if not np.isfinite(values).all() or (values[:, 3] < 0).any():
                raise ValueError("Nonfinite/negative-rain SSP126 training data.")
            daily[lo:hi] = values
            monthly[row["monthly_index"]] = values.mean(0, dtype=np.float64).astype(np.float32)
            n, mean, m2 = combine_moments(n, mean, m2, values.astype(np.float64))
        daily.flush()
        monthly.flush()
        del daily, monthly
        std = np.sqrt(m2 / n)
        if not (std > 0).all():
            raise ValueError("Degenerate SSP126 normalization.")
        norm = dict(mean=mean.tolist(), std=std.tolist(), count_per_channel=n,
                    scenarios=[SCENARIO], fit_scope="SSP126 train only",
                    variables=list(VARIABLES), units=list(UNITS))
        (output / "normalization.json").write_text(json.dumps(norm, indent=2))
        (output / "records.json").write_text(json.dumps(rows, indent=2))
        np.savez(output / "coords.npz", lat=lat, lon=lon)
        np.save(output / "region_mask.npy", mask)
    names = ("ssp126_daily.npy", "ssp126_monthly.npy", "normalization.json",
             "records.json", "coords.npz", "region_mask.npy")
    hashes = {name: sha256(output / name) for name in names}
    identities = {name: dict(bytes=(output / name).stat().st_size,
                            mtime_ns=(output / name).stat().st_mtime_ns)
                  for name in names[:2]}
    receipt = dict(status="complete", training_scenarios=[SCENARIO], months=NMONTHS, days=NDAYS,
                   hashes=hashes, identities=identities, source_sha256=sha256(source),
                   region_geometry_sha256=sha256(region_mask),
                   conditioning="Monthly mean of SSP126 TRAIN targets only")
    (output / "complete.json").write_text(json.dumps(receipt, indent=2))
    print(json.dumps({"status": "complete", "directory": str(output), "training_scenarios": [SCENARIO]}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ssp126", required=True, help="Single SSP126 training NetCDF")
    parser.add_argument("--region-mask", required=True, help="Static NPZ: mask, lat, lon; no target fields")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    prepare(args.ssp126, args.region_mask, args.output)


if __name__ == "__main__":
    main()
