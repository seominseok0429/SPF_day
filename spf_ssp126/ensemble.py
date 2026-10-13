"""Equal mean of already computed member annual-index fields, never mean daily weather."""
import argparse
from pathlib import Path
import numpy as np

INDICES = ("TXx", "TNn", "SU", "TR", "FD", "ID", "WSDI", "CSDI",
           "GSL", "Rx5day", "CDD", "CWD", "R95p", "R10mm", "SDII")


def annual_index_mean(paths, expected_members=26):
    paths = [Path(path) for path in paths]
    if len(paths) != expected_members or len({path.resolve() for path in paths}) != len(paths):
        raise ValueError("Exactly the requested number of distinct member annual-index files is required.")
    totals, years = {}, None
    for path in paths:
        with np.load(path, allow_pickle=False) as fields:
            if set(fields.files) != set(INDICES) | {"years"}:
                raise ValueError("Each NPZ must contain the same official 15 annual fields plus years.")
            member_years = fields["years"]
            if years is None:
                years = member_years.copy()
                if not np.array_equal(years, np.arange(2015, 2101)):
                    raise ValueError("The comparison scope is exactly 2015..2100 (86 years).")
            elif not np.array_equal(years, member_years):
                raise ValueError("Members do not share the same calendar.")
            for name in INDICES:
                value = np.asarray(fields[name], np.float64)
                if value.shape != (86, 192, 288) or not np.isfinite(value).all():
                    raise ValueError(f"Invalid annual field: {name}")
                if name not in totals:
                    totals[name] = value.copy()
                else:
                    totals[name] += value
    return dict(years=years, **{name: total / expected_members for name, total in totals.items()})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("members", nargs="+", help="One official annual-index NPZ per member")
    parser.add_argument("--count", type=int, default=26)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    result = annual_index_mean(args.members, args.count)
    with Path(args.output).open("xb") as stream:
        np.savez(stream, **result)


if __name__ == "__main__":
    main()
