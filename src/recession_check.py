"""Held-out G-RUN recession times versus the simulated store contributions to recovery."""
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr

from train import load, recession

ROOT = Path(__file__).resolve().parents[1]
RUN = os.environ.get("RUN", "mcssmc")


def main():
    basins, dates, x, _ = load()
    yr = torch.tensor(dates.year.values)[:, None].expand_as(x["p"])
    q = x["q_grun"]
    mask = (yr >= 2010) & (yr <= 2019) & torch.isfinite(q)
    month = torch.tensor(dates.month.values)
    rate, n = recession(torch.nan_to_num(q), x["p"], month, mask)
    rec = pd.DataFrame({"recession_rate": rate.numpy(), "n_months": n.numpy()}, index=basins)
    # Slower recession = less negative rate; express as e-folding time in months.
    rec["efold_months"] = -1 / rec.recession_rate.where(rec.recession_rate < 0)

    nn = pd.read_csv(ROOT / "data/processed/nonnested_basins.csv", dtype={"basin": str}).basin.tolist()
    ev = pd.concat([pd.read_csv(ROOT / f"runs/recovery_events_{RUN}_s{s}.csv", dtype={"basin": str}).assign(seed=s) for s in range(10)])
    lag = ev[ev.target == "runoff"].groupby(["seed", "basin"]).lag.mean().groupby("basin").mean()
    rs = pd.read_csv(ROOT / f"runs/state_reset_{RUN}.csv", dtype={"basin": str})
    rs = rs[rs.target == "runoff"]
    fast = rs.groupby(["seed", "basin"])["short_fast gw"].mean().groupby("basin").mean()
    slow = rs.groupby(["seed", "basin"])["short_slow gw"].mean().groupby("basin").mean()
    d = rec.join(pd.DataFrame({"runoff_lag": lag, "short_fast": fast, "short_slow": slow})).loc[nn]
    d = d[d.n_months >= 10].dropna(subset=["efold_months"])
    d.to_csv(ROOT / f"runs/recession_check_{RUN}.csv", float_format="%.4g")
    print(f"{len(d)} non-overlapping catchments with >= 10 recession months in 2010-2019; "
          f"median e-folding time {d.efold_months.median():.1f} months")
    for c in ("runoff_lag", "short_fast", "short_slow"):
        r, p = spearmanr(d.efold_months, d[c])
        print(f"  observed recession e-folding time vs simulated {c}: rho = {r:.2f}, p = {p:.3f}")


if __name__ == "__main__":
    main()
