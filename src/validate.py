"""Simulated recovery lags compared with gauged streamflow and GRACE storage."""
import os
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm, spearmanr

from recovery import runs, simulate, std_index
from spatial_test import groups

ROOT = Path(__file__).resolve().parents[1]
RUN = os.environ.get("RUN", "mcssmc")
NSEED = int(os.environ.get("NSEED", 10))


def obs_runoff(basins, index):
    out = pd.DataFrame(np.nan, index=index, columns=basins)
    for b in basins:
        d = pd.read_csv(ROOT / f"data/hydromlyr/1_data/{b}/streamflow_raw.txt", parse_dates=["date"])
        m = d.set_index("date").q.resample("MS").mean()
        out[b] = m.reindex(index)
    return out


def own_index(x):
    """3-month mean (at least 2 sampled months), standardized by calendar month over available months."""
    r = x.rolling(3, min_periods=2).mean()
    out = pd.DataFrame(np.nan, index=r.index, columns=r.columns)
    for m in range(1, 13):
        sel = r.index.month == m
        for c in r.columns:
            v = r.loc[sel, c]
            ok = v.notna()
            if ok.sum() >= 5:
                p = (v[ok].rank() - 0.44) / (ok.sum() + 0.12)
                out.loc[v[ok].index, c] = norm.ppf(p)
    return out


def lags(idx, spei):
    rows = []
    for b in idx.columns:
        z, sp = idx[b].to_numpy(), spei[b].to_numpy()
        for s, e in runs(z):
            if e >= len(z) - 1 or not np.isfinite(z[e + 1]):
                continue  # end not observed
            ok = np.where((sp[s:-1] >= -0.5) & (sp[s + 1:] >= -0.5))[0]
            if ok.size and s + ok[0] <= e + 12:
                rows.append({"basin": b, "onset": idx.index[s], "lag": e - (s + ok[0]), "duration": e - s + 1})
    return pd.DataFrame(rows)


def compare(name, obs, sim, spei, attr):
    lo, ls = lags(own_index(obs), spei), lags(own_index(sim.where(obs.notna())), spei)
    bo, bs = lo.groupby("basin").lag.agg(["mean", "size"]), ls.groupby("basin").lag.agg(["mean", "size"])
    j = bo.join(bs, lsuffix="_obs", rsuffix="_sim").dropna()
    j = j[(j.size_obs >= 3) & (j.size_sim >= 3)].join(attr[["elev_band"]])
    r, p = spearmanr(j.mean_obs, j.mean_sim)
    print(f"\n{name}: events obs {len(lo)}, sim {len(ls)}; median lag obs {lo.lag.median():.1f}, sim {ls.lag.median():.1f} months")
    print(f"  basin-mean lag, {len(j)} basins with >=3 events each: Spearman {r:.2f} (p={p:.3f})")
    print(j.groupby("elev_band")[["mean_obs", "mean_sim"]].median().round(2).to_string())
    j.to_csv(ROOT / f"runs/validation_{name}_{RUN}.csv", float_format="%.4g")


def main():
    frames = [simulate(f"{RUN}_s{s}") for s in range(NSEED)]
    q = sum(f["q"] for f in frames) / NSEED
    tws = sum(f["soil"] + f["snow"] + f["g1"] + f["g2"] for f in frames) / NSEED
    spei = std_index(frames[0]["pb"])
    basins = list(q.columns)
    attr = groups(basins)
    recent = q.index >= "2000-01-01"
    compare("station_runoff", obs_runoff(basins, q.index[recent]), q[recent], spei[recent], attr)
    g = pd.read_csv(ROOT / "data/processed/grace_basin_monthly.csv", dtype={"basin": str}, parse_dates=["date"])
    g = g.pivot(index="date", columns="basin", values="tws").reindex(q.index[q.index >= "2002-04-01"])[basins]
    # GRACE is already an anomaly; remove each basin's calendar-month mean from both series.
    clim = lambda x: x - x.groupby(x.index.month).transform("mean")
    ts = tws.loc[g.index].where(g.notna())
    compare("grace_tws", clim(g), clim(ts), spei.loc[g.index], attr)


if __name__ == "__main__":
    main()
