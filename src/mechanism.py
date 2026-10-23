"""State-reset experiment: which store's remaining deficit keeps land drought going after the atmosphere recovers."""
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import norm

from model import ParamNet, water_balance
from spatial_test import groups
from train import WARM, dyn_inputs, load

ROOT = Path(__file__).resolve().parents[1]
RUN = os.environ.get("RUN", "mcssmc")
NSEED = int(os.environ.get("NSEED", 10))
# RESET_MODE: clim (1961-1990 mean, default) | median | equal10 | frac
RESET_MODE = os.environ.get("RESET_MODE", "clim")
THR = float(os.environ.get("THR", -0.5))
FRAC = float(os.environ.get("FRAC", 1.0))
EVENTS = os.environ.get("EVENTS")
LABEL = os.environ.get("LABEL")
H = 48  # months simulated after the reset
RESETS = {"snow": ["snow"], "soil": ["soil"], "fast gw": ["g1"], "slow gw": ["g2"], "subsurface": ["g1", "g2"]}


def reference(series, dates, years=(1961, 1990)):
    """Sorted 3-month sums of the reference years per (calendar month, basin) for the nonparametric index."""
    s3 = torch.nn.functional.avg_pool1d(series.T[:, None], 3, 1)[:, 0].T * 3   # (T-2, N), ends at t
    d = dates[2:]
    ref = {}
    for m in range(1, 13):
        sel = torch.tensor((d.month == m) & (d.year >= years[0]) & (d.year <= years[1]))
        ref[m] = torch.sort(s3[sel], 0).values
    return ref


def index_of(v, month, col, ref):
    r = ref[month][:, col]
    rank = (torch.searchsorted(r.contiguous(), v.contiguous()).double()
            + torch.searchsorted(r.contiguous(), v.contiguous(), right=True).double()) / 2 + 0.5  # mid-rank
    p = (rank - 0.44) / (r.shape[0] + 0.12)  # Gringorten
    return norm.ppf(np.clip(p.numpy(), 1e-3, 1 - 1e-3))


def termination(z):
    ok = (z[:-1] >= THR) & (z[1:] >= THR)
    return int(np.argmax(ok)) - 1 if ok.any() else np.nan


def run_seed(seed, basins, dates, x, attr, dyn, clim_cache):
    tag = f"{RUN}_s{seed}"
    net = ParamNet(attr.shape[1], dyn.shape[-1], dynamic=RUN != "static")
    net.load_state_dict(torch.load(ROOT / f"runs/{tag}.pt"))
    with torch.no_grad():
        prm = net(attr, dyn)
        full = water_balance(prm, x["p"], x["pet"], x["t"], WARM)
    d = dates[WARM:]
    p, pet, t = (x[k][WARM:] for k in ("p", "pet", "t"))
    prm = {k: v[WARM:] for k, v in prm.items()}
    early = torch.tensor((d.year >= 1961) & (d.year <= 1990))
    agg = (lambda v: v.median(0).values) if RESET_MODE == "median" else (lambda v: v.mean(0))
    clim = {k: torch.stack([agg(full[k][early & torch.tensor(d.month == m)]) for m in range(1, 13)])
            for k in ("snow", "soil", "g1", "g2")}
    ref = reference(full["q"].double(), d)
    if EVENTS:
        ev = pd.read_csv(ROOT / EVENTS, dtype={"basin": str}, parse_dates=["onset", "atm_recovery"])
        ev = ev[ev.seed == seed].drop(columns=["seed"])
    else:
        ev = pd.read_csv(ROOT / f"runs/recovery_events_{tag}.csv", dtype={"basin": str}, parse_dates=["onset", "atm_recovery"])
    ev = ev[ev.target == "runoff"]
    pos = {b: i for i, b in enumerate(basins)}
    ev["r"] = d.get_indexer(ev.atm_recovery)
    ev["col"] = ev.basin.map(pos)
    ev = ev[(ev.r >= 3) & (ev.r + H <= len(d)) & (ev.lag >= 0)].reset_index(drop=True)  # only droughts still ongoing at recovery
    r, col = torch.tensor(ev.r.values), torch.tensor(ev.col.values)
    steps = r[None] + torch.arange(H)[:, None]                     # (H, E)
    f = {k: v[steps, col[None]] for k, v in (("p", p), ("pet", pet), ("t", t))}
    pr = {k: v[steps, col[None]] for k, v in prm.items()}
    start = {k: full[k][r - 1, col] for k in ("snow", "soil", "g1", "g2")}
    mon_prev = torch.tensor(d.month.values)[r - 1] - 1
    out = {}
    for name, stores in [("baseline", [])] + list(RESETS.items()):
        init = dict(start)
        for k in stores:
            if RESET_MODE == "equal10":
                init[k] = start[k] + 10.0 / len(stores)  # 10 mm per reset in total, split over grouped stores
            elif RESET_MODE == "frac":
                init[k] = start[k] + FRAC * (clim[k][mon_prev, col] - start[k]).clamp_min(0)  # refill part of the deficit
            else:
                init[k] = torch.maximum(start[k], clim[k][mon_prev, col])
        with torch.no_grad():
            sim = water_balance(pr, f["p"], f["pet"], f["t"], warm=0, init=init)
        out[name] = sim
    rows = []
    months = d.month.values
    for i, e in ev.iterrows():
        prev = full["q"][e.r - 2:e.r, e.col]
        row = {"seed": seed, "basin": e.basin, "target": e.target, "onset": e.onset, "lag_factual": e.lag}
        for name, sim in out.items():
            series = torch.cat([prev, sim["q"][:, i]]).double()
            s3 = series[2:] + series[1:-1] + series[:-2]               # sums ending at r .. r+H-1
            z = np.array([index_of(s3[j:j + 1], int(months[e.r + j]), e.col,
                                   ref)[0] for j in range(H)])
            row[name] = termination(z)
        rows.append(row)
    return pd.DataFrame(rows)


def main():
    basins, dates, x, attr = load()
    dyn = dyn_inputs(x, dates)
    res = pd.concat([run_seed(s, basins, dates, x, attr, dyn, None) for s in range(NSEED)])
    for k in RESETS:
        res[f"short_{k}"] = res.baseline - res[k]
    attrs = groups(sorted(res.basin.unique()))
    res = res.merge(attrs[["elev_band", "wetness"]], left_on="basin", right_index=True)
    suffix = f"_{LABEL}" if LABEL else ""
    res.to_csv(ROOT / f"runs/state_reset_{RUN}{suffix}.csv", index=False, float_format="%.4g")
    cols = [f"short_{k}" for k in RESETS]
    pd.set_option("display.width", 200)
    print("baseline lag (months) and mean lag shortening by removing each store's deficit")
    for by in (["target"], ["target", "elev_band"]):
        print(res.groupby(by)[["baseline"] + cols].mean().round(2).to_string(), "\n")


if __name__ == "__main__":
    main()
