"""Baseline and subsurface-reset runoff index for one lowland drought, for Fig. 2b."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from mechanism import H, index_of, reference, termination
from model import ParamNet, water_balance
from spatial_test import groups
from train import WARM, dyn_inputs, load

ROOT = Path(__file__).resolve().parents[1]
SEED = 0
PRE = 12


def pick_event(basins):
    r = pd.read_csv(ROOT / "runs/state_reset_mcssmc.csv", dtype={"basin": str}, parse_dates=["onset"])
    nn = pd.read_csv(ROOT / "data/processed/nonnested_basins.csv", dtype={"basin": str}).basin
    zone = groups(sorted(nn)).elev_band
    r = r[(r.seed == SEED) & (r.target == "runoff") & r.basin.isin(nn) & r.basin.isin(basins)]
    r = r[r.basin.map(zone) == "<1000 m"].dropna(subset=["baseline", "subsurface"])
    r = r[(r.baseline >= 6) & (r.subsurface <= 1) & (r.short_snow == 0)]
    return r.iloc[(r.baseline - 8).abs().argsort()]


def trajectories(e, allev, basins, d, x, prm, full):
    ev = allev[(allev.target == "runoff") & (allev.basin == e.basin) & (allev.onset == e.onset)].iloc[0]
    col, r = list(basins).index(e.basin), d.get_indexer([ev.atm_recovery])[0]
    early = torch.tensor((d.year >= 1961) & (d.year <= 1990))
    start = {k: full[k][r - 1, col] for k in ("snow", "soil", "g1", "g2")}
    m_prev = d.month[r - 1]
    clim = {k: full[k][early & torch.tensor(d.month == m_prev), col].mean() for k in ("g1", "g2")}
    steps = torch.arange(r, r + H)
    f = {k: x[k][WARM:][steps, col][:, None] for k in ("p", "pet", "t")}
    pr = {k: v[WARM:][steps, col][:, None] for k, v in prm.items()}
    ref = reference(full["q"].double(), d)
    out = {}
    for name, stores in (("baseline", []), ("reset", ["g1", "g2"])):
        init = {k: v.reshape(1) for k, v in start.items()}
        for k in stores:
            init[k] = torch.maximum(init[k], clim[k].reshape(1))
        with torch.no_grad():
            q = water_balance(pr, f["p"], f["pet"], f["t"], warm=0, init=init)["q"][:, 0]
        series = torch.cat([full["q"][r - 2:r, col], q]).double()
        s3 = series[2:] + series[1:-1] + series[:-2]
        out[name] = np.array([index_of(s3[j:j + 1], int(d.month[r + j]), col, ref)[0] for j in range(H)])
    if out["reset"][:termination(out["baseline"]) + 1].min() < -0.5:
        return None
    return ev, out


def main():
    basins, dates, x, attr = load()
    dyn = dyn_inputs(x, dates)
    net = ParamNet(attr.shape[1], dyn.shape[-1])
    net.load_state_dict(torch.load(ROOT / f"runs/mcssmc_s{SEED}.pt"))
    with torch.no_grad():
        prm = net(attr, dyn)
        full = water_balance(prm, x["p"], x["pet"], x["t"], WARM)
    d = dates[WARM:]
    cands = pick_event(basins)
    allev = pd.read_csv(ROOT / f"runs/recovery_events_mcssmc_s{SEED}.csv", dtype={"basin": str}, parse_dates=["onset", "atm_recovery"])
    for _, e in cands.iterrows():
        res = trajectories(e, allev, basins, d, x, prm, full)
        if res is not None:
            break
    ev, out = res
    col, r = list(basins).index(ev.basin), d.get_indexer([ev.atm_recovery])[0]
    ref = reference(full["q"].double(), d)
    q = full["q"][:, col].double()
    s3 = q[2:] + q[1:-1] + q[:-2]
    pre = np.array([index_of(s3[t - 2:t - 1], int(d.month[t]), col, ref)[0] for t in range(r - PRE, r)])
    pd.DataFrame({"month": np.arange(-PRE, H), "factual": np.r_[pre, out["baseline"]],
                  "baseline": np.r_[[np.nan] * PRE, out["baseline"]], "reset": np.r_[[np.nan] * PRE, out["reset"]]}
                 ).to_csv(ROOT / "runs/fig2b_event.csv", index=False, float_format="%.4g")
    meta = {"basin": ev.basin, "atm_recovery": str(ev.atm_recovery.date()), "onset": str(ev.onset.date()),
            "lag_baseline": int(termination(out["baseline"])), "lag_reset": int(termination(out["reset"]))}
    json.dump(meta, open(ROOT / "runs/fig2b_event.json", "w"), indent=1)
    print(meta)


if __name__ == "__main__":
    main()
