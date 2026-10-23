"""Drought events and recovery lags of soil moisture and runoff after atmospheric recovery."""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import norm

from model import ParamNet, water_balance
from train import WARM, dyn_inputs, load

ROOT = Path(__file__).resolve().parents[1]
STORES = ["snow", "soil", "g1", "g2"]


def simulate(tag):
    basins, dates, x, attr = load()
    dyn = dyn_inputs(x, dates)
    net = ParamNet(attr.shape[1], dyn.shape[-1], dynamic=not tag.startswith("static"))
    net.load_state_dict(torch.load(ROOT / f"runs/{tag}.pt"))
    with torch.no_grad():
        out = water_balance(net(attr, dyn), x["p"], x["pet"], x["t"], WARM)
    d = dates[WARM:]
    frames = {k: pd.DataFrame(v.numpy(), index=d, columns=basins) for k, v in out.items()}
    frames["pb"] = pd.DataFrame((x["p"] - x["pet"])[WARM:].numpy(), index=d, columns=basins)
    return frames


def std_index(df, scale=3, ref=(1961, 1990), ref_df=None):
    """Nonparametric (Gringorten) standardized index of `scale`-month sums, fitted by calendar month."""
    s = df.rolling(scale).sum()
    base = s if ref_df is None else ref_df.rolling(scale).sum()
    out = pd.DataFrame(np.nan, index=s.index, columns=s.columns)
    for m in range(1, 13):
        sel = s.index.month == m
        refsel = sel & (s.index.year >= ref[0]) & (s.index.year <= ref[1])
        for c in s.columns:
            r = base.loc[refsel, c].dropna().to_numpy()
            v = s.loc[sel, c].to_numpy()
            rs = np.sort(r)
            rank = (np.searchsorted(rs, v, "left") + np.searchsorted(rs, v, "right")) / 2 + 0.5  # mid-rank
            p = (rank - 0.44) / (len(r) + 0.12)  # Gringorten
            out.loc[sel, c] = norm.ppf(np.clip(p, 1e-3, 1 - 1e-3))
    return out


def runs(z, thr=-0.5, merge=True):
    """Start/end indices of drought runs; one-month gaps are merged unless merge=False."""
    below = np.nan_to_num(z, nan=0) < thr
    ev, i, n = [], 0, len(below)
    while i < n:
        if below[i]:
            j = i
            while j + 1 < n and (below[j + 1] or (merge and j + 2 < n and below[j + 2])):
                j += 1 if below[j + 1] else 2
            if j - i + 1 >= 2:
                ev.append((i, j))
            i = j + 1
        else:
            i += 1
    return ev


def events(frames, target, ref_frames=None, index_fn=None, thr=-0.5, merge=True, spei_scale=3, counts=None):
    """`index_fn(series, ref_series)` replaces the standardization (used by the sensitivity analyses)."""
    rf = ref_frames or frames
    key = "soil" if target == "soil" else "q"
    fn = index_fn or (lambda s, r: std_index(s, ref_df=r))
    spei = fn(frames["pb"], rf["pb"]) if spei_scale == 3 else std_index(frames["pb"], scale=spei_scale, ref_df=rf["pb"])
    idx = fn(frames[key], rf[key])
    clim = {k: frames[k] - rf[k].groupby(rf[k].index.month).transform(
        lambda s: s[(s.index.year >= 1961) & (s.index.year <= 1990)].mean()) for k in STORES}
    rows = []
    dates = idx.index
    for b in idx.columns:
        sp = spei[b].to_numpy()
        n = len(sp)
        for s, e in runs(idx[b].to_numpy(), thr, merge):
            if counts is not None:
                counts["droughts"] = counts.get("droughts", 0) + 1
            if e >= n - 1:
                if counts is not None:
                    counts["at_record_end"] = counts.get("at_record_end", 0) + 1
                continue  # still in drought at the record end: censored
            # Atmospheric recovery: first month from onset with SPEI-3 back above the drought threshold
            ok = np.where((sp[s:-1] >= thr) & (sp[s + 1:] >= thr))[0]
            rec = s + ok[0] if ok.size else None
            if rec is None or rec > e + 12:
                if counts is not None:
                    counts["no_atm_recovery"] = counts.get("no_atm_recovery", 0) + 1
                continue
            row = {"basin": b, "target": target, "onset": dates[s], "end": dates[e], "duration": e - s + 1,
                   "atm_recovery": dates[rec], "lag": e - rec}
            anom = {k: clim[k][b].iloc[max(rec - 1, 0)] for k in STORES}  # state when the reset is applied
            for k in STORES:
                row[f"def_{k}"] = anom[k]
            rows.append(row)
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="mcssmc_s0")
    a = ap.parse_args()
    frames = simulate(a.tag)
    ev = pd.concat([events(frames, t) for t in ("soil", "runoff")])
    ev.to_csv(ROOT / f"runs/recovery_events_{a.tag}.csv", index=False, float_format="%.4g")
    print(ev.groupby("target").lag.describe().round(2).to_string())


if __name__ == "__main__":
    main()
