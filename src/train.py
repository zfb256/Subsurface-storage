"""Train MC-SSM-ABCD+ on 102 Yellow River basins against G-RUN runoff, GLEAM soil and GRACE TWS."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from model import NAMES, ParamNet, water_balance

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data/processed"
ATTRS = ["area", "elev", "slope", "lat", "lon", "pre_mean", "pet_mean", "tem_mean", "frac_snow_daily",
         "ndvi_mean", "lai_max", "cropland", "grassland", "sand", "clay", "por", "geol_permeability",
         "geol_porosity", "root_depth_99", "pdep", "pop_dnsty"]
WARM = 120  # 1951-1960 spin-up


def load():
    df = pd.read_csv(PROC / "drought_indices_v2.csv", dtype={"basin": str}, parse_dates=["date"],
                     usecols=["basin", "date", "p", "t", "pet", "q_grun", "sm_gleam"])
    gr = pd.read_csv(PROC / "grace_basin_monthly.csv", dtype={"basin": str}, parse_dates=["date"])
    df = df.merge(gr, on=["basin", "date"], how="left")
    basins = sorted(df.basin.unique())
    wide = {v: df.pivot(index="date", columns="basin", values=v)[basins] for v in ("p", "t", "pet", "q_grun", "sm_gleam", "tws")}
    dates = wide["p"].index
    attr = pd.DataFrame([json.load(open(ROOT / f"data/hydromlyr/1_data/{b}/attributes.json")) for b in basins])[ATTRS]
    attr["area"] = np.log(attr["area"])
    attr["pop_dnsty"] = np.log1p(attr["pop_dnsty"])
    attr = (attr - attr.mean()) / attr.std()
    T = lambda x: torch.tensor(np.asarray(x, dtype=np.float32))
    return basins, dates, {k: T(v) for k, v in wide.items()}, T(attr.fillna(0).values)


def dyn_inputs(x, dates):
    m = torch.tensor(dates.month.values, dtype=torch.float32)[:, None].expand_as(x["p"])
    feats = [torch.log1p(x["p"]), torch.log1p(x["pet"]), x["t"] / 10, torch.sin(2 * np.pi * m / 12), torch.cos(2 * np.pi * m / 12)]
    # 12-month running precipitation gives the encoder a direct handle on multi-month deficits.
    p12 = torch.nn.functional.avg_pool1d(torch.log1p(x["p"]).T[:, None], 12, 1, padding=0)[:, 0].T
    feats.append(torch.cat([p12[:1].expand(11, -1), p12]))
    return torch.stack(feats, -1)


def masked_stats(x, m, dim=0):
    w = m.float()
    n = w.sum(dim).clamp_min(1)
    mu = (x * w).sum(dim) / n
    sd = torch.sqrt((((x - mu) ** 2) * w).sum(dim) / n).clamp_min(1e-6)
    return mu, sd


def month_anom(x, month, mask):
    """Standardize by calendar month over masked months; values outside the mask are zero."""
    out = torch.zeros_like(x)
    for k in range(1, 13):
        sel = month == k
        mu, sd = masked_stats(x[sel], mask[sel])
        out[sel] = torch.where(mask[sel], (x[sel] - mu) / sd, 0.0)
    return out


def corr(a, b, m):
    w = m.float()
    n = w.sum(0).clamp_min(1)
    a = (a - (a * w).sum(0) / n) * w
    b = (b - (b * w).sum(0) / n) * w
    return (a * b).sum(0) / torch.sqrt((a * a).sum(0) * (b * b).sum(0) + 1e-9), m.sum(0)


def kge(sim, obs, m):
    r, _ = corr(sim, obs, m)
    ms, ss = masked_stats(sim, m)
    mo, so = masked_stats(obs, m)
    return 1 - torch.sqrt((r - 1) ** 2 + (ss / so - 1) ** 2 + (ms / mo.clamp_min(1e-6) - 1) ** 2 + 1e-12)


def evaluate(out, x, dates, period):
    d = dates[WARM:]
    month = torch.tensor(d.month.values)
    yr = torch.tensor(d.year.values)[:, None].expand_as(out["q"])
    sel = (yr >= period[0]) & (yr <= period[1])
    q_obs, sm_obs, tws_obs = (x[k][WARM:] for k in ("q_grun", "sm_gleam", "tws"))
    mq, msm, mtw = (sel & torch.isfinite(v) for v in (q_obs, sm_obs, tws_obs))
    q_obs, sm_obs, tws_obs = (torch.nan_to_num(v) for v in (q_obs, sm_obs, tws_obs))
    k_q = kge(out["q"], q_obs, mq)
    r_sm, _ = corr(month_anom(out["soil"], month, msm), month_anom(sm_obs, month, msm), msm)
    tws = out["soil"] + out["snow"] + out["g1"] + out["g2"]
    r_tw, n_tw = corr(tws, tws_obs, mtw)
    return k_q, r_sm, r_tw, n_tw


def recession(q, p, month, mask):
    """Mean log monthly recession rate in Nov-Mar when next-month precipitation is below its basin median."""
    p_med = torch.nanmedian(torch.where(mask, p, torch.nan), 0).values
    m = (mask[:-1] & mask[1:] & torch.isin(month[1:], torch.tensor([11, 12, 1, 2, 3]))[:, None]
         & (p[1:] < p_med) & (q[:-1] > 0.01))
    r = torch.log(q[1:].clamp_min(1e-3) / q[:-1].clamp_min(1e-3))
    w = m.float()
    return (r * w).sum(0) / w.sum(0).clamp_min(1), m.sum(0)


def slope(y, mask):
    """Least-squares linear trend (per year) of y over masked months."""
    tt = torch.arange(y.shape[0], dtype=y.dtype)[:, None].expand_as(y) / 12
    w = mask.float()
    n = w.sum(0).clamp_min(1)
    tm, ym = (tt * w).sum(0) / n, (y * w).sum(0) / n
    return (((tt - tm) * (y - ym) * w).sum(0)) / (((tt - tm) ** 2 * w).sum(0)).clamp_min(1e-6)


def loss_fn(out, x, dates, constrained=False):
    k_q, r_sm, r_tw, n_tw = evaluate(out, x, dates, (1961, 2009))
    l_tw = torch.where(n_tw >= 24, 1 - r_tw, 0).sum() / (n_tw >= 24).sum().clamp_min(1)
    loss = (1 - k_q).mean() + 0.5 * (1 - r_sm).mean() + 0.5 * l_tw
    if not constrained:
        return loss
    d = dates[WARM:]
    month = torch.tensor(d.month.values)
    yr = torch.tensor(d.year.values)[:, None].expand_as(out["q"])
    q_obs = x["q_grun"][WARM:]
    mq = (yr <= 2009) & torch.isfinite(q_obs)
    rec_sim, n1 = recession(out["q"], x["p"][WARM:], month, mq)
    rec_obs, _ = recession(torch.nan_to_num(q_obs), x["p"][WARM:], month, mq)
    l_rec = torch.where(n1 >= 10, (rec_sim - rec_obs) ** 2 / 0.05, 0).mean()
    tws_obs = x["tws"][WARM:]
    mt = (yr <= 2009) & torch.isfinite(tws_obs)
    s_obs = slope(torch.nan_to_num(tws_obs), mt)
    s_sim = slope(out["soil"] + out["snow"] + out["g1"] + out["g2"], mt)
    l_trend = ((s_sim - s_obs) ** 2).mean() / s_obs.var().clamp_min(1e-6)
    return loss + 0.5 * l_rec + 0.25 * l_trend


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--static", action="store_true", help="ablation: no SSM, time-invariant parameters")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--constrained", action="store_true", help="add recession and GRACE trend signatures")
    ap.add_argument("--tag", default="mcssmc")
    a = ap.parse_args()
    torch.manual_seed(a.seed)
    basins, dates, x, attr = load()
    dyn = dyn_inputs(x, dates)
    net = ParamNet(attr.shape[1], dyn.shape[-1], dynamic=not a.static)
    opt = torch.optim.Adam(net.parameters(), lr=3e-3)
    for ep in range(a.epochs):
        prm = net(attr, dyn)
        out = water_balance(prm, x["p"], x["pet"], x["t"], WARM)
        loss = loss_fn(out, x, dates, a.constrained)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step()
        if ep % 10 == 0 or ep == a.epochs - 1:
            print(f"epoch {ep:4d} loss {loss.item():.4f}", flush=True)

    with torch.no_grad():
        prm = net(attr, dyn)
        out = water_balance(prm, x["p"], x["pet"], x["t"], WARM)
        res = {}
        for name, period in (("train", (1961, 2009)), ("test", (2010, 2019)), ("test_tws", (2010, 2024))):
            k_q, r_sm, r_tw, n_tw = evaluate(out, x, dates, period)
            res[name] = {"kge_q_median": k_q.median().item(), "r_sm_median": r_sm.median().item(),
                         "r_tws_median": r_tw[n_tw >= 24].median().item()}
        # Exact mass balance check: P = ET + Q + change in storage.
        st = out["soil"] + out["snow"] + out["g1"] + out["g2"]
        resid = x["p"][WARM + 1:] - out["et"][1:] - out["q"][1:] - (st[1:] - st[:-1])
        res["max_mass_residual_mm"] = resid.abs().max().item()
    print(json.dumps(res, indent=1))
    odir = ROOT / "runs"
    odir.mkdir(exist_ok=True)
    torch.save(net.state_dict(), odir / f"{a.tag}_s{a.seed}.pt")
    json.dump(res, open(odir / f"{a.tag}_s{a.seed}.json", "w"), indent=1)


if __name__ == "__main__":
    main()
