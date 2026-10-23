"""Lag shortening under all design variants, undammed catchments, headwater leave-one-out and spatial blocks."""
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from levels import summarize
from recovery import simulate
from spatial_test import groups

ROOT = Path(__file__).resolve().parents[1]
RUN = os.environ.get("RUN", "mcssmc")
ZONES = ["<1000 m", "1000-2500 m", ">=2500 m"]
COLS = ["baseline", "short_snow", "short_soil", "short_fast gw", "short_slow gw", "short_subsurface"]
NN = pd.read_csv(ROOT / "data/processed/nonnested_basins.csv", dtype={"basin": str}).basin.tolist()
ATTR = groups(NN)
NAT = set(open(ROOT / "data/hydromlyr/natural_basins.txt").read().split())


def reset_file(name):
    r = pd.read_csv(ROOT / f"runs/{name}.csv", dtype={"basin": str}, parse_dates=["onset"])
    r = r[(r.target == "runoff") & r.basin.isin(NN)].drop(columns=["elev_band"], errors="ignore")
    return r.merge(ATTR[["elev_band"]], left_on="basin", right_index=True)


def zone_means(r, cols=COLS, basins=None):
    if basins is not None:
        r = r[r.basin.isin(basins)]
    b = r.groupby(["seed", "basin"])[[c for c in cols if c in r]].mean().groupby("basin").mean()
    return b.groupby(ATTR.elev_band.reindex(b.index)).mean()


def main():
    pd.set_option("display.width", 220)
    main_r = reset_file(f"state_reset_{RUN}")

    print("A. Lag shortening (months) by zone and variant")
    variants = {"main (10 models)": main_r, "main (models 0-2)": main_r[main_r.seed < 3]}
    variants["series structure (10 models)"] = reset_file("state_reset_mcssmcser")
    for label, f in (("time-invariant parameters (3 models)", "state_reset_static_static3"),
                     ("equal 10 mm in total", f"state_reset_{RUN}_equal10split"),
                     ("refill 25% of deficit", f"state_reset_{RUN}_frac25"), ("refill 50% of deficit", f"state_reset_{RUN}_frac50"),
                     ("reset to median", f"state_reset_{RUN}_median"), ("SPEI-1", f"state_reset_{RUN}_spei1"),
                     ("SPEI-6", f"state_reset_{RUN}_spei6"), ("no gap merging", f"state_reset_{RUN}_nomerge"),
                     ("threshold -1.0", f"state_reset_{RUN}_thr10")):
        if (ROOT / f"runs/{f}.csv").exists():
            variants[label] = reset_file(f)
    rows = []
    for label, r in variants.items():
        z = zone_means(r)
        z["variant"] = label
        rows.append(z.reset_index())
    a = pd.concat(rows)
    a.to_csv(ROOT / f"runs/sensitivity_variants_{RUN}.csv", index=False, float_format="%.4g")
    print(a.round(2).to_string(index=False), "\n")
    rng = np.random.default_rng(0)
    members = {z: ATTR.index[ATTR.elev_band == z].to_numpy() for z in ZONES}
    sub = pd.DataFrame(summarize(main_r, ["short_subsurface"], members, rng))
    print("   main run, combined subsurface reset with intervals:")
    print(sub.round(2).to_string(index=False), "\n")

    print("B. Main run, catchments without dams")
    nat = [b for b in NN if b in NAT]
    print(zone_means(main_r, basins=nat).round(2).to_string())
    print("   catchments per zone:", ATTR.loc[nat].elev_band.value_counts().to_dict(), "\n")

    print("C. Headwater catchments (main run, mean of models)")
    high = ATTR.index[ATTR.elev_band == ">=2500 m"]
    hc = main_r[main_r.basin.isin(high)].groupby(["seed", "basin"])[COLS].mean().groupby("basin").mean()
    print(hc.round(2).to_string())
    loo = pd.DataFrame({b: hc.drop(b).mean() for b in high}).T
    print("   leave-one-out headwater means, range over the seven drops:")
    print(pd.DataFrame({"min": loo.min(), "max": loo.max()}).round(2).to_string(), "\n")

    print("D. Slow-store shortening by zone for each of the ten models")
    d = main_r.groupby(["seed", "basin"])["short_slow gw"].mean().reset_index().merge(ATTR[["elev_band"]], left_on="basin", right_index=True)
    print(d.groupby(["seed", "elev_band"])["short_slow gw"].mean().unstack().round(2).to_string(), "\n")

    print("E. Negative recovery lags (runoff events, all models)")
    ev = pd.concat([pd.read_csv(ROOT / f"runs/recovery_events_{RUN}_s{s}.csv", dtype={"basin": str}).assign(seed=s) for s in range(10)])
    ev = ev[(ev.target == "runoff") & ev.basin.isin(NN)].merge(ATTR[["elev_band"]], left_on="basin", right_index=True)
    print("   share with lag < 0:", ev.groupby("elev_band").lag.apply(lambda s: (s < 0).mean()).round(3).to_dict())
    zl = zone_means(ev.rename(columns={"lag": "lag_all"}).assign(lag_pos=ev.lag.clip(lower=0)), ["lag_all", "lag_pos"])
    print(zl.round(2).to_string(), "\n")

    print("F. Baseline lag with censored restarts counted as 48 months (upper bound)")
    f = main_r.assign(baseline_ub=main_r.baseline.fillna(48))
    print(zone_means(f, ["baseline", "baseline_ub"]).round(2).to_string(), "\n")

    print("G. Spatial-block bootstrap (8 blocks from catchment centroids) for key correlations")
    att = pd.DataFrame({b: json.load(open(ROOT / f"data/hydromlyr/1_data/{b}/attributes.json")) for b in NN}).T.apply(pd.to_numeric, errors="coerce")
    att["aridity_P_PET"] = att.pre_mean / att.pet_mean
    xy = att[["lon", "lat"]].to_numpy()
    cent = xy[np.random.default_rng(1).choice(len(xy), 8, replace=False)]
    for _ in range(50):  # plain k-means
        lab = np.argmin(((xy[:, None] - cent[None]) ** 2).sum(-1), 1)
        cent = np.array([xy[lab == k].mean(0) if (lab == k).any() else cent[k] for k in range(8)])
    blocks = {k: att.index[lab == k].to_numpy() for k in range(8)}
    rec = pd.read_csv(ROOT / f"runs/recession_check_{RUN}.csv", index_col=0)
    rec.index = rec.index.astype(str).str.zfill(4)
    y = pd.DataFrame({
        "short_fast": main_r.groupby(["seed", "basin"])["short_fast gw"].mean().groupby("basin").mean(),
        "short_slow": main_r.groupby(["seed", "basin"])["short_slow gw"].mean().groupby("basin").mean(),
        "recession_efold": rec.efold_months}).reindex(NN)
    pairs = [("short_fast", "clay"), ("short_fast", "sand"), ("short_slow", "aridity_P_PET"), ("short_slow", "slope"),
             ("recession_efold", "aridity_P_PET"), ("recession_efold", "slope"), ("recession_efold", "clay"), ("recession_efold", "sand")]
    rng = np.random.default_rng(0)
    for yc, xc in pairs:
        r0 = spearmanr(att.loc[NN, xc], y[yc], nan_policy="omit")[0]
        reps = []
        for _ in range(2000):
            pick = np.concatenate([blocks[k] for k in rng.integers(0, 8, 8)])
            reps.append(spearmanr(att.loc[pick, xc], y.loc[pick, yc], nan_policy="omit")[0])
        lo, hi = np.nanpercentile(reps, [2.5, 97.5])
        print(f"   {yc} vs {xc}: rho {r0:.2f} [{lo:.2f}, {hi:.2f}]")
    print()

    print("H. Snow in the headwaters")
    print("   fraction of precipitation as snow (HydroMLYR frac_snow_daily), headwater catchments:",
          att.loc[high, "frac_snow_daily"].round(2).to_dict())
    fr = simulate(f"{RUN}_s0")
    snow = fr["snow"][list(high)]
    peak = snow.groupby(snow.index.year).max().mean()
    print("   simulated mean annual peak snow storage (mm, model 0):", peak.round(1).to_dict())


if __name__ == "__main__":
    main()
