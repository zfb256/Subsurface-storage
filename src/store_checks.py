"""Store deficits, cold-season snow effect, 1961-2024 reference and attribute correlations."""
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from levels import summarize
from recovery import events, simulate, std_index
from spatial_test import bh, groups

ROOT = Path(__file__).resolve().parents[1]
RUN = os.environ.get("RUN", "mcssmc")
ZONES = ["<1000 m", "1000-2500 m", ">=2500 m"]
SHORT = {"snow": "short_snow", "soil": "short_soil", "g1": "short_fast gw", "g2": "short_slow gw"}


def load(nn):
    ev = pd.concat([pd.read_csv(ROOT / f"runs/recovery_events_{RUN}_s{s}.csv", dtype={"basin": str},
                                parse_dates=["onset", "atm_recovery"]).assign(seed=s) for s in range(10)])
    ev = ev[ev.basin.isin(nn)]
    rs = pd.read_csv(ROOT / f"runs/state_reset_{RUN}.csv", dtype={"basin": str}, parse_dates=["onset"])
    rs = rs[rs.basin.isin(nn) & (rs.target == "runoff")]
    m = rs.merge(ev[ev.target == "runoff"][["seed", "basin", "onset", "atm_recovery"] + [f"def_{k}" for k in SHORT]],
                 on=["seed", "basin", "onset"], how="left")
    return ev, m


def basin_zone_mean(df, col, attr):
    b = df.groupby(["seed", "basin"])[col].mean().groupby("basin").mean()
    return b.groupby(attr.elev_band.reindex(b.index)).mean()


def main():
    pd.set_option("display.width", 200)
    nn = pd.read_csv(ROOT / "data/processed/nonnested_basins.csv", dtype={"basin": str}).basin.tolist()
    attr = groups(nn)
    ev, m = load(nn)
    m = m.merge(attr[["elev_band"]], left_on="basin", right_index=True, suffixes=("_x", ""))
    m["rec_month"] = m.atm_recovery.dt.month
    m["cold"] = m.rec_month.isin([11, 12, 1, 2, 3, 4])

    print("A. Share of runoff events whose atmospheric recovery falls in Nov-Apr, by zone:")
    print(m.groupby("elev_band").cold.mean().round(2).to_string())
    print("   Month distribution in the headwaters (share):")
    print(m[m.elev_band == ">=2500 m"].rec_month.value_counts(normalize=True).sort_index().round(2).to_string())
    for cold in (True, False):
        sub = m[(m.elev_band == ">=2500 m") & (m.cold == cold)]
        print(f"   Headwater {'Nov-Apr' if cold else 'May-Oct'} recoveries: n per model {len(sub) / 10:.0f}; "
              f"snow deficit {-sub.def_snow.clip(upper=0).mean():.1f} mm; snow shortening {sub.short_snow.mean():.2f} months; "
              f"fast subsurface {sub['short_fast gw'].mean():.2f}; slow {sub['short_slow gw'].mean():.2f}")
    print()

    print("B. Mean deficit at atmospheric recovery (mm, positive = below 1961-1990 mean) and shortening per 10 mm:")
    rows = []
    for z, g in m.groupby("elev_band"):
        for k, col in SHORT.items():
            deficit = (-g[f"def_{k}"]).clip(lower=0)
            mean_def = deficit.mean()
            ok = deficit > 1
            per10 = (g.loc[ok, col] / deficit[ok] * 10).median() if ok.any() else np.nan
            rows.append({"zone": z, "store": k, "deficit_mm": mean_def, "short_months": g[col].mean(), "median_short_per_10mm": per10})
    print(pd.DataFrame(rows).round(2).to_string(index=False), "\n")

    print("C. Recovery lag by zone with a 1961-2024 reference (fixed 1961-1990 for comparison):")
    rng = np.random.default_rng(0)
    members = {z: attr.index[attr.elev_band == z].to_numpy() for z in ZONES}
    full = []
    for s in range(10):
        fr = simulate(f"{RUN}_s{s}")
        fn = lambda x, _r: std_index(x, ref=(1961, 2024))
        for t in ("soil", "runoff"):
            full.append(events(fr, t, index_fn=fn).assign(seed=s))
    full = pd.concat(full)
    full = full[full.basin.isin(nn)]
    full.to_csv(ROOT / f"runs/store_checks_events_ref6124_{RUN}.csv", index=False, float_format="%.4g")
    out = []
    for name, e in (("1961-1990", ev), ("1961-2024", full)):
        for t, et in e.groupby("target"):
            for r in summarize(et, ["lag"], members, rng, n_boot=1000):
                out.append(dict(r, reference=name, target=t))
    out = pd.DataFrame(out)
    out.to_csv(ROOT / f"runs/store_checks_reference_{RUN}.csv", index=False, float_format="%.4g")
    print(out[["reference", "target", "group", "value", "lo", "hi"]].round(2).to_string(index=False), "\n")

    print("D. Spearman correlation of catchment values with attributes (56 catchments):")
    att = pd.DataFrame({b: json.load(open(ROOT / f"data/hydromlyr/1_data/{b}/attributes.json")) for b in nn}).T
    att = att.apply(pd.to_numeric, errors="coerce")
    att["aridity_P_PET"] = att.pre_mean / att.pet_mean
    att["log_area"] = np.log(att.area)
    cols = ["elev", "slope", "aridity_P_PET", "frac_snow_daily", "sand", "clay", "por", "geol_permeability",
            "geol_porosity", "root_depth_99", "log_area"]
    y = pd.DataFrame({
        "runoff_lag": ev[ev.target == "runoff"].groupby(["seed", "basin"]).lag.mean().groupby("basin").mean(),
        "short_fast": m.groupby(["seed", "basin"])["short_fast gw"].mean().groupby("basin").mean(),
        "short_slow": m.groupby(["seed", "basin"])["short_slow gw"].mean().groupby("basin").mean(),
    })
    rows = []
    for yc in y:
        for c in cols:
            r, p = spearmanr(att.loc[y.index, c], y[yc], nan_policy="omit")
            rows.append({"response": yc, "attribute": c, "rho": r, "p": p})
    cor = pd.DataFrame(rows)
    cor["q_bh"] = bh(cor.p.fillna(1))
    cor.to_csv(ROOT / f"runs/store_checks_attributes_{RUN}.csv", index=False, float_format="%.4g")
    print(cor.sort_values("p").round(3).to_string(index=False), "\n")

    print("E. Restarted baseline vs recovery lag (runoff):")
    m["baseline_censored"] = m.baseline.isna()
    print(f"   events with no termination within 48 months: {m.baseline_censored.mean():.1%}")
    for z, g in m.groupby("elev_band"):
        both = g[g.baseline.notna()]
        print(f"   {z}: recovery lag all events {g.lag_factual.mean():.1f}, same events with finite baseline "
              f"{both.lag_factual.mean():.1f}, baseline {both.baseline.mean():.1f}; censored {g.baseline_censored.mean():.1%}")


if __name__ == "__main__":
    main()
