"""Simulated snowpack compared with ERA5-Land and passive-microwave snow water equivalent."""
import os
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from recovery import simulate
from spatial_test import groups

ROOT = Path(__file__).resolve().parents[1]
RUN = os.environ.get("RUN", "mcssmc")


def era5_swe():
    d = pd.read_csv(ROOT / "data/processed/basin_monthly.csv", dtype={"basin": str}, parse_dates=["date"], usecols=["basin", "date", "sd"])
    w = d.pivot(index="date", columns="basin", values="sd") * 1000.0  # m water equivalent -> mm
    return w.loc["1961":"2024"]


def anomaly(x):
    ref = x.loc["1961":"1990"]
    clim = ref.groupby(ref.index.month).mean()
    return x - clim.reindex(x.index.month).to_numpy()


def main():
    pd.set_option("display.width", 200)
    nn = pd.read_csv(ROOT / "data/processed/nonnested_basins.csv", dtype={"basin": str}).basin.tolist()
    attr = groups(nn)
    high = attr.index[attr.elev_band == ">=2500 m"].tolist()
    era = era5_swe()
    model = sum(simulate(f"{RUN}_s{s}")["snow"] for s in range(10)) / 10
    model = model.loc[era.index]

    print("1. Snow storage, headwater catchments (mm)")
    rows = []
    for b in high:
        m, e = model[b], era[b]
        r, _ = spearmanr(anomaly(m.to_frame())[b], anomaly(e.to_frame())[b])
        rows.append({"basin": b, "model_peak": m.groupby(m.index.year).max().mean(), "era5_peak": e.groupby(e.index.year).max().mean(),
                     "model_mean": m.mean(), "era5_mean": e.mean(), "anomaly_rho": r})
    t = pd.DataFrame(rows).set_index("basin")
    print(t.round(2).to_string())
    clim = pd.DataFrame({"model": model[high].mean(1).groupby(model.index.month).mean(),
                         "era5": era[high].mean(1).groupby(era.index.month).mean()})
    print("   headwater mean seasonal cycle (mm):")
    print(clim.T.round(1).to_string(), "\n")

    ev = pd.concat([pd.read_csv(ROOT / f"runs/recovery_events_{RUN}_s{s}.csv", dtype={"basin": str},
                                parse_dates=["atm_recovery"]).assign(seed=s) for s in range(10)])
    ev = ev[(ev.target == "runoff") & ev.basin.isin(nn)].merge(attr[["elev_band"]], left_on="basin", right_index=True)
    am, ae = anomaly(model), anomaly(era)
    prev = ev.atm_recovery - pd.DateOffset(months=1)
    ev["model_def"] = [-am.at[p, b] if p in am.index else np.nan for p, b in zip(prev, ev.basin)]
    ev["era5_def"] = [-ae.at[p, b] if p in ae.index else np.nan for p, b in zip(prev, ev.basin)]
    ev["cold"] = ev.atm_recovery.dt.month.isin([11, 12, 1, 2, 3, 4])
    print("2. Snowpack deficit at atmospheric recovery (mm, positive = below 1961-1990 mean; mean over events)")
    g = ev.groupby("elev_band")[["model_def", "era5_def"]].agg(lambda s: s.clip(lower=0).mean())
    print(g.round(2).to_string())
    h = ev[(ev.elev_band == ">=2500 m") & ev.cold]
    print(f"   headwater Nov-Apr recoveries: model {h.model_def.clip(lower=0).mean():.1f} mm, ERA5-Land {h.era5_def.clip(lower=0).mean():.1f} mm; "
          f"ERA5-Land 90th percentile {h.era5_def.clip(lower=0).quantile(0.9):.1f} mm, max {h.era5_def.max():.1f} mm")
    ae.to_csv(ROOT / "data/processed/era5land_swe_anomaly.csv", float_format="%.3f")

    # passive-microwave snow depth (TPDC, 1979-2020) converted to SWE at 0.24 g cm-3
    f = ROOT / "data/processed/snowdepth_basin_monthly.csv"
    if not f.exists():
        return
    sd = pd.read_csv(f, dtype={"basin": str}, parse_dates=["date"]).pivot(index="date", columns="basin", values="depth_cm")
    sd = sd.loc["1979":"2020"]
    obs = sd * 10 * 0.24  # mm SWE

    def anom_79(x):
        ref = x.loc["1979":"2008"]
        return x - ref.groupby(ref.index.month).mean().reindex(x.index.month).to_numpy()

    print("\n3. TPDC passive-microwave snow depth, headwater catchments (SWE with density 0.24; 1979-2020)")
    rows = []
    for b in high:
        o, m, e = obs[b], model[b].loc[obs.index], era[b].loc[obs.index]
        ao, am_, ae_ = (anom_79(s.to_frame())[b] for s in (o, m, e))
        rows.append({"basin": b, "tpdc_peak": o.groupby(o.index.year).max().mean(), "model_peak": m.groupby(m.index.year).max().mean(),
                     "era5_peak": e.groupby(e.index.year).max().mean(),
                     "rho_model_tpdc": spearmanr(am_, ao, nan_policy="omit")[0], "rho_era5_tpdc": spearmanr(ae_, ao, nan_policy="omit")[0]})
    print(pd.DataFrame(rows).set_index("basin").round(2).to_string())
    cyc = pd.DataFrame({"tpdc (0.24)": obs[high].mean(1).groupby(obs.index.month).mean(),
                        "model": model.loc[obs.index, high].mean(1).groupby(obs.index.month).mean(),
                        "era5": era.loc[obs.index, high].mean(1).groupby(obs.index.month).mean()})
    print("   headwater mean seasonal cycle (mm SWE):")
    print(cyc.T.round(1).to_string())
    ao, am_, ae_ = anom_79(obs), anom_79(model.loc[obs.index]), anom_79(era.loc[obs.index])
    e2 = ev[(ev.atm_recovery.dt.year >= 1979) & (ev.atm_recovery.dt.year <= 2020)].copy()
    p2 = e2.atm_recovery - pd.DateOffset(months=1)
    for name, a in (("tpdc_def", ao), ("model_def79", am_), ("era5_def79", ae_)):
        e2[name] = [-a.at[p, b] if (p in a.index and b in a.columns) else np.nan for p, b in zip(p2, e2.basin)]
    h2 = e2[(e2.elev_band == ">=2500 m")]
    hc = h2[h2.cold]
    print("   snowpack deficit at atmospheric recovery, headwaters, 1979-2020 events (mm SWE, mean of positive part):")
    for lab, s in (("all", h2), ("Nov-Apr", hc)):
        print(f"     {lab}: TPDC {s.tpdc_def.clip(lower=0).mean():.1f} (density 0.20-0.30: "
              f"{s.tpdc_def.clip(lower=0).mean() / 0.24 * 0.20:.1f}-{s.tpdc_def.clip(lower=0).mean() / 0.24 * 0.30:.1f}), "
              f"model {s.model_def79.clip(lower=0).mean():.1f}, ERA5-Land {s.era5_def79.clip(lower=0).mean():.1f}; "
              f"TPDC 90th percentile {s.tpdc_def.clip(lower=0).quantile(0.9):.1f}")


if __name__ == "__main__":
    main()
