"""Held-out G-RUN recession versus catchment attributes, headwater intervals and censoring by zone."""
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from spatial_test import bh, groups

ROOT = Path(__file__).resolve().parents[1]
RUN = os.environ.get("RUN", "mcssmc")
ATTRS = ["elev", "slope", "aridity_P_PET", "frac_snow_daily", "sand", "clay", "por", "geol_permeability",
         "geol_porosity", "root_depth_99", "log_area"]


def main():
    pd.set_option("display.width", 200)
    nn = pd.read_csv(ROOT / "data/processed/nonnested_basins.csv", dtype={"basin": str}).basin.tolist()
    attr = groups(nn)

    rec = pd.read_csv(ROOT / f"runs/recession_check_{RUN}.csv", dtype={0: str}, index_col=0)
    rec.index = rec.index.astype(str).str.zfill(4)
    att = pd.DataFrame({b: json.load(open(ROOT / f"data/hydromlyr/1_data/{b}/attributes.json")) for b in nn}).T
    att = att.apply(pd.to_numeric, errors="coerce")
    att["aridity_P_PET"] = att.pre_mean / att.pet_mean
    att["log_area"] = np.log(att.area)
    rows = []
    for c in ATTRS:
        r, p = spearmanr(att.loc[rec.index, c], rec.efold_months, nan_policy="omit")
        rows.append({"attribute": c, "rho": r, "p": p})
    cor = pd.DataFrame(rows)
    cor["q_bh"] = bh(cor.p)
    cor.to_csv(ROOT / f"runs/attribute_checks_recession_attributes_{RUN}.csv", index=False, float_format="%.4g")
    print(f"1. Held-out recession e-folding time vs attributes ({len(rec)} catchments):")
    print(cor.sort_values("p").round(3).to_string(index=False), "\n")

    ev = pd.concat([pd.read_csv(ROOT / f"runs/recovery_events_{RUN}_s{s}.csv", dtype={"basin": str},
                                parse_dates=["onset", "atm_recovery"]).assign(seed=s) for s in range(10)])
    rs = pd.read_csv(ROOT / f"runs/state_reset_{RUN}.csv", dtype={"basin": str}, parse_dates=["onset"])
    rs = rs[(rs.target == "runoff") & rs.basin.isin(nn)]
    m = rs.merge(ev[ev.target == "runoff"][["seed", "basin", "onset", "atm_recovery"]], on=["seed", "basin", "onset"])
    m["cold"] = m.atm_recovery.dt.month.isin([11, 12, 1, 2, 3, 4])
    m["block"] = (m.onset.dt.year - 1961) // 5
    high = attr.index[attr.elev_band == ">=2500 m"].to_numpy()
    h = m[m.basin.isin(high) & m.cold]
    print(f"2. Headwater Nov-Apr recoveries: {len(h) / 10:.0f} events per model "
          f"({h.groupby('seed').basin.nunique().mean():.1f} catchments with events)")
    rng = np.random.default_rng(0)
    by_seed = {s: g for s, g in h.groupby("seed")}
    for col in ("short_snow", "short_fast gw"):
        def est(g, wb, wk):
            w = g.block.map(wk).to_numpy(float)
            v = g[col].to_numpy(float)
            ok = np.isfinite(v) & (w > 0)
            d = pd.DataFrame({"b": g.basin.to_numpy()[ok], "wv": (w * v)[ok], "w": w[ok]}).groupby("b").sum()
            bm = (d.wv / d.w).reindex(list(wb)).to_numpy()
            c = np.array(list(wb.values()), float)
            k = np.isfinite(bm)
            return np.sum(bm[k] * c[k]) / np.sum(c[k]) if k.any() else np.nan
        pt = np.mean([est(g, {b: 1 for b in high}, {k: 1 for k in range(13)}) for g in by_seed.values()])
        reps = []
        for _ in range(2000):
            g = by_seed[rng.integers(10)]
            wk = dict(zip(range(13), rng.multinomial(13, np.full(13, 1 / 13))))
            wb = pd.Series(rng.choice(high, high.size)).value_counts().to_dict()
            reps.append(est(g, wb, wk))
        lo, hi = np.nanpercentile(reps, [2.5, 97.5])
        print(f"   {col}: {pt:.2f} months [{lo:.2f}, {hi:.2f}]")
    print()

    m = m.merge(attr[["elev_band"]], left_on="basin", right_index=True, suffixes=("_old", ""))
    print("3. Runoff events without termination within 48 months, by zone:")
    print(m.groupby("elev_band").baseline.apply(lambda s: s.isna().mean()).round(3).to_string())


if __name__ == "__main__":
    main()
