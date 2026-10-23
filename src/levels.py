"""Zone means of recovery lag and state-reset lag shortening, with bootstrap intervals."""
import os
from pathlib import Path

import numpy as np
import pandas as pd

from spatial_test import groups, p_boot

ROOT = Path(__file__).resolve().parents[1]
RUN = os.environ.get("RUN", "mcssmc")
ZONES = ["<1000 m", "1000-2500 m", ">=2500 m"]
CONTRASTS = [("1000-2500 m", ">=2500 m"), ("<1000 m", ">=2500 m"), ("<1000 m", "1000-2500 m")]
STORE_COLS = ["baseline", "short_snow", "short_soil", "short_fast gw", "short_slow gw", "short_subsurface"]


def lag_events(nn):
    e = pd.concat([pd.read_csv(ROOT / f"runs/recovery_events_{RUN}_s{s}.csv", dtype={"basin": str},
                               parse_dates=["onset"]).assign(seed=s) for s in range(10)])
    return e[e.basin.isin(nn)]


def reset_events(nn):
    r = pd.read_csv(ROOT / f"runs/state_reset_{RUN}.csv", dtype={"basin": str}, parse_dates=["onset"])
    return r[r.basin.isin(nn) & (r.target == "runoff")]


def zone_means(e, col, members, wk, draws):
    """Equal-catchment zone means; events within a catchment weighted by their block weight."""
    w = e.block.map(wk).to_numpy(float)
    v = e[col].to_numpy(float)
    ok = np.isfinite(v) & (w > 0)
    d = pd.DataFrame({"basin": e.basin.to_numpy()[ok], "wv": (w * v)[ok], "w": w[ok]}).groupby("basin").sum()
    bm = d.wv / d.w
    out = {}
    for z, mb in members.items():
        cnt = draws[z]
        b = bm.reindex(list(cnt)).to_numpy()
        c = np.array(list(cnt.values()), float)
        good = np.isfinite(b)
        out[z] = np.sum(b[good] * c[good]) / np.sum(c[good]) if good.any() else np.nan
    return out


def summarize(e, cols, members, rng, n_boot=2000):
    e = e.assign(block=(e.onset.dt.year - 1961) // 5)
    blocks = np.arange(13)
    by_seed = {s: g for s, g in e.groupby("seed")}
    seeds = sorted(by_seed)
    ones_k = {k: 1 for k in blocks}
    ones_b = {z: {b: 1 for b in mb} for z, mb in members.items()}
    rows = []
    reps = {c: [] for c in cols}
    for _ in range(n_boot):
        g = by_seed[seeds[rng.integers(len(seeds))]]
        wk = dict(zip(blocks, rng.multinomial(13, np.full(13, 1 / 13))))
        draws = {z: pd.Series(rng.choice(mb, mb.size)).value_counts().to_dict() for z, mb in members.items()}
        for c in cols:
            reps[c].append(zone_means(g, c, members, wk, draws))
    for c in cols:
        pt = pd.DataFrame([zone_means(by_seed[s], c, members, ones_k, ones_b) for s in seeds]).mean()
        rp = pd.DataFrame(reps[c])
        for z in ZONES:
            lo, hi = np.nanpercentile(rp[z], [2.5, 97.5])
            rows.append({"metric": c, "group": z, "value": pt[z], "lo": lo, "hi": hi, "p_boot": np.nan})
        for a, b in CONTRASTS:
            d = (rp[a] - rp[b]).to_numpy()
            lo, hi = np.nanpercentile(d, [2.5, 97.5])
            rows.append({"metric": c, "group": f"{a} minus {b}", "value": pt[a] - pt[b], "lo": lo, "hi": hi, "p_boot": p_boot(d)})
    return rows


def main():
    nn = pd.read_csv(ROOT / "data/processed/nonnested_basins.csv", dtype={"basin": str}).basin.tolist()
    attr = groups(nn)
    members = {z: attr.index[attr.elev_band == z].to_numpy() for z in ZONES}
    rng = np.random.default_rng(0)
    rows = []
    for target, et in lag_events(nn).groupby("target"):
        rows += [dict(r, variant="fixed", target=target) for r in summarize(et, ["lag"], members, rng)]
    rows += [dict(r, variant="fixed", target="runoff") for r in summarize(reset_events(nn), STORE_COLS, members, rng)]
    out = pd.DataFrame(rows)
    out["excludes_zero"] = (out.lo > 0) | (out.hi < 0)
    out.to_csv(ROOT / f"runs/levels_main_{RUN}.csv", index=False, float_format="%.4g")
    pd.set_option("display.width", 220)
    print(out[~out.group.str.contains("minus")][["variant", "target", "metric", "group", "value", "lo", "hi"]].round(2).to_string(index=False))
    print(out[out.group.str.contains("minus")][["variant", "target", "metric", "group", "value", "lo", "hi", "p_boot"]]
          .round(3).to_string(index=False))


if __name__ == "__main__":
    main()
