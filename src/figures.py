"""Manuscript figures (Fig. 1-4) from the saved results in runs/."""
import json
from pathlib import Path

import geopandas as gpd
import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RUNS, FIG, BND = ROOT / "runs", ROOT / "figures", ROOT / "data/hydromlyr/0_basin_boundary"
RUN = "mcssmc"
BANDS = ["<1000 m", "1000-2500 m", ">=2500 m"]
BAND_LABEL = {"<1000 m": "<1000 m", "1000-2500 m": "1000–2500 m", ">=2500 m": "≥2500 m"}
LBL = {"runoff": "Runoff drought", "soil": "Soil moisture drought"}
BAND_COLOR = {"<1000 m": "#d9a441", "1000-2500 m": "#7a9a5b", ">=2500 m": "#4f7cac"}
plt.rcParams.update({"font.size": 8, "axes.titlesize": 8, "axes.titleweight": "bold", "axes.titlelocation": "left"})

NN = pd.read_csv(ROOT / "data/processed/nonnested_basins.csv", dtype={"basin": str}).basin.tolist()
NAT = set(open(ROOT / "data/hydromlyr/natural_basins.txt").read().split())


def attrs():
    a = pd.DataFrame({b.stem: json.load(open(ROOT / f"data/hydromlyr/1_data/{b.stem}/attributes.json"))
                      for b in sorted(BND.glob("*.shp"))}).T
    a["elev_band"] = pd.cut(a.elev.astype(float), [-1, 1000, 2500, 9000], labels=BANDS).astype(str)
    return a


def shapes(basins):
    g = pd.concat([gpd.read_file(BND / f"{b}.shp").to_crs(4326).assign(basin=b) for b in basins])
    g = gpd.GeoDataFrame(g, crs=4326)
    return g.assign(_a=g.to_crs(3857).area).sort_values("_a", ascending=False)


def axis_geo(ax):
    ax.set_xlim(95.5, 119.5)
    ax.set_ylim(32, 42)
    ax.set_xlabel("Longitude (°E)")
    ax.set_ylabel("Latitude (°N)")
    ax.set_aspect(1.2)


def fig1(a):
    g = shapes(a.index).merge(a[["elev_band"]], left_on="basin", right_index=True)
    fig, ax = plt.subplots(figsize=(6.5, 3.4), constrained_layout=True)
    big = g[g._a > 4e10]
    big.boundary.plot(ax=ax, color="0.2", linewidth=1.0)
    for band in BANDS:
        sub = g[(g.elev_band == band) & g.basin.isin(NN)]
        sub.plot(ax=ax, color=BAND_COLOR[band], edgecolor="white", linewidth=0.3,
                 label=f"{BAND_LABEL[band]} ({len(sub)})")
    nat = g[g.basin.isin(NN) & g.basin.isin(NAT)]
    nat.plot(ax=ax, facecolor="none", edgecolor="0.1", hatch="////", linewidth=0)
    g[~g.basin.isin(NN)].boundary.plot(ax=ax, color="0.6", linewidth=0.3)
    handles = [plt.Rectangle((0, 0), 1, 1, color=BAND_COLOR[b]) for b in BANDS]
    labels = [f"{BAND_LABEL[b]} (n = {((g.elev_band == b) & g.basin.isin(NN)).sum()})" for b in BANDS]
    handles += [plt.Rectangle((0, 0), 1, 1, facecolor="none", hatch="////", edgecolor="0.1"),
                plt.Line2D([], [], color="0.6", lw=0.6), plt.Line2D([], [], color="0.2", lw=1.0)]
    labels += [f"No dam (GRanD), n = {len(nat)}", "Nested catchments (not in tests)", "Large nested catchments (> 40,000 km²)"]
    ax.legend(handles, labels, loc="upper left", frameon=False, fontsize=7)
    axis_geo(ax)
    fig.savefig(FIG / "Fig1_study_area.png", dpi=300)


def basin_lag(target):
    """Catchment mean recovery lag over all droughts 1961-2024, averaged over the ten models."""
    ev = pd.concat([pd.read_csv(RUNS / f"recovery_events_{RUN}_s{s}.csv", dtype={"basin": str}).assign(seed=s)
                    for s in range(10)])
    ev = ev[ev.target == target]
    return ev.groupby(["seed", "basin"]).lag.mean().groupby("basin").mean().rename("lag")


def fig2():
    lv = pd.read_csv(RUNS / f"levels_main_{RUN}.csv")
    fig = plt.figure(figsize=(7.2, 5.0), constrained_layout=True)
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 0.75])
    g = shapes(NN)
    for j, (target, vmax) in enumerate((("runoff", 16), ("soil", 8))):
        ax = fig.add_subplot(gs[0, j])
        d = g.merge(basin_lag(target).reset_index(), on="basin")
        d.plot(column="lag", cmap="YlOrBr", vmin=0, vmax=vmax, edgecolor="0.3", linewidth=0.3, ax=ax,
               legend=True, legend_kwds={"label": "Recovery lag (months)", "shrink": 0.6, "location": "bottom"})
        axis_geo(ax)
        ax.set_title(f"({'ab'[j]}) {LBL[target]}")
    for j, target in enumerate(("runoff", "soil")):
        ax = fig.add_subplot(gs[1, j])
        for i, band in enumerate(BANDS):
            r = lv[(lv.target == target) & (lv.metric == "lag") & (lv.group == band)].iloc[0]
            ax.bar(i, r.value, 0.6, color=BAND_COLOR[band])
            ax.plot([i, i], [r.lo, r.hi], color="0.1", lw=1.2)
        ax.set_xticks(range(3))
        ax.set_xticklabels([BAND_LABEL[b] for b in BANDS])
        ax.set_ylabel("Mean recovery lag (months)")
        ax.set_title(f"({'cd'[j]}) {LBL[target]}, zone means")
    fig.savefig(FIG / "Fig4_recovery_lag.png", dpi=300)


def fig3():
    """State-reset lag shortening by store and zone, with the equal-volume reset."""
    lv = pd.read_csv(RUNS / f"levels_main_{RUN}.csv")
    lv = lv[lv.target == "runoff"]
    eq = pd.read_csv(RUNS / f"sensitivity_variants_{RUN}.csv")
    eq = eq[eq.variant == "equal 10 mm in total"].set_index("elev_band")
    stores = [("short_snow", "Snow"), ("short_soil", "Soil"), ("short_fast gw", "Fast\nsubsurface"),
              ("short_slow gw", "Slow\ngroundwater"), ("short_subsurface", "Fast + slow\nsubsurface")]
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.9), sharey=True, constrained_layout=True)
    x = np.arange(len(stores))
    for k, band in enumerate(BANDS):
        pos = x + (k - 1) * 0.26
        vals = [lv[(lv.metric == s) & (lv.group == band)].iloc[0] for s, _ in stores]
        base = lv[(lv.metric == "baseline") & (lv.group == band)].iloc[0].value
        axes[0].bar(pos, [v.value for v in vals], 0.26, color=BAND_COLOR[band],
                    label=f"{BAND_LABEL[band]} (baseline {base:.1f} months)")
        for p_, v in zip(pos, vals):
            axes[0].plot([p_, p_], [v.lo, v.hi], color="0.1", lw=0.9)
        axes[0].axhline(base, color=BAND_COLOR[band], ls="--", lw=0.8)  # baseline lag of the zone
        axes[1].bar(pos, [eq.loc[band, s] for s, _ in stores], 0.26, color=BAND_COLOR[band])
    for ax, title in zip(axes, ("(a) Deficit refilled to the 1961–1990 mean", "(b) 10 mm added (fast + slow: 5 mm each)")):
        ax.set_xticks(x)
        ax.set_xticklabels([n for _, n in stores], fontsize=7)
        ax.set_title(title)
    axes[0].set_ylabel("Lag shortening (months)")
    h, l = axes[0].get_legend_handles_labels()
    axes[1].legend(h, l, frameon=False, fontsize=7, loc="upper left")
    fig.savefig(FIG / "Fig3_state_reset.png", dpi=300)


def fig4():
    v = pd.read_csv(RUNS / f"validation_station_runoff_{RUN}.csv", dtype={"basin": str})
    a = attrs()
    v["area"] = v.basin.map(a.area.astype(float))
    v["nat"] = v.basin.isin(NAT)
    fig, axes = plt.subplots(1, 2, figsize=(6.4, 3.0), constrained_layout=True, sharex=True, sharey=True)
    from scipy.stats import spearmanr
    for ax, (title, sub), letter in zip(axes, (("Catchments < 10,000 km²", v[v.area < 1e4]),
                                               ("Catchments ≥ 10,000 km²", v[v.area >= 1e4])), "ab"):
        ax.scatter(sub.mean_obs, sub.mean_sim, s=14, c=np.where(sub.nat, "#4f7cac", "#d9a441"), edgecolor="none")
        rho, p = spearmanr(sub.mean_obs, sub.mean_sim)
        ax.set_title(f"({letter}) {title}\nρ = {rho:.2f}, n = {len(sub)}".replace("-", "−"))
        lim = [0, max(v.mean_obs.max(), v.mean_sim.max()) + 1]
        ax.plot(lim, lim, color="0.6", lw=0.6, ls="--")
        ax.set_xlabel("Gauged mean recovery lag (months)")
    axes[0].set_ylabel("Simulated mean recovery lag (months)")
    axes[0].scatter([], [], c="#4f7cac", s=14, label="No dam")
    axes[0].scatter([], [], c="#d9a441", s=14, label="With dam")
    axes[0].legend(frameon=False)
    fig.savefig(FIG / "Fig5_validation.png", dpi=300)


if __name__ == "__main__":
    FIG.mkdir(exist_ok=True)
    a = attrs()
    fig1(a)
    fig2()
    fig3()
    fig4()
