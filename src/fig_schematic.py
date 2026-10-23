"""Model stores (Fig. 2a) and the state-reset experiment for one simulated drought (Fig. 2b)."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import json

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

plt.rcParams.update({"font.size": 8, "axes.titlesize": 8, "axes.titleweight": "bold", "axes.titlelocation": "left"})
ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "figures"
GREY = "0.25"


def box(ax, x, y, w, h, text, fc):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02", fc=fc, ec="0.2", lw=0.8))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center")


def arrow(ax, a, b, text=None, off=(0, 0), sa=2):
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=9, color=GREY, lw=0.8, shrinkA=sa))
    if text:
        ax.text((a[0] + b[0]) / 2 + off[0], (a[1] + b[1]) / 2 + off[1], text, fontsize=7, color="0.3", ha="center")


fig, (ax, bx) = plt.subplots(1, 2, figsize=(7.2, 3.1), gridspec_kw={"width_ratios": [1, 1.25]}, constrained_layout=True)

ax.set_xlim(0, 10)
ax.set_ylim(0, 10)
ax.axis("off")
ax.set_title("(a) Model stores")
box(ax, 0.1, 8.2, 3.8, 1.2, "Precipitation, PET,\ntemperature", "white")
box(ax, 0.3, 5.6, 3.2, 1.3, "Snow", "#dbe7f3")
box(ax, 4.6, 5.6, 3.2, 1.3, "Soil moisture", "#efe3c8")
box(ax, 1.4, 2.6, 2.9, 1.3, "Fast\nsubsurface", "#e3ecd9")
box(ax, 4.6, 2.6, 3.2, 1.3, "Slow\ngroundwater", "#d3e0c4")
box(ax, 4.4, 0.1, 2.6, 1.1, "Runoff", "white")
arrow(ax, (1.9, 8.2), (1.9, 6.9), "snow", (-0.6, 0))
arrow(ax, (3.4, 8.2), (5.45, 7.05), "rain", (0.5, 0.2))
arrow(ax, (3.5, 6.25), (4.6, 6.25), "melt", (0, 0.25))
box(ax, 6.9, 8.2, 3.0, 1.2, "Evapo-\ntranspiration", "white")
arrow(ax, (7.6, 6.9), (8.2, 8.2))
arrow(ax, (5.6, 5.6), (3.2, 3.9), "recharge", (-0.9, 0))
arrow(ax, (6.6, 5.6), (6.3, 3.9))
ax.plot([7.8, 9.2, 9.2], [6.25, 6.25, 0.65], color=GREY, lw=0.8, solid_capstyle="butt")
arrow(ax, (9.2, 0.65), (7.0, 0.65), sa=0)
ax.text(9.55, 3.25, "direct", fontsize=7, color="0.3", rotation=90, va="center")
arrow(ax, (2.9, 2.6), (4.8, 1.25))
arrow(ax, (6.2, 2.6), (5.8, 1.2))

ev = pd.read_csv(ROOT / "runs/fig2b_event.csv")
meta = json.load(open(ROOT / "runs/fig2b_event.json"))
ev = ev[ev.month.between(-6, 24)]
lag_b, lag_r = meta["lag_baseline"], meta["lag_reset"]
bx.axhline(-0.5, color="0.5", lw=0.7, ls="--")
bx.axvline(0, color="#4f7cac", lw=0.9)
bx.plot(ev.month, ev.factual.where(ev.month <= 0), color="0.6", lw=1.2, label="Before recovery")
bx.plot(ev.month, ev.baseline, color="0.15", lw=1.4, label="Baseline restart")
bx.plot(ev.month, ev.reset, color="#d9a441", lw=1.4, label="Subsurface stores refilled")
y0 = min(ev[["factual", "baseline", "reset"]].min()) - 0.35
bx.annotate("", (lag_b, y0), (0, y0), arrowprops=dict(arrowstyle="<->", color="0.15", lw=0.8, shrinkA=0, shrinkB=0))
bx.text(lag_b / 2, y0 + 0.07, f"baseline lag ({lag_b} months)", ha="center", fontsize=7)
bx.annotate("", (lag_b, y0 - 0.3), (lag_r, y0 - 0.3), arrowprops=dict(arrowstyle="<->", color="#d9a441", lw=0.8, shrinkA=0, shrinkB=0))
bx.text((lag_b + lag_r) / 2, y0 - 0.55, f"lag shortening ({lag_b - lag_r} months)", ha="center", fontsize=7, color="#a07020")
bx.text(0.3, ev[["factual", "baseline", "reset"]].max().max() + 0.1, "Atmospheric recovery", color="#4f7cac", fontsize=7, va="bottom")
bx.text(24, -0.45, "drought threshold", fontsize=7, color="0.4", ha="right", va="bottom")
bx.set_xlim(-6, 24)
bx.set_ylim(y0 - 0.75, ev[["factual", "baseline", "reset"]].max().max() + 0.45)
bx.set_xlabel("Months after atmospheric recovery")
bx.set_ylabel("Standardized runoff index")
bx.set_title("(b) State-reset experiment")
bx.legend(loc="upper center", frameon=False, fontsize=6.5, ncol=3, bbox_to_anchor=(0.45, -0.17), handlelength=1.5, columnspacing=1.0)
fig.savefig(FIG / "Fig2_schematic.png", dpi=300)
