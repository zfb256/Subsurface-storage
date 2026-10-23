"""Zone assignment and statistical helpers shared by the analysis scripts."""
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def groups(basins):
    a = pd.DataFrame({b: json.load(open(ROOT / f"data/hydromlyr/1_data/{b}/attributes.json")) for b in basins}).T
    a["aridity"] = a.pre_mean.astype(float) / a.pet_mean.astype(float)
    a["elev_band"] = pd.cut(a.elev.astype(float), [-1, 1000, 2500, 9000], labels=["<1000 m", "1000-2500 m", ">=2500 m"]).astype(str)
    med = a.aridity.median()
    a["wetness"] = np.where(a.aridity < med, f"dry (P/PET<{med:.2f})", f"wet (P/PET>={med:.2f})")
    return a


def p_boot(v):
    """Two-sided bootstrap p-value for a zero change: twice the smaller tail share."""
    v = v[np.isfinite(v)]
    return min(1.0, 2 * min((v <= 0).mean(), (v >= 0).mean()))


def bh(p):
    """Benjamini-Hochberg adjusted p-values."""
    p = np.asarray(p, float)
    order = np.argsort(p)
    ranked = p[order] * len(p) / (np.arange(len(p)) + 1)
    q = np.minimum.accumulate(ranked[::-1])[::-1]
    out = np.empty_like(q)
    out[order] = np.minimum(q, 1)
    return out
