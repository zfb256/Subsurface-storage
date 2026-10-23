"""Recovery events under alternative drought definitions, and exclusion counts."""
import os
from pathlib import Path

import pandas as pd

from recovery import events, simulate

ROOT = Path(__file__).resolve().parents[1]
RUN = os.environ.get("RUN", "mcssmc")
VARIANTS = {"spei1": dict(spei_scale=1), "spei6": dict(spei_scale=6), "thr10": dict(thr=-1.0), "nomerge": dict(merge=False)}


def main():
    counts = {}
    frames = {s: simulate(f"{RUN}_s{s}") for s in range(10)}
    for s, fr in frames.items():
        for t in ("soil", "runoff"):
            c = {}
            events(fr, t, counts=c)
            for k, v in c.items():
                counts[(t, k)] = counts.get((t, k), 0) + v
    print("Main definition, all models: droughts and exclusions")
    for t in ("runoff", "soil"):
        n = counts.get((t, "droughts"), 0)
        print(f"  {t}: {n / 10:.0f} droughts per model; at record end {counts.get((t, 'at_record_end'), 0) / n:.1%}; "
              f"no atmospheric recovery within 12 months of the end {counts.get((t, 'no_atm_recovery'), 0) / n:.1%}")
    for name, kw in VARIANTS.items():
        out = [pd.concat([events(frames[s], t, **kw) for t in ("soil", "runoff")]).assign(seed=s) for s in range(3)]
        pd.concat(out).to_csv(ROOT / f"runs/defn_events_{name}.csv", index=False, float_format="%.4g")
        print(name, "written", flush=True)


if __name__ == "__main__":
    main()
