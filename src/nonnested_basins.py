"""Select non-overlapping HydroMLYR catchments: of any pair sharing over 5% of the smaller area, keep the smaller."""
from pathlib import Path
import numpy as np
import pandas as pd
import geopandas as gpd

ROOT = Path(__file__).resolve().parents[1]
BND = ROOT / "data/hydromlyr/0_basin_boundary"
ALBERS = "+proj=aea +lat_1=25 +lat_2=47 +lat_0=0 +lon_0=105 +datum=WGS84 +units=m"

geom = {shp.stem: gpd.read_file(shp).to_crs(ALBERS).union_all() for shp in sorted(BND.glob("*.shp"))}
area = {b: g.area / 1e6 for b, g in geom.items()}

drop = set()
names = list(geom)
for i, a in enumerate(names):
    for b in names[i + 1:]:
        small, big = (a, b) if area[a] <= area[b] else (b, a)
        if geom[a].intersection(geom[b]).area / 1e6 > 0.05 * area[small]:
            drop.add(big)

keep = sorted((b for b in names if b not in drop), key=lambda b: area[b])
out = pd.DataFrame({"basin": keep, "area_km2": [area[b] for b in keep]})
dst = ROOT / "data/processed/nonnested_basins.csv"
if dst.exists():
    old = pd.read_csv(dst, dtype={"basin": str})
    same = old.basin.tolist() == out.basin.tolist() and np.allclose(old.area_km2, out.area_km2, rtol=1e-4)
    # Keep the existing file unless the new selection reproduces it within rounding.
    if not same:
        dst = dst.with_name("nonnested_basins_v2.csv")
        print("differs from existing nonnested_basins.csv; writing", dst.name)
out.to_csv(dst, index=False)
print(len(names), "catchments,", len(out), "non-nested ->", dst.name)
