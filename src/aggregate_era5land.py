"""Area-weighted ERA5-Land monthly snow water equivalent (sd, m w.e.) for each HydroMLYR basin."""
from pathlib import Path
import numpy as np
import pandas as pd
import geopandas as gpd
import xarray as xr
from shapely.geometry import box

ROOT = Path(__file__).resolve().parents[1]
BND = ROOT / "data/hydromlyr/0_basin_boundary"
SRC = sorted((ROOT / "data/era5land/monthly").glob("era5land_monthly_*.nc"))
HALF = 0.05

ds = xr.open_mfdataset(SRC, combine="by_coords")
sd = np.nan_to_num(ds.sd.values)
valid = ~np.isnan(ds.sd.isel(valid_time=0).values)
month = pd.DatetimeIndex(ds.valid_time.values).to_period("M").to_timestamp()
lat, lon = ds.latitude.values, ds.longitude.values
coslat = np.cos(np.deg2rad(lat))

rows = []
for shp in sorted(BND.glob("*.shp")):
    poly = gpd.read_file(shp).to_crs(4326).union_all()
    w = np.zeros((lat.size, lon.size))
    x0, y0, x1, y1 = poly.bounds
    for i in np.where((lat > y0 - 2 * HALF) & (lat < y1 + 2 * HALF))[0]:
        for j in np.where((lon > x0 - 2 * HALF) & (lon < x1 + 2 * HALF))[0]:
            if valid[i, j]:
                cell = box(lon[j] - HALF, lat[i] - HALF, lon[j] + HALF, lat[i] + HALF)
                w[i, j] = cell.intersection(poly).area / cell.area * coslat[i]
    series = np.tensordot(sd, w, axes=([1, 2], [0, 1])) / w.sum()
    rows.append(pd.DataFrame({"basin": shp.stem, "date": month, "sd": series}))

out = pd.concat(rows)
dst = ROOT / "data/processed/basin_monthly.csv"
if not dst.exists():
    out.to_csv(dst, index=False, float_format="%.5g")
else:
    old = pd.read_csv(dst, dtype={"basin": str}, parse_dates=["date"])
    m = old[["basin", "date", "sd"]].merge(out, on=["basin", "date"], how="outer", suffixes=("", "_new"))
    # Replace the sd column only if the new series reproduces it within rounding.
    if len(m) == len(old) == len(out) and np.allclose(m.sd, m.sd_new, rtol=1e-4, atol=1e-7):
        old = old.drop(columns="sd").merge(out, on=["basin", "date"])[old.columns]
        old.to_csv(dst, index=False, float_format="%.5g")
    else:
        dst = dst.with_name("era5land_swe_v2.csv")
        print("sd differs from existing basin_monthly.csv; writing", dst.name)
        out.to_csv(dst, index=False, float_format="%.5g")
print(dst.name, out.basin.nunique(), "basins", out.date.nunique(), "months", out.date.min().date(), "to", out.date.max().date())
