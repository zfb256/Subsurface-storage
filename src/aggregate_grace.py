"""Area-weighted CSR RL06.3 mascon TWS anomaly (mm) for each HydroMLYR basin, by calendar month."""
from pathlib import Path
import numpy as np
import pandas as pd
import geopandas as gpd
import xarray as xr
from shapely.geometry import box

ROOT = Path(__file__).resolve().parents[1]
BND = ROOT / "data/hydromlyr/0_basin_boundary"
SRC = ROOT / "data/grace/CSR_GRACE_GRACE-FO_RL0603_Mascons_all-corrections.nc"

ds = xr.open_dataset(SRC, decode_times=False).sel(lat=slice(30, 43), lon=slice(94, 121))
tws = ds.lwe_thickness.values * 10.0  # cm -> mm
mid = pd.Timestamp("2002-01-01") + pd.to_timedelta(ds.time.values, unit="D")
month = mid.to_period("M").to_timestamp()
lat, lon = ds.lat.values, ds.lon.values
coslat = np.cos(np.deg2rad(lat))

rows = []
for shp in sorted(BND.glob("*.shp")):
    poly = gpd.read_file(shp).to_crs(4326).union_all()
    w = np.zeros((lat.size, lon.size))
    x0, y0, x1, y1 = poly.bounds
    for i in np.where((lat > y0 - 0.25) & (lat < y1 + 0.25))[0]:
        for j in np.where((lon > x0 - 0.25) & (lon < x1 + 0.25))[0]:
            cell = box(lon[j] - 0.125, lat[i] - 0.125, lon[j] + 0.125, lat[i] + 0.125)
            w[i, j] = cell.intersection(poly).area / cell.area * coslat[i]
    series = np.tensordot(tws, w, axes=([1, 2], [0, 1])) / w.sum()
    rows.append(pd.DataFrame({"basin": shp.stem, "date": month, "tws": series}))

out = pd.concat(rows)
# Two solutions can fall in one calendar month around GRACE gaps; average them.
out = out.groupby(["basin", "date"], as_index=False).tws.mean()
out.to_csv(ROOT / "data/processed/grace_basin_monthly.csv", index=False, float_format="%.5g")
print(out.basin.nunique(), "basins", out.date.nunique(), "months", out.date.min().date(), "to", out.date.max().date())
