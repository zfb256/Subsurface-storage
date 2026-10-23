"""Builds the catchment-mean monthly model input; its PET differs by about 1% from the input used to train the released models."""
from pathlib import Path
import numpy as np
import pandas as pd
import geopandas as gpd
import xarray as xr
from shapely.geometry import box

ROOT = Path(__file__).resolve().parents[1]
BND = ROOT / "data/hydromlyr/0_basin_boundary"
CMFD = ROOT / "data/cmfd/raw"
LAT, LON = slice(31, 43), slice(94, 121)
DATES = pd.date_range("1951-01-01", "2024-12-01", freq="MS")
POLYS = {shp.stem: gpd.read_file(shp).to_crs(4326).union_all() for shp in sorted(BND.glob("*.shp"))}


def weights(lat, lon, half, valid):
    """Basin-by-cell matrix of covered cell fraction times cos(lat), restricted to valid cells."""
    w = np.zeros((len(POLYS), lat.size, lon.size))
    for k, poly in enumerate(POLYS.values()):
        x0, y0, x1, y1 = poly.bounds
        for i in np.where((lat > y0 - 2 * half) & (lat < y1 + 2 * half))[0]:
            for j in np.where((lon > x0 - 2 * half) & (lon < x1 + 2 * half))[0]:
                if valid[i, j]:
                    cell = box(lon[j] - half, lat[i] - half, lon[j] + half, lat[i] + half)
                    w[k, i, j] = cell.intersection(poly).area / cell.area * np.cos(np.deg2rad(lat[i]))
    w = w.reshape(len(POLYS), -1)
    return w / w.sum(axis=1, keepdims=True)


def basin_mean(field, w):
    """Weighted mean of a (time, lat, lon) array for every basin, returned as (time, basin)."""
    return np.nan_to_num(field.reshape(field.shape[0], -1)) @ w.T


def cmfd(var):
    da = xr.open_dataset(CMFD / f"{var}_CMFD_V0200_B-01_01mo_010deg_195101-202412.nc")[var]
    return da.sel(lat=LAT, lon=LON).values.astype("float64")


def clear_sky(lat, dates):
    """FAO-56 clear-sky radiation Rso (MJ m-2 day-1) at mid-month, shape (time, lat, 1), sea level."""
    j = (dates + pd.Timedelta(days=14)).dayofyear.values[:, None, None]
    phi = np.deg2rad(lat)[None, :, None]
    dr = 1 + 0.033 * np.cos(2 * np.pi * j / 365)
    dec = 0.409 * np.sin(2 * np.pi * j / 365 - 1.39)
    ws = np.arccos(np.clip(-np.tan(phi) * np.tan(dec), -1, 1))
    ra = 24 * 60 / np.pi * 0.082 * dr * (ws * np.sin(phi) * np.sin(dec) + np.cos(phi) * np.cos(dec) * np.sin(ws))
    return 0.75 * ra


def fao56_pet(temp, pres, shum, wind, srad, rso, elev):
    """FAO-56 Penman-Monteith reference ET (mm/day) from monthly-mean CMFD fields, monthly soil heat flux."""
    t = temp - 273.15
    g = np.empty_like(t)
    g[1:-1] = 0.07 * (t[2:] - t[:-2])
    g[0], g[-1] = 0.14 * (t[1] - t[0]), 0.14 * (t[-1] - t[-2])
    p = pres / 1000.0
    es = 0.6108 * np.exp(17.27 * t / (t + 237.3))
    ea = shum * p / (0.622 + 0.378 * shum)
    delta = 4098.0 * es / (t + 237.3) ** 2
    gamma = 0.000665 * p
    u2 = wind * 4.87 / np.log(67.8 * 10.0 - 5.42)
    rs = srad * 0.0864
    rso = rso + 2e-5 * elev * rso / 0.75
    rnl = 4.903e-9 * temp ** 4 * (0.34 - 0.14 * np.sqrt(ea)) * (1.35 * np.clip(rs / rso, 0.3, 1.0) - 0.35)
    rn = 0.77 * rs - rnl
    num = 0.408 * delta * (rn - g) + gamma * 900.0 / (t + 273.0) * u2 * np.maximum(es - ea, 0.0)
    return np.maximum(num / (delta + gamma * (1.0 + 0.34 * u2)), 0.0)


days = DATES.days_in_month.values[:, None]
out = {}

ref = xr.open_dataset(CMFD / "temp_CMFD_V0200_B-01_01mo_010deg_195101-202412.nc").sel(lat=LAT, lon=LON)
lat, lon = ref.lat.values, ref.lon.values
temp = cmfd("temp")
w = weights(lat, lon, 0.05, ~np.isnan(temp[0]))
out["t"] = basin_mean(temp - 273.15, w)
out["p"] = basin_mean(cmfd("prec"), w) * 86400.0 * days
elev = xr.open_dataset(CMFD / "elev_CMFD_V0200_B-00_fx_010deg.nc").elev.sel(lat=LAT, lon=LON).values.astype("float64")
pet = fao56_pet(temp, cmfd("pres"), cmfd("shum"), cmfd("wind"), cmfd("srad"), clear_sky(lat, DATES), elev)
out["pet"] = basin_mean(pet, w) * days
del temp, pet

grun = xr.open_dataset(ROOT / "data/grun/G-RUN_ENSEMBLE_MMM.nc").Runoff.sel(Y=LAT, X=LON)
grun = grun.sel(time=slice("1951-01-01", "2024-12-31"))
w = weights(grun.Y.values, grun.X.values, 0.25, ~np.isnan(grun.isel(time=0).values))
q = pd.DataFrame(basin_mean(grun.values, w), index=pd.DatetimeIndex(grun.time.values).to_period("M").to_timestamp())
out["q_grun"] = q.reindex(DATES).values * days

gleam = xr.open_mfdataset(sorted((ROOT / "data/gleam/SMrz").glob("SMrz_*_yrb.nc")), combine="by_coords").SMrz
gleam = gleam.sel(time=slice("1951-01-01", "2024-12-31"), lat=slice(LAT.stop, LAT.start), lon=LON)
w = weights(gleam.lat.values, gleam.lon.values, 0.05, ~np.isnan(gleam.isel(time=0).values))
sm = pd.DataFrame(basin_mean(gleam.values, w), index=pd.DatetimeIndex(gleam.time.values).to_period("M").to_timestamp())
out["sm_gleam"] = sm.reindex(DATES).values

basins = list(POLYS)
frame = pd.DataFrame({"basin": np.repeat(basins, len(DATES)), "date": np.tile(DATES, len(basins))})
for v in ("p", "t", "pet", "q_grun", "sm_gleam"):
    frame[v] = out[v].T.ravel()
frame.to_csv(ROOT / "data/processed/catchment_monthly.csv", index=False, float_format="%.5g")
print(frame.basin.nunique(), "basins", len(DATES), "months", DATES[0].date(), "to", DATES[-1].date())
