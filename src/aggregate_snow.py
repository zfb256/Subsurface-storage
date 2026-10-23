"""Catchment-mean monthly snow depth (cm) from the TPDC long-term daily snow depth dataset of China."""
import re
import shutil
import subprocess
import tarfile
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import box

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "data/snow/raw/snow depth"
TMP = ROOT / "data/snow/tmp"
SEVENZIP = r"C:\Program Files\7-Zip\7z.exe"
NCOL, NROW, X0, Y0, DX = 321, 161, 60.0, 15.0, 0.25
LON = X0 + DX * np.arange(NCOL)
LAT = Y0 + DX * (NROW - 1) - DX * np.arange(NROW)  # first row is the northernmost


def weights():
    out = {}
    for shp in sorted((ROOT / "data/hydromlyr/0_basin_boundary").glob("*.shp")):
        poly = gpd.read_file(shp).to_crs(4326).union_all()
        x0, y0, x1, y1 = poly.bounds
        w = np.zeros((NROW, NCOL))
        for i in np.where((LAT > y0 - DX) & (LAT < y1 + DX))[0]:
            for j in np.where((LON > x0 - DX) & (LON < x1 + DX))[0]:
                cell = box(LON[j] - DX / 2, LAT[i] - DX / 2, LON[j] + DX / 2, LAT[i] + DX / 2)
                w[i, j] = cell.intersection(poly).area / cell.area * np.cos(np.deg2rad(LAT[i]))
        out[shp.stem] = w
    return out


def extract(archive):
    if TMP.exists():
        shutil.rmtree(TMP)
    TMP.mkdir(parents=True)
    if archive.name.endswith(".tar.gz"):
        with tarfile.open(archive) as t:
            t.extractall(TMP, filter="data")
    else:
        subprocess.run([SEVENZIP, "x", "-y", f"-o{TMP}", str(archive)], check=True, capture_output=True)
    return sorted(TMP.rglob("*.txt"))


def read_grid(path):
    tok = path.read_text(encoding="ascii", errors="ignore").split()
    # files up to 2014 start with a 12-token header; later files hold only the grid
    vals = np.array(tok[12:] if len(tok) == NROW * NCOL + 12 else tok, dtype=float)
    if vals.size != NROW * NCOL:
        print(f"skipped {path.name}: {vals.size} values", flush=True)
        return None
    g = vals.reshape(NROW, NCOL)
    g[g < 0] = np.nan
    return g


def main():
    w = weights()
    basins = list(w)
    W = np.stack([w[b] for b in basins])  # (basin, row, col)
    archives = sorted(SRC.glob("snowdepth-*.*"), key=lambda p: p.name)
    archives = [a for a in archives if 1979 <= int(re.search(r"(\d{4})", a.name).group(1)) <= 2020]
    archives += [p for p in SRC.glob("*.rar") if not p.name.startswith("snowdepth-")]  # leap-day archive
    rows = []
    cache = ROOT / "data/snow/yearly"
    cache.mkdir(parents=True, exist_ok=True)
    for a in archives:
        cf = cache / (re.sub(r"\W+", "_", a.stem) + ".csv")
        if cf.exists():
            rows.append(pd.read_csv(cf, dtype={"basin": str}, parse_dates=["date"]))
            print(a.name, "cached", flush=True)
            continue
        year_rows = []
        for f in extract(a):
            m = re.match(r"(\d{4})(\d{3})\.txt$", f.name)
            if not m:
                continue
            date = pd.Timestamp(int(m.group(1)), 1, 1) + pd.Timedelta(days=int(m.group(2)) - 1)
            g = read_grid(f)
            if g is None:
                continue
            ok = np.isfinite(g)
            num = np.einsum("bij,ij->b", W, np.where(ok, g, 0.0))
            den = np.einsum("bij,ij->b", W, ok.astype(float))
            year_rows.append(pd.DataFrame({"basin": basins, "date": date, "depth_cm": num / np.where(den > 0, den, np.nan)}))
        y = pd.concat(year_rows)
        y.to_csv(cf, index=False, float_format="%.4f")
        rows.append(y)
        print(a.name, "done", flush=True)
    shutil.rmtree(TMP, ignore_errors=True)
    d = pd.concat(rows)
    d["month"] = d.date.dt.to_period("M").dt.to_timestamp()
    out = d.groupby(["basin", "month"], as_index=False).depth_cm.mean().rename(columns={"month": "date"})
    out.to_csv(ROOT / "data/processed/snowdepth_basin_monthly.csv", index=False, float_format="%.4f")
    print(out.basin.nunique(), "basins,", out.date.min().date(), "to", out.date.max().date(),
          "| days read:", d.date.nunique())


if __name__ == "__main__":
    main()
