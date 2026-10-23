#!/usr/bin/env bash
# Rebuild all results from the trained models, one process at a time.
set -euo pipefail
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONWARNINGS=ignore
PY=${PY:-../.venv/Scripts/python.exe}
# Optional data preparation from the raw sources; skipped when the output exists.
[ -f ../data/processed/nonnested_basins.csv ] || $PY nonnested_basins.py
[ -f ../data/processed/basin_monthly.csv ] || $PY aggregate_era5land.py
[ -f ../data/processed/grace_basin_monthly.csv ] || $PY aggregate_grace.py
[ -f ../data/processed/catchment_monthly.csv ] || $PY prepare_inputs.py
for s in 0 1 2 3 4 5 6 7 8 9; do
  [ -f ../runs/mcssmc_s$s.pt ] || $PY train.py --epochs 300 --seed $s --constrained --tag mcssmc > ../logs/mcssmc_s$s.log 2>&1
  $PY recovery.py --tag mcssmc_s$s > /dev/null
done
for s in 0 1 2; do $PY recovery.py --tag static_s$s > /dev/null; done
$PY defn_events.py
bash sensitivity_runs.sh
bash partial_reset_runs.sh
bash series_run.sh
$PY levels.py
$PY validate.py
$PY store_checks.py
$PY recession_check.py
$PY attribute_checks.py
$PY sensitivity_summary.py
[ -f ../data/processed/snowdepth_basin_monthly.csv ] || $PY aggregate_snow.py
$PY snow_check.py
$PY figures.py
$PY fig2b_event.py
$PY fig_schematic.py
echo ALL DONE
