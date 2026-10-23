#!/usr/bin/env bash
# Time-invariant models, equal-volume reset and partial resets; finished steps are skipped.
set -euo pipefail
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONWARNINGS=ignore
PY=${PY:-../.venv/Scripts/python.exe}
for s in 0 1 2; do
  [ -f ../runs/static_s$s.pt ] || $PY train.py --epochs 300 --static --seed $s --tag static > ../logs/static_s$s.log 2>&1
done
[ -f ../runs/state_reset_static_static3.csv ] || RUN=static NSEED=3 LABEL=static3 $PY mechanism.py > /dev/null
[ -f ../runs/state_reset_mcssmc_equal10split.csv ] || RUN=mcssmc NSEED=3 RESET_MODE=equal10 LABEL=equal10split $PY mechanism.py > /dev/null
for f in 25 50; do
  [ -f ../runs/state_reset_mcssmc_frac$f.csv ] || RUN=mcssmc NSEED=3 RESET_MODE=frac FRAC=0.$f LABEL=frac$f $PY mechanism.py > /dev/null
done
