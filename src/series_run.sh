#!/usr/bin/env bash
# Alternative structure with the fast and slow stores in series.
set -euo pipefail
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONWARNINGS=ignore STRUCTURE=series
PY=${PY:-../.venv/Scripts/python.exe}
for s in 0 1 2 3 4 5 6 7 8 9; do
  [ -f ../runs/mcssmcser_s$s.pt ] || $PY train.py --epochs 300 --seed $s --constrained --tag mcssmcser > ../logs/mcssmcser_s$s.log 2>&1
  $PY recovery.py --tag mcssmcser_s$s > /dev/null
done
RUN=mcssmcser NSEED=10 $PY mechanism.py > /dev/null
RUN=mcssmcser $PY levels.py > /dev/null
