#!/usr/bin/env bash
# State-reset experiment: main design and reset and drought-definition variants.
set -euo pipefail
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONWARNINGS=ignore
PY=${PY:-../.venv/Scripts/python.exe}
RUN=mcssmc NSEED=10 $PY mechanism.py > /dev/null
RUN=mcssmc NSEED=3 RESET_MODE=median LABEL=median $PY mechanism.py > /dev/null
for v in spei1 spei6 nomerge; do
  RUN=mcssmc NSEED=3 EVENTS=runs/defn_events_$v.csv LABEL=$v $PY mechanism.py > /dev/null
done
RUN=mcssmc NSEED=3 EVENTS=runs/defn_events_thr10.csv THR=-1.0 LABEL=thr10 $PY mechanism.py > /dev/null
