#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN=/home/fansenwei/anaconda3/envs/Nematics3D/bin/python
SOLVER_SCRIPT=/home/fansenwei/Desktop/Develop/PSSolver/Plane_free_slip_dealiased.py

"${PYTHON_BIN}" "${SOLVER_SCRIPT}" \
  --zero-mode-policy zero_mean \
  --steps 100000 \
  --dt 0.001 \
  --save-interval 1000 \
  --dealias-rule two_thirds \
  --output-dir data_free_slip_dealias_two_thirds_dt0p001_100000_save1000 \
  > dealias_two_thirds_dt0p001_100000.log 2>&1

"${PYTHON_BIN}" "${SOLVER_SCRIPT}" \
  --zero-mode-policy zero_mean \
  --steps 100000 \
  --dt 0.001 \
  --save-interval 1000 \
  --dealias-rule cubic_half \
  --output-dir data_free_slip_dealias_cubic_half_dt0p001_100000_save1000 \
  > dealias_cubic_half_dt0p001_100000.log 2>&1
