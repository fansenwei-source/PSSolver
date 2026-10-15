#!/usr/bin/env bash
#SBATCH --job-name=pssolver-rc5-g1
#SBATCH --partition=hagan-gpu
#SBATCH --account=hagan-lab
#SBATCH --qos=medium
#SBATCH --gres=gpu:H100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --no-requeue

set -euo pipefail

: "${RC5_G1_DRIVER_PYTHON:?set the installed qualification-driver Python}"
: "${RC5_G1_SUPPORT_ROOT:?set the qualification-only support root}"
: "${RC5_G1_STAGE_PLAN:?set the preflighted stage-plan JSON}"
: "${RC5_G1_OUTPUT_ROOT:?set the initially absent H100 output directory}"

test -f "${RC5_G1_DRIVER_PYTHON}"
test -f "${RC5_G1_SUPPORT_ROOT}/benchmarks/run_rc5_g1_stage_plan.py"
test -f "${RC5_G1_STAGE_PLAN}"
test ! -e "${RC5_G1_OUTPUT_ROOT}"
test ! -e "${RC5_G1_SUPPORT_ROOT}/pssolver"

unset LD_LIBRARY_PATH
export PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1
export NVIDIA_TF32_OVERRIDE=0
export PYTHONPATH="${RC5_G1_SUPPORT_ROOT}"

"${RC5_G1_DRIVER_PYTHON}" \
  "${RC5_G1_SUPPORT_ROOT}/benchmarks/run_rc5_g1_stage_plan.py" \
  --plan "${RC5_G1_STAGE_PLAN}" \
  --output-root "${RC5_G1_OUTPUT_ROOT}"
