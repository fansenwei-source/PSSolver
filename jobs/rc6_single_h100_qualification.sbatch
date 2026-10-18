#!/usr/bin/env bash
#SBATCH --job-name=pssolver-rc6
#SBATCH --partition=hagan-gpu
#SBATCH --account=hagan-lab
#SBATCH --qos=medium
#SBATCH --gres=gpu:H100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --no-requeue
#SBATCH --output=/home/fansenwei/pssolver_rc6_h100_ea801ec_208edd0_20261018_v1/logs/slurm_%j.out
#SBATCH --error=/home/fansenwei/pssolver_rc6_h100_ea801ec_208edd0_20261018_v1/logs/slurm_%j.err

set -euo pipefail

: "${RC6_DRIVER_PYTHON:?set the installed qualification-driver Python}"
: "${RC6_SUPPORT_ROOT:?set the qualification-only support root}"
: "${RC6_STAGE_PLAN:?set the frozen stage-plan JSON}"
: "${RC6_OUTPUT_ROOT:?set the initially absent H100 output directory}"
: "${RC6_CONTROL_ROOT:?set the existing control directory}"

test -f "${RC6_DRIVER_PYTHON}"
test -f "${RC6_SUPPORT_ROOT}/benchmarks/run_rc6_single_h100_stage_plan.py"
test -f "${RC6_SUPPORT_ROOT}/benchmarks/run_rc5_g1_checkpoint_tmp_preflight.py"
test -f "${RC6_STAGE_PLAN}"
test -d "${RC6_CONTROL_ROOT}/logs"
test ! -e "${RC6_OUTPUT_ROOT}"
test ! -e "${RC6_SUPPORT_ROOT}/pssolver"

unset LD_LIBRARY_PATH
export PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1
export NVIDIA_TF32_OVERRIDE=0
export PYTHONPATH="${RC6_SUPPORT_ROOT}"
export CC=/opt/ohpc/pub/compiler/gcc/7.3.0/bin/gcc
export CXX=/opt/ohpc/pub/compiler/gcc/7.3.0/bin/g++

node_root="/tmp/pssolver_rc6_${SLURM_JOB_ID}"
preflight_root="${node_root}/rename_noreplace"
pytest_tmp="${node_root}/pytest_tmp"
test ! -e "${node_root}"
mkdir -p "${pytest_tmp}"
export TMPDIR="${pytest_tmp}"

"${RC6_DRIVER_PYTHON}" \
  "${RC6_SUPPORT_ROOT}/benchmarks/run_rc5_g1_checkpoint_tmp_preflight.py" \
  --root "${preflight_root}" \
  --expected-tmpdir "${pytest_tmp}" \
  --output "${RC6_CONTROL_ROOT}/preflight/checkpoint_tmp.json"

"${RC6_DRIVER_PYTHON}" \
  "${RC6_SUPPORT_ROOT}/benchmarks/run_rc6_single_h100_stage_plan.py" \
  --plan "${RC6_STAGE_PLAN}" \
  --output-root "${RC6_OUTPUT_ROOT}"
