#!/usr/bin/env bash
#SBATCH --job-name=pssolver-rc6-pressure-diag
#SBATCH --partition=hagan-gpu
#SBATCH --account=hagan-lab
#SBATCH --qos=medium
#SBATCH --gres=gpu:H100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:20:00
#SBATCH --no-requeue
#SBATCH --output=/home/fansenwei/pssolver_rc6_pressure_diagnostic_ea801ec_20261018_v1/logs/slurm_%j.out
#SBATCH --error=/home/fansenwei/pssolver_rc6_pressure_diagnostic_ea801ec_20261018_v1/logs/slurm_%j.err

set -euo pipefail

: "${RC6_DRIVER_PYTHON:?set the installed candidate-wheel Python}"
: "${RC6_SUPPORT_ROOT:?set the qualification-support root}"
: "${RC6_DIAGNOSTIC_OUTPUT:?set the initially absent diagnostic output directory}"
: "${RC6_CONTROL_ROOT:?set the existing diagnostic control directory}"

test -f "${RC6_DRIVER_PYTHON}"
test -f "${RC6_SUPPORT_ROOT}/benchmarks/diagnose_rc6_pressure_floor_h100.py"
test -f "${RC6_SUPPORT_ROOT}/benchmarks/run_rc5_g1_checkpoint_tmp_preflight.py"
test -f "${RC6_SUPPORT_ROOT}/benchmarks/run_rc6_cuda_toolkit_preflight.py"
test -d "${RC6_CONTROL_ROOT}/logs"
test -d "${RC6_CONTROL_ROOT}/preflight"
test ! -e "${RC6_DIAGNOSTIC_OUTPUT}"
test ! -e "${RC6_SUPPORT_ROOT}/pssolver"

unset LD_LIBRARY_PATH
export PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1
export NVIDIA_TF32_OVERRIDE=0
export PYTHONPATH="${RC6_SUPPORT_ROOT}"
export CC=/opt/ohpc/pub/compiler/gcc/7.3.0/bin/gcc
export CXX=/opt/ohpc/pub/compiler/gcc/7.3.0/bin/g++

if ! type module >/dev/null 2>&1; then
  set +u
  source /opt/ohpc/admin/lmod/lmod/init/bash
  set -u
fi
module load cuda/12.1

node_root="/tmp/pssolver_rc6_pressure_diag_${SLURM_JOB_ID}"
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
  "${RC6_SUPPORT_ROOT}/benchmarks/run_rc6_cuda_toolkit_preflight.py" \
  --expected-module cuda/12.1 \
  --expected-release 12.1 \
  --output "${RC6_CONTROL_ROOT}/preflight/cuda_toolkit.json"

"${RC6_DRIVER_PYTHON}" \
  "${RC6_SUPPORT_ROOT}/benchmarks/diagnose_rc6_pressure_floor_h100.py" \
  --output-root "${RC6_DIAGNOSTIC_OUTPUT}" \
  --device cuda
