#!/usr/bin/env bash
#SBATCH --job-name=pssolver-v04-s19
#SBATCH --partition=hagan-gpu
#SBATCH --account=hagan-lab
#SBATCH --qos=medium
#SBATCH --gres=gpu:H100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --no-requeue
#SBATCH --output=/scratch1/vincent/PSSolver/pssolver_v0_4_s19_retry_c7c940e_20261103_v2_control/logs/slurm_%j.out
#SBATCH --error=/scratch1/vincent/PSSolver/pssolver_v0_4_s19_retry_c7c940e_20261103_v2_control/logs/slurm_%j.err
set -euo pipefail
export LC_ALL=C

# D-0027: fail before qualification if any required node executable is absent.
# Git is deliberately absent: history tests execute only in login preflight.
for executable in env bash sh; do
  command -v "$executable" || { printf 'MISSING_REQUIRED_COMMAND: %s\n' "$executable" >&2; exit 2; }
done

# This exact block is replayed on the login node with --preflight-only.
: "${S19_CONTROL_ROOT:?}"
: "${S19_OUTPUT_ROOT:?}"
: "${S19_PROVIDER_SUPPORT:?}"
: "${S19_PROVIDER_REFERENCE:?}"
: "${S19_CONSUMER_REFERENCE:?}"
: "${S19_BASELINE_PYTHON:?}"
: "${S19_CANDIDATE_PYTHON:?}"
: "${S19_CONSUMER_PYTHON:?}"
: "${S19_BASELINE_WHEEL:?}"
: "${S19_CANDIDATE_WHEEL:?}"
: "${S19_CONSUMER_WHEEL:?}"
: "${S19_MIRROR_VERSIONS:?}"
: "${S19_MANIFEST_SHA256:?}"
test "$S19_CONTROL_ROOT" = /scratch1/vincent/PSSolver/pssolver_v0_4_s19_retry_c7c940e_20261103_v2_control
test -d "$S19_CONTROL_ROOT/logs"
test ! -e "$S19_OUTPUT_ROOT"
test -f "$S19_BASELINE_PYTHON"
test -f "$S19_CANDIDATE_PYTHON"
test -f "$S19_CONSUMER_PYTHON"
test -d "$S19_PROVIDER_SUPPORT/tests"
test -d "$S19_PROVIDER_SUPPORT/benchmarks"
test ! -e "$S19_PROVIDER_SUPPORT/pssolver"
test ! -e "$S19_PROVIDER_SUPPORT/pssolver_control"
test -d "$S19_PROVIDER_REFERENCE/pssolver"
test -d "$S19_CONSUMER_REFERENCE/.git"
test -f /opt/ohpc/pub/compiler/gcc/7.3.0/bin/gcc
test -f /opt/ohpc/pub/compiler/gcc/7.3.0/bin/g++
test -f /opt/ohpc/pub/modulefiles/cuda/12.1
unset LD_LIBRARY_PATH
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 PIP_DISABLE_PIP_VERSION_CHECK=1 PIP_NO_CACHE_DIR=1
export NVIDIA_TF32_OVERRIDE=0
export CC=/opt/ohpc/pub/compiler/gcc/7.3.0/bin/gcc
export CXX=/opt/ohpc/pub/compiler/gcc/7.3.0/bin/g++
if ! type module >/dev/null 2>&1; then
  set +u
  source /opt/ohpc/admin/lmod/lmod/init/bash
  set -u
fi
command -v module || { printf 'MISSING_REQUIRED_COMMAND: module\n' >&2; exit 2; }
module load cuda/12.1
for executable in nvcc nvidia-smi "$CC" "$CXX" "$S19_BASELINE_PYTHON" "$S19_CANDIDATE_PYTHON" "$S19_CONSUMER_PYTHON" /sbin/ldconfig; do
  command -v "$executable" || { printf 'MISSING_REQUIRED_COMMAND: %s\n' "$executable" >&2; exit 2; }
done
# The compiler drivers invoke their own assembler/linker helpers. Query the
# same configured driver, not an unrelated PATH compiler.
for driver in "$CC" "$CXX"; do
  for program in as ld cc1 cc1plus collect2; do
    executable="$("$driver" -print-prog-name="$program")"
    command -v "$executable" || { printf 'MISSING_REQUIRED_COMMAND: %s\n' "$executable" >&2; exit 2; }
  done
done
# Triton's assembler is distributed in the same verified wheel on each role.
for interpreter in "$S19_BASELINE_PYTHON" "$S19_CANDIDATE_PYTHON" "$S19_CONSUMER_PYTHON"; do
  assembler="$("$interpreter" -I -B -c 'import importlib.util; from pathlib import Path; print(Path(importlib.util.find_spec("triton").origin).parent / "backends/nvidia/bin/ptxas")')"
  command -v "$assembler" || { printf 'MISSING_REQUIRED_COMMAND: %s\n' "$assembler" >&2; exit 2; }
done
"$CC" --version
"$CXX" --version
preflight_label="${SLURM_JOB_ID:-login}"
preflight_arguments=(--mode login)
if [[ "${1:-}" != --preflight-only ]]; then
  : "${SLURM_JOB_ID:?must run only in the separately confirmed job}"
  : "${S19_LOGIN_PREFLIGHT_SHA256:?must bind reviewed successful login preflight}"
  preflight_arguments=(--mode job --login-preflight "$S19_CONTROL_ROOT/logs/preflight_login.json" --login-preflight-sha256 "$S19_LOGIN_PREFLIGHT_SHA256")
fi
"$S19_CANDIDATE_PYTHON" -I -B "$S19_PROVIDER_SUPPORT/benchmarks/validate_v0_4_s19_deployment.py" \
  --root "$S19_CONTROL_ROOT" --manifest "$S19_CONTROL_ROOT/plans/deployment.json" \
  --sha256 "$S19_MANIFEST_SHA256" --ready-output "$S19_CONTROL_ROOT/logs/preflight_${preflight_label}.json" "${preflight_arguments[@]}"
if [[ "${1:-}" == --preflight-only ]]; then exit 0; fi
: "${SLURM_JOB_ID:?must run only in the separately confirmed job}"
case "$SLURM_JOB_ID" in (*[!0-9]*|'') exit 2;; esac

node_root="/tmp/pssolver_v04_s19_${SLURM_JOB_ID}"
test ! -e "$node_root"
"$S19_CANDIDATE_PYTHON" -I -B -c 'import os,sys; os.mkdir(sys.argv[1])' "$node_root"
# Only this new per-job node-local directory is removed by this trap.
trap '"$S19_CANDIDATE_PYTHON" -I -B -c '\''import shutil,sys; shutil.rmtree(sys.argv[1])'\'' "$node_root"' EXIT
"$S19_CANDIDATE_PYTHON" -I -B -c 'import os,sys; os.mkdir(sys.argv[1])' "$node_root/pytest_tmp"
export TMPDIR="$node_root/pytest_tmp"
export MPLCONFIGDIR="$node_root/matplotlib"
export PYTHONPYCACHEPREFIX="$node_root/pycache"
"$CC" --version > "$S19_CONTROL_ROOT/logs/gcc_${SLURM_JOB_ID}.txt" 2>&1
module list > "$S19_CONTROL_ROOT/logs/modules_${SLURM_JOB_ID}.txt" 2>&1
nvcc --version > "$S19_CONTROL_ROOT/logs/nvcc_${SLURM_JOB_ID}.txt" 2>&1
"$S19_CANDIDATE_PYTHON" -I -B "$S19_PROVIDER_SUPPORT/benchmarks/run_rc5_g1_checkpoint_tmp_preflight.py" \
  --root "$node_root/rename_noreplace" --expected-tmpdir "$TMPDIR" --output "$S19_CONTROL_ROOT/logs/tmp_${SLURM_JOB_ID}.json"
"$S19_CANDIDATE_PYTHON" -I -B "$S19_PROVIDER_SUPPORT/benchmarks/run_v0_4_s19_job.py" \
  --device cuda --support "$S19_PROVIDER_SUPPORT" --output "$S19_OUTPUT_ROOT" \
  --provider-reference "$S19_PROVIDER_REFERENCE" --consumer-reference "$S19_CONSUMER_REFERENCE" \
  --baseline-python "$S19_BASELINE_PYTHON" --candidate-python "$S19_CANDIDATE_PYTHON" --consumer-python "$S19_CONSUMER_PYTHON" \
  --baseline-wheel "$S19_BASELINE_WHEEL" --candidate-wheel "$S19_CANDIDATE_WHEEL" --consumer-wheel "$S19_CONSUMER_WHEEL" \
  --mirror-versions "$S19_MIRROR_VERSIONS" --tmp-preflight "$S19_CONTROL_ROOT/logs/tmp_${SLURM_JOB_ID}.json" \
  --login-preflight "$S19_CONTROL_ROOT/logs/preflight_login.json" --login-preflight-sha256 "$S19_LOGIN_PREFLIGHT_SHA256"
