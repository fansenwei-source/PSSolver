#!/usr/bin/env bash
#SBATCH --job-name=pssolver-rc5-g1-v16
#SBATCH --partition=hagan-gpu
#SBATCH --account=hagan-lab
#SBATCH --qos=medium
#SBATCH --gres=gpu:H100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --no-requeue
#SBATCH --output=/home/fansenwei/pssolver_rc5_g1_balanced_crossover_dabeb9a_20261016_v16/logs/slurm_%j.out
#SBATCH --error=/home/fansenwei/pssolver_rc5_g1_balanced_crossover_dabeb9a_20261016_v16/logs/slurm_%j.err

set -euo pipefail

: "${RC5_G1_BASELINE_PYTHON:?set the installed baseline Python}"
: "${RC5_G1_CANDIDATE_PYTHON:?set the installed candidate Python}"
: "${RC5_G1_CONSUMER_PYTHON:?set the installed Consumer Python}"
: "${RC5_G1_BASELINE_WHEEL:?set the immutable baseline wheel}"
: "${RC5_G1_CANDIDATE_WHEEL:?set the immutable candidate wheel}"
: "${RC5_G1_CONSUMER_WHEEL:?set the immutable Consumer wheel}"
: "${RC5_G1_SUPPORT_ROOT:?set the qualification-only support root}"
: "${RC5_G1_OUTPUT_ROOT:?set the initially absent output root}"
: "${RC5_G1_CONTROL_ROOT:?set the persistent control root}"
: "${RC5_G1_AUTHORIZATION_RECORD:?set the source-bound authorization record}"

test -f "${RC5_G1_BASELINE_PYTHON}"
test -f "${RC5_G1_CANDIDATE_PYTHON}"
test -f "${RC5_G1_CONSUMER_PYTHON}"
test -f "${RC5_G1_BASELINE_WHEEL}"
test -f "${RC5_G1_CANDIDATE_WHEEL}"
test -f "${RC5_G1_CONSUMER_WHEEL}"
test -f "${RC5_G1_AUTHORIZATION_RECORD}"
test -f "${RC5_G1_SUPPORT_ROOT}/benchmarks/run_rc5_g1_checkpoint_tmp_preflight.py"
test -f "${RC5_G1_SUPPORT_ROOT}/benchmarks/run_rc5_g1_environment.py"
test -f "${RC5_G1_SUPPORT_ROOT}/benchmarks/run_rc5_g1_balanced_crossover_profile_matrix.py"
test ! -e "${RC5_G1_SUPPORT_ROOT}/pssolver"
test ! -e "${RC5_G1_OUTPUT_ROOT}"

RC5_G1_NODE_LOCAL_ROOT="/tmp/pssolver_rc5_g1_balanced_crossover_v16_${SLURM_JOB_ID}"
RC5_G1_TMP_PREFLIGHT_REPORT="${RC5_G1_CONTROL_ROOT}/provenance/node_local_tmp_preflight.json"
test ! -e "${RC5_G1_NODE_LOCAL_ROOT}"
test ! -e "${RC5_G1_TMP_PREFLIGHT_REPORT}"
case "${RC5_G1_NODE_LOCAL_ROOT}" in
  /tmp/pssolver_rc5_g1_balanced_crossover_v16_*) ;;
  *) echo "unexpected node-local root" >&2; exit 2 ;;
esac

cleanup_node_local() {
  rm -rf -- "${RC5_G1_NODE_LOCAL_ROOT}"
}
trap cleanup_node_local EXIT

unset LD_LIBRARY_PATH
export PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1
export NVIDIA_TF32_OVERRIDE=0
export PYTHONPATH="${RC5_G1_SUPPORT_ROOT}"

mkdir -m 700 "${RC5_G1_NODE_LOCAL_ROOT}"
mkdir -m 700 "${RC5_G1_NODE_LOCAL_ROOT}/pytest_tmp"
export TMPDIR="${RC5_G1_NODE_LOCAL_ROOT}/pytest_tmp"
export TMP="${TMPDIR}"
export TEMP="${TMPDIR}"

"${RC5_G1_CANDIDATE_PYTHON}" \
  "${RC5_G1_SUPPORT_ROOT}/benchmarks/run_rc5_g1_checkpoint_tmp_preflight.py" \
  --root "${RC5_G1_NODE_LOCAL_ROOT}/rename_noreplace_preflight" \
  --expected-tmpdir "${TMPDIR}" \
  --output "${RC5_G1_TMP_PREFLIGHT_REPORT}"

"${RC5_G1_CANDIDATE_PYTHON}" \
  "${RC5_G1_SUPPORT_ROOT}/benchmarks/run_rc5_g1_environment.py" \
  --baseline-python "${RC5_G1_BASELINE_PYTHON}" \
  --candidate-python "${RC5_G1_CANDIDATE_PYTHON}" \
  --consumer-python "${RC5_G1_CONSUMER_PYTHON}" \
  --baseline-wheel "${RC5_G1_BASELINE_WHEEL}" \
  --candidate-wheel "${RC5_G1_CANDIDATE_WHEEL}" \
  --consumer-wheel "${RC5_G1_CONSUMER_WHEEL}" \
  --provider-support "${RC5_G1_SUPPORT_ROOT}" \
  --output "${RC5_G1_OUTPUT_ROOT}/results/environment.json"

"${RC5_G1_CANDIDATE_PYTHON}" \
  "${RC5_G1_SUPPORT_ROOT}/benchmarks/run_rc5_g1_balanced_crossover_profile_matrix.py" \
  --baseline-python "${RC5_G1_BASELINE_PYTHON}" \
  --candidate-python "${RC5_G1_CANDIDATE_PYTHON}" \
  --support-root "${RC5_G1_SUPPORT_ROOT}" \
  --output-root "${RC5_G1_OUTPUT_ROOT}/profile_matrix" \
  --command-plan "${RC5_G1_OUTPUT_ROOT}/results/profile_command_plan.json" \
  --authorization-record "${RC5_G1_AUTHORIZATION_RECORD}"

test -f "${RC5_G1_OUTPUT_ROOT}/profile_matrix/MATRIX_COMPLETE.json"
test -f "${RC5_G1_OUTPUT_ROOT}/profile_matrix/analysis.json"
test ! -e "${RC5_G1_OUTPUT_ROOT}/COMPLETE"
touch "${RC5_G1_OUTPUT_ROOT}/BALANCED_CROSSOVER_COMPLETE"
