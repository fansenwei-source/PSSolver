#!/usr/bin/env bash
#SBATCH --job-name=pssolver-v03-s09
#SBATCH --partition=hagan-gpu
#SBATCH --account=hagan-lab
#SBATCH --qos=medium
#SBATCH --gres=gpu:H100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --no-requeue
#SBATCH --output=/home/fansenwei/pssolver_v0_3_s09_h100_85ba3ae_48dd4d3_20261023_v1/logs/slurm_%j.out
#SBATCH --error=/home/fansenwei/pssolver_v0_3_s09_h100_85ba3ae_48dd4d3_20261023_v1/logs/slurm_%j.err

set -euo pipefail

: "${S09_BASELINE_PYTHON:?set baseline installed-wheel Python}"
: "${S09_CANDIDATE_PYTHON:?set candidate installed-wheel Python}"
: "${S09_CONSUMER_PYTHON:?set Consumer installed-wheel Python}"
: "${S09_BASELINE_WHEEL:?set the v0.2.2 wheel path}"
: "${S09_CANDIDATE_WHEEL:?set the v0.3 candidate wheel path}"
: "${S09_CONSUMER_WHEEL:?set the Control candidate wheel path}"
: "${S09_PROVIDER_SUPPORT:?set the non-shadowing Provider support root}"
: "${S09_CONSUMER_SUPPORT:?set the non-shadowing Consumer support root}"
: "${S09_PROVIDER_REFERENCE:?set the candidate Provider reference archive root}"
: "${S09_CONTROL_ROOT:?set the existing qualification control root}"
: "${S09_OUTPUT_ROOT:?set the initially absent result root}"

test -f "${S09_BASELINE_PYTHON}"
test -f "${S09_CANDIDATE_PYTHON}"
test -f "${S09_CONSUMER_PYTHON}"
test -f "${S09_BASELINE_WHEEL}"
test -f "${S09_CANDIDATE_WHEEL}"
test -f "${S09_CONSUMER_WHEEL}"
test -d "${S09_PROVIDER_SUPPORT}/benchmarks"
test -d "${S09_PROVIDER_SUPPORT}/tests"
test -d "${S09_CONSUMER_SUPPORT}/tests"
test -d "${S09_PROVIDER_REFERENCE}/pssolver"
test ! -e "${S09_PROVIDER_SUPPORT}/pssolver"
test ! -e "${S09_CONSUMER_SUPPORT}/pssolver_control"
test ! -e "${S09_OUTPUT_ROOT}"
test -d "${S09_CONTROL_ROOT}/logs"

unset LD_LIBRARY_PATH
export PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1
export NVIDIA_TF32_OVERRIDE=0
export PYTHONPATH="${S09_PROVIDER_SUPPORT}"
export CC=/opt/ohpc/pub/compiler/gcc/7.3.0/bin/gcc
export CXX=/opt/ohpc/pub/compiler/gcc/7.3.0/bin/g++

if ! type module >/dev/null 2>&1; then
  set +u
  source /opt/ohpc/admin/lmod/lmod/init/bash
  set -u
fi
module load cuda/12.1

node_root="/tmp/pssolver_v03_s09_${SLURM_JOB_ID}"
pytest_tmp="${node_root}/pytest_tmp"
rename_root="${node_root}/rename_noreplace"
test ! -e "${node_root}"
trap 'rm -rf "${node_root}"' EXIT
mkdir -p "${pytest_tmp}"
export TMPDIR="${pytest_tmp}"
mkdir -p "${S09_OUTPUT_ROOT}/results" "${S09_OUTPUT_ROOT}/logs"

"${S09_CANDIDATE_PYTHON}" \
  "${S09_PROVIDER_SUPPORT}/benchmarks/run_rc5_g1_checkpoint_tmp_preflight.py" \
  --root "${rename_root}" \
  --expected-tmpdir "${pytest_tmp}" \
  --output "${S09_OUTPUT_ROOT}/results/tmp_preflight.json"

"${S09_CANDIDATE_PYTHON}" \
  "${S09_PROVIDER_SUPPORT}/benchmarks/run_v0_3_s09_environment.py" \
  --baseline-python "${S09_BASELINE_PYTHON}" \
  --candidate-python "${S09_CANDIDATE_PYTHON}" \
  --consumer-python "${S09_CONSUMER_PYTHON}" \
  --baseline-wheel "${S09_BASELINE_WHEEL}" \
  --candidate-wheel "${S09_CANDIDATE_WHEEL}" \
  --consumer-wheel "${S09_CONSUMER_WHEEL}" \
  --provider-support "${S09_PROVIDER_SUPPORT}" \
  --tmp-preflight "${S09_OUTPUT_ROOT}/results/tmp_preflight.json" \
  --compiler-work-root "${S09_OUTPUT_ROOT}/compiler_preflight" \
  --output "${S09_OUTPUT_ROOT}/results/environment.json"

"${S09_CANDIDATE_PYTHON}" \
  "${S09_PROVIDER_SUPPORT}/benchmarks/run_v0_3_s09_cuda_only.py" \
  --test-root "${S09_PROVIDER_SUPPORT}" \
  --detail-output "${S09_OUTPUT_ROOT}/results/cuda_detail.json" \
  --policy-output "${S09_OUTPUT_ROOT}/results/cuda_tf32.json" \
  --output "${S09_OUTPUT_ROOT}/results/cuda.json"

"${S09_CANDIDATE_PYTHON}" \
  "${S09_PROVIDER_SUPPORT}/benchmarks/run_v0_3_s09_long_horizon.py" \
  --python "${S09_CANDIDATE_PYTHON}" \
  --support-root "${S09_PROVIDER_SUPPORT}" \
  --case-root "${S09_OUTPUT_ROOT}/long_horizon_cases" \
  --output "${S09_OUTPUT_ROOT}/results/long_horizon.json"

"${S09_CANDIDATE_PYTHON}" \
  "${S09_PROVIDER_SUPPORT}/benchmarks/run_v0_3_s09_tf32_backward.py" \
  --support-root "${S09_PROVIDER_SUPPORT}" \
  --output "${S09_OUTPUT_ROOT}/results/tf32_backward.json"

"${S09_CANDIDATE_PYTHON}" \
  "${S09_PROVIDER_SUPPORT}/benchmarks/run_v0_3_s09_checkpoints.py" \
  --test-root "${S09_PROVIDER_SUPPORT}" \
  --junit "${S09_OUTPUT_ROOT}/logs/checkpoints.junit.xml" \
  --output "${S09_OUTPUT_ROOT}/results/checkpoints.json"

"${S09_CANDIDATE_PYTHON}" \
  "${S09_PROVIDER_SUPPORT}/benchmarks/run_v0_3_s09_numerical.py" \
  --baseline-python "${S09_BASELINE_PYTHON}" \
  --candidate-python "${S09_CANDIDATE_PYTHON}" \
  --support-root "${S09_PROVIDER_SUPPORT}" \
  --output "${S09_OUTPUT_ROOT}/results/numerical.json"

# Correctness and independent-consumer validation intentionally precede the
# 192-profile performance/memory matrix.
"${S09_CONSUMER_PYTHON}" \
  "${S09_PROVIDER_SUPPORT}/benchmarks/run_v0_3_s09_consumer_h100.py" \
  --support-root "${S09_CONSUMER_SUPPORT}" \
  --provider-reference "${S09_PROVIDER_REFERENCE}" \
  --junit "${S09_OUTPUT_ROOT}/logs/consumer.junit.xml" \
  --output "${S09_OUTPUT_ROOT}/results/consumer.json"

"${S09_CANDIDATE_PYTHON}" \
  "${S09_PROVIDER_SUPPORT}/benchmarks/run_v0_3_s09_profile_matrix.py" \
  --baseline-python "${S09_BASELINE_PYTHON}" \
  --candidate-python "${S09_CANDIDATE_PYTHON}" \
  --support-root "${S09_PROVIDER_SUPPORT}" \
  --output-root "${S09_OUTPUT_ROOT}/profile_matrix" \
  --command-plan "${S09_OUTPUT_ROOT}/profile_command_plan.json"

"${S09_CANDIDATE_PYTHON}" \
  "${S09_PROVIDER_SUPPORT}/benchmarks/assemble_v0_3_s09_h100.py" \
  --node-local-tmp-environment-and-host-compiler "${S09_OUTPUT_ROOT}/results/environment.json" \
  --installed-wheel-cuda-only-tests "${S09_OUTPUT_ROOT}/results/cuda.json" \
  --cuda-long-horizon-hermitian "${S09_OUTPUT_ROOT}/results/long_horizon.json" \
  --tf32-forward-backward-scope "${S09_OUTPUT_ROOT}/results/tf32_backward.json" \
  --checkpoint-restart-and-released-fixture-matrix "${S09_OUTPUT_ROOT}/results/checkpoints.json" \
  --numerical-gradient-taylor-and-replay "${S09_OUTPUT_ROOT}/results/numerical.json" \
  --independent-control-consumer "${S09_OUTPUT_ROOT}/results/consumer.json" \
  --output "${S09_OUTPUT_ROOT}/results/auxiliary.json"

"${S09_CANDIDATE_PYTHON}" \
  "${S09_PROVIDER_SUPPORT}/benchmarks/analyze_v0_3_s09_h100.py" \
  --profile-root "${S09_OUTPUT_ROOT}/profile_matrix/profiles" \
  --auxiliary "${S09_OUTPUT_ROOT}/results/auxiliary.json" \
  --output "${S09_OUTPUT_ROOT}/results/analysis.json"

(
  cd "${S09_OUTPUT_ROOT}"
  find . -type f ! -name checksums.sha256 ! -name COMPLETE -print0 \
    | sort -z \
    | xargs -0 sha256sum > checksums.sha256
  sha256sum --check checksums.sha256
  printf '%s\n' PASS_V0_3_S09_SINGLE_H100_CUMULATIVE_QUALIFICATION > COMPLETE
)
