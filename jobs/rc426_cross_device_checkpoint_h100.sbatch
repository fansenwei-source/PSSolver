#!/usr/bin/env bash
#SBATCH --job-name=pssolver_rc426
#SBATCH --partition=hagan-gpu
#SBATCH --account=hagan-lab
#SBATCH --qos=medium
#SBATCH --gres=gpu:H100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --no-requeue

set -euo pipefail

: "${PSSOLVER_RC426_VENV:?set the fresh installed-wheel venv}"
: "${PSSOLVER_RC426_CONTROL:?set the immutable control directory}"
: "${PSSOLVER_RC426_SCRATCH:?set the initially empty scratch directory}"
: "${PSSOLVER_RC426_SOURCE_ROOT:?set the forbidden source worktree}"

PYTHON="${PSSOLVER_RC426_VENV}/bin/python"
PURELIB="$(${PYTHON} -c 'import sysconfig; print(sysconfig.get_path("purelib"))')"
QUALIFICATION="${PSSOLVER_RC426_VENV}/qualification"

test -x "${PYTHON}"
test -d "${PURELIB}"
test -f "${QUALIFICATION}/benchmarks/capture_rc4_h100_preflight.py"
test -f "${QUALIFICATION}/benchmarks/diagnose_plane_nyquist_storage.py"
test -f "${QUALIFICATION}/benchmarks/run_rc426_cuda_tests.py"
test -f "${QUALIFICATION}/benchmarks/run_rc426_cross_device_checkpoint.py"
test -f "${QUALIFICATION}/benchmarks/analyze_rc426_cross_device_checkpoint.py"
test "$(find "${PSSOLVER_RC426_SCRATCH}" -mindepth 1 -maxdepth 1 | wc -l)" -eq 0

export PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1
export NVIDIA_TF32_OVERRIDE=0
unset LD_LIBRARY_PATH
export PYTHONPATH="${QUALIFICATION}"
export CC=/opt/ohpc/pub/compiler/gcc/7.3.0/bin/gcc
export CXX=/opt/ohpc/pub/compiler/gcc/7.3.0/bin/g++
export TRITON_CACHE_DIR="${PSSOLVER_RC426_SCRATCH}/triton_cache"
export TORCHINDUCTOR_CACHE_DIR="${PSSOLVER_RC426_SCRATCH}/torchinductor_cache"
mkdir -p "${TRITON_CACHE_DIR}" "${TORCHINDUCTOR_CACHE_DIR}"

cd "${PSSOLVER_RC426_CONTROL}"

"${PYTHON}" "${QUALIFICATION}/benchmarks/capture_rc4_h100_preflight.py" \
  --q-root "${PSSOLVER_RC426_VENV}" \
  --output "${PSSOLVER_RC426_CONTROL}/preflight/environment.json"

CUDA_TESTS="${QUALIFICATION}/tests"
"${PYTHON}" "${QUALIFICATION}/benchmarks/run_rc426_cuda_tests.py" \
  --node "${CUDA_TESTS}/test_bounded_axis_execution_plan.py::test_cuda_plan_key_uses_the_allocated_device_identity" \
  --node "${CUDA_TESTS}/test_benchmark_bounded_axis_applicability.py::test_cuda_smoke_records_actual_device_and_memory" \
  --node "${CUDA_TESTS}/test_phase4_combined_sbdf2_modal_block.py::test_cuda_canary_binds_concrete_device_and_remains_finite" \
  --node "${CUDA_TESTS}/test_phase8_p845_h100_qualification.py::test_unindexed_cuda_request_binds_lifting_to_allocated_device_identity" \
  --node "${CUDA_TESTS}/test_phase8_p856_finite_q_h100_qualification.py::test_cuda_runtime_matches_cpu_and_binds_allocated_device" \
  --node "${CUDA_TESTS}/test_phase8_p856_finite_q_h100_qualification.py::test_cuda_continuous_split_and_file_restart_are_exact" \
  --node "${CUDA_TESTS}/test_rc4_plane_nyquist_cuda.py::test_plane_nyquist_repair_binds_allocated_cuda_device_and_storage_equivalence" \
  --expected 7 \
  --policy-output "${PSSOLVER_RC426_CONTROL}/preflight/cuda_test_tf32_policy.json" \
  --output "${PSSOLVER_RC426_CONTROL}/cuda_only_tests.json"

"${PYTHON}" "${QUALIFICATION}/benchmarks/run_rc426_cross_device_checkpoint.py" \
  --output "${PSSOLVER_RC426_SCRATCH}/matrix" \
  --expected-purelib "${PURELIB}" \
  --forbidden-source-root "${PSSOLVER_RC426_SOURCE_ROOT}"

"${PYTHON}" "${QUALIFICATION}/benchmarks/analyze_rc426_cross_device_checkpoint.py" \
  --matrix-root "${PSSOLVER_RC426_SCRATCH}/matrix" \
  --output "${PSSOLVER_RC426_CONTROL}/analysis/qualification.json"
