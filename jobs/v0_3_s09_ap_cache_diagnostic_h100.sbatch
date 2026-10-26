#!/usr/bin/env bash
#SBATCH --partition=hagan-gpu
#SBATCH --account=hagan-lab
#SBATCH --qos=medium
#SBATCH --gres=gpu:H100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --no-requeue

set -euo pipefail

: "${S09_DIAG_CONTROL_ROOT:?}"
: "${S09_DIAG_OUTPUT_ROOT:?}"
: "${S09_DIAG_SUPPORT_ROOT:?}"
: "${S09_DIAG_PLAN:?}"
: "${S09_DIAG_BASELINE_ENV:?}"
: "${S09_DIAG_CANDIDATE_ENV:?}"
: "${S09_DIAG_PLAN_SHA256:?}"
: "${S09_DIAG_RUNNER_SHA256:?}"
: "${S09_DIAG_PROFILE_SHA256:?}"
: "${S09_DIAG_CHILD_SHA256:?}"
: "${S09_DIAG_ANALYZER_SHA256:?}"
: "${S09_DIAG_REVERT_PROXY_SHA256:?}"
: "${S09_DIAG_SNAPSHOT_SHA256:?}"

test -d "${S09_DIAG_CONTROL_ROOT}"
test ! -e "${S09_DIAG_OUTPUT_ROOT}"
test -x "${S09_DIAG_BASELINE_ENV}/bin/python"
test -x "${S09_DIAG_CANDIDATE_ENV}/bin/python"
test -f "${S09_DIAG_PLAN}"
test -f "${S09_DIAG_SUPPORT_ROOT}/benchmarks/run_v0_3_s09_ap_cache_diagnostic.py"
test -f "${S09_DIAG_SUPPORT_ROOT}/benchmarks/profile_v0_3_s09_ap_cache_diagnostic.py"
test -f "${S09_DIAG_SUPPORT_ROOT}/benchmarks/run_v0_3_s09_ap_cache_diagnostic_child.py"
test -f "${S09_DIAG_SUPPORT_ROOT}/benchmarks/analyze_v0_3_s09_ap_cache_diagnostic.py"
test -f "${S09_DIAG_SUPPORT_ROOT}/benchmarks/v0_3_s09_revert_pressure_proxy.py"
test -f "${S09_DIAG_SUPPORT_ROOT}/benchmarks/run_v0_3_s09_ap_cache_memory_snapshot.py"

test "$(sha256sum "${S09_DIAG_PLAN}" | awk '{print $1}')" = "${S09_DIAG_PLAN_SHA256}"
test "$(sha256sum "${S09_DIAG_SUPPORT_ROOT}/benchmarks/run_v0_3_s09_ap_cache_diagnostic.py" | awk '{print $1}')" = "${S09_DIAG_RUNNER_SHA256}"
test "$(sha256sum "${S09_DIAG_SUPPORT_ROOT}/benchmarks/profile_v0_3_s09_ap_cache_diagnostic.py" | awk '{print $1}')" = "${S09_DIAG_PROFILE_SHA256}"
test "$(sha256sum "${S09_DIAG_SUPPORT_ROOT}/benchmarks/run_v0_3_s09_ap_cache_diagnostic_child.py" | awk '{print $1}')" = "${S09_DIAG_CHILD_SHA256}"
test "$(sha256sum "${S09_DIAG_SUPPORT_ROOT}/benchmarks/analyze_v0_3_s09_ap_cache_diagnostic.py" | awk '{print $1}')" = "${S09_DIAG_ANALYZER_SHA256}"
test "$(sha256sum "${S09_DIAG_SUPPORT_ROOT}/benchmarks/v0_3_s09_revert_pressure_proxy.py" | awk '{print $1}')" = "${S09_DIAG_REVERT_PROXY_SHA256}"
test "$(sha256sum "${S09_DIAG_SUPPORT_ROOT}/benchmarks/run_v0_3_s09_ap_cache_memory_snapshot.py" | awk '{print $1}')" = "${S09_DIAG_SNAPSHOT_SHA256}"

mkdir -p "${S09_DIAG_OUTPUT_ROOT}"
test -z "$(find "${S09_DIAG_OUTPUT_ROOT}" -mindepth 1 -maxdepth 1 -print -quit)"

export PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1
export NVIDIA_TF32_OVERRIDE=0
export PYTHONPATH="${S09_DIAG_SUPPORT_ROOT}"

nvidia-smi -q > "${S09_DIAG_OUTPUT_ROOT}/nvidia-smi-q.txt"
nvidia-smi --query-gpu=name,uuid,driver_version,memory.total --format=csv,noheader \
  > "${S09_DIAG_OUTPUT_ROOT}/nvidia-smi-summary.txt"
scontrol show job "${SLURM_JOB_ID}" > "${S09_DIAG_OUTPUT_ROOT}/scontrol-running.txt"

"${S09_DIAG_CANDIDATE_ENV}/bin/python" \
  "${S09_DIAG_SUPPORT_ROOT}/benchmarks/run_v0_3_s09_ap_cache_diagnostic.py" \
  --plan "${S09_DIAG_PLAN}" \
  --output-root "${S09_DIAG_OUTPUT_ROOT}/matrix" \
  --baseline-python "${S09_DIAG_BASELINE_ENV}/bin/python" \
  --candidate-python "${S09_DIAG_CANDIDATE_ENV}/bin/python" \
  --support-root "${S09_DIAG_SUPPORT_ROOT}"

mkdir "${S09_DIAG_OUTPUT_ROOT}/memory_snapshots"
for role in candidate_cache_on candidate_revert_proxy; do
  "${S09_DIAG_CANDIDATE_ENV}/bin/python" \
    "${S09_DIAG_SUPPORT_ROOT}/benchmarks/run_v0_3_s09_ap_cache_memory_snapshot.py" \
    --diagnostic-role "${role}" \
    --provider-commit 85ba3aef75548d549d9db453ca4a5ab604b17a9d \
    --provider-wheel-sha256 4003645025da6d81317eeb7261ccf606aef7f60f20d072518f8baacf56cd4aa8 \
    --shape 512,40,40 \
    --lengths 160.0,12.0,12.0 \
    --execution-device cuda \
    --snapshot "${S09_DIAG_OUTPUT_ROOT}/memory_snapshots/${role}.pickle" \
    --output "${S09_DIAG_OUTPUT_ROOT}/memory_snapshots/${role}.json"
done

"${S09_DIAG_CANDIDATE_ENV}/bin/python" \
  "${S09_DIAG_SUPPORT_ROOT}/benchmarks/analyze_v0_3_s09_ap_cache_diagnostic.py" \
  --manifest "${S09_DIAG_OUTPUT_ROOT}/matrix/matrix_manifest.json" \
  --output "${S09_DIAG_OUTPUT_ROOT}/analysis.json"

"${S09_DIAG_CANDIDATE_ENV}/bin/python" - "${S09_DIAG_OUTPUT_ROOT}/analysis.json" <<'PY'
import json
import pathlib
import sys

value = json.loads(pathlib.Path(sys.argv[1]).read_text(encoding="utf-8"))
assert value["classification"] == "S09_AP_CACHE_DIAGNOSTIC_DATA_COMPLETE"
assert value["diagnostic_only"] is True
assert value["qualification_complete"] is False
assert value["s09_complete"] is False
assert value["profile_count"] == 72
assert value["all_profiles_finite"] is True
assert value["all_revert_proxy_proofs_passed"] is True
assert value["candidate_cache_on_revert_proxy_state_byte_identity"] is True
PY

"${S09_DIAG_CANDIDATE_ENV}/bin/python" - "${S09_DIAG_OUTPUT_ROOT}/memory_snapshots" <<'PY'
import json
import pathlib
import sys

root = pathlib.Path(sys.argv[1])
for role in ("candidate_cache_on", "candidate_revert_proxy"):
    value = json.loads((root / f"{role}.json").read_text(encoding="utf-8"))
    assert value["diagnostic_only"] is True
    assert value["included_in_timing_statistics"] is False
    assert value["device"] == "cuda"
    assert value["finite"] is True
    assert value["snapshot"]["taken"] is True
    assert (root / f"{role}.pickle").is_file()
assert json.loads((root / "candidate_revert_proxy.json").read_text(encoding="utf-8"))["revert_proxy"]["proved_revert_proxy"] is True
PY

find "${S09_DIAG_OUTPUT_ROOT}" -type f ! -name checksums.sha256 ! -name COMPLETE \
  -print0 | sort -z | xargs -0 sha256sum > "${S09_DIAG_OUTPUT_ROOT}/checksums.sha256"
sha256sum --check "${S09_DIAG_OUTPUT_ROOT}/checksums.sha256"
cat > "${S09_DIAG_OUTPUT_ROOT}/COMPLETE" <<'EOF'
S09 A-p cache diagnostic evidence is complete.
This marker is not an S-09 qualification PASS and grants no promotion authority.
EOF
