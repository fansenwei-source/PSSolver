#!/usr/bin/env bash
set -euo pipefail

# Unified controller for the Shendruk et al. Fig. 4 activity-number scan.
# Plane_fig4_benchmark.py runs exactly one A value per process.

FIG4_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FIG4_PROJECT_ROOT="$(cd "${FIG4_SCRIPT_DIR}/.." && pwd)"

FIG4_DEFAULT_PYTHON="/home/fansenwei/anaconda3/envs/Nematics3D/bin/python"
if [[ ! -x "${FIG4_DEFAULT_PYTHON}" ]]; then
    FIG4_DEFAULT_PYTHON="python3"
fi

# Override any setting from the environment, for example:
#   FIG4_ACTIVITY_NUMBERS="16 17 18 19 20" FIG4_DRY_RUN=1 bash scripts_plane/run_fig4_scan.sh
FIG4_PYTHON_BIN="${FIG4_PYTHON_BIN:-${FIG4_DEFAULT_PYTHON}}"
FIG4_ACTIVITY_NUMBERS="${FIG4_ACTIVITY_NUMBERS:-12 15 17 18 19 20 22 25 30 35 40 44.7}"
FIG4_HEIGHT="${FIG4_HEIGHT:-20}"
FIG4_PARAMETERIZATION="${FIG4_PARAMETERIZATION:-paper-window}"
FIG4_FRANK_K="${FIG4_FRANK_K:-0.01}"
FIG4_COEFFICIENT_MIN="${FIG4_COEFFICIENT_MIN:-0.01}"
FIG4_COEFFICIENT_MAX="${FIG4_COEFFICIENT_MAX:-0.05}"
FIG4_LX="${FIG4_LX:-100}"
FIG4_LY="${FIG4_LY:-100}"
FIG4_NX="${FIG4_NX:-256}"
FIG4_NY="${FIG4_NY:-256}"
FIG4_NZ="${FIG4_NZ:-64}"
FIG4_DT="${FIG4_DT:-0.01}"
FIG4_STEPS="${FIG4_STEPS:-10000}"
FIG4_SAVE_START_STEP="${FIG4_SAVE_START_STEP:-5000}"
FIG4_SAVE_INTERVAL="${FIG4_SAVE_INTERVAL:-500}"
FIG4_DIAGNOSTIC_INTERVAL="${FIG4_DIAGNOSTIC_INTERVAL:-100}"
FIG4_SEED="${FIG4_SEED:-24}"
FIG4_LDG_A="${FIG4_LDG_A:-0.0}"
FIG4_LDG_B="${FIG4_LDG_B:--0.3}"
FIG4_LDG_C="${FIG4_LDG_C:-0.3}"
FIG4_GAMMA="${FIG4_GAMMA:-2.94}"
FIG4_FLOW_ALIGNMENT="${FIG4_FLOW_ALIGNMENT:-0.3}"
FIG4_ETA="${FIG4_ETA:-0.6666666666666666}"
FIG4_ZERO_MODE_POLICY="${FIG4_ZERO_MODE_POLICY:-zero_mean}"
FIG4_FRICTION_MODE_FRIC="${FIG4_FRICTION_MODE_FRIC:-0.1}"
FIG4_DEALIAS_RULE="${FIG4_DEALIAS_RULE:-cubic_half}"
FIG4_BETA="${FIG4_BETA:--1.0}"
FIG4_S_INITIAL="${FIG4_S_INITIAL:-0.3333333333333333}"
FIG4_NUM_DEFECT_PAIRS="${FIG4_NUM_DEFECT_PAIRS:-6}"
FIG4_DEFECT_MIN_SEPARATION="${FIG4_DEFECT_MIN_SEPARATION:-10}"
FIG4_DEFECT_CORE_RADIUS="${FIG4_DEFECT_CORE_RADIUS:-1.5}"
FIG4_BACKGROUND_ANGLE="${FIG4_BACKGROUND_ANGLE:-0}"
FIG4_TWIST_AMPLITUDE="${FIG4_TWIST_AMPLITUDE:-0.01}"
FIG4_TWIST_MODES="${FIG4_TWIST_MODES:-1 2 3}"
FIG4_DEVICE="${FIG4_DEVICE:-auto}"
FIG4_DIAGNOSTICS="${FIG4_DIAGNOSTICS:-1}"
FIG4_SAVE_HYDRODYNAMICS="${FIG4_SAVE_HYDRODYNAMICS:-0}"
FIG4_DRY_RUN="${FIG4_DRY_RUN:-0}"
FIG4_SKIP_EXISTING="${FIG4_SKIP_EXISTING:-1}"
FIG4_ANALYZE_AFTER_SCAN="${FIG4_ANALYZE_AFTER_SCAN:-0}"
FIG4_DEFECT_THRESHOLD="${FIG4_DEFECT_THRESHOLD:-0.0}"
FIG4_OUTPUT_ROOT="${FIG4_OUTPUT_ROOT:-${FIG4_PROJECT_ROOT}/data_fig4_H${FIG4_HEIGHT}}"

read -r -a FIG4_A_VALUES <<< "${FIG4_ACTIVITY_NUMBERS}"
if [[ ${#FIG4_A_VALUES[@]} -eq 0 ]]; then
    echo "FIG4_ACTIVITY_NUMBERS contains no values" >&2
    exit 2
fi

mkdir -p "${FIG4_OUTPUT_ROOT}"

FIG4_COMMON_ARGS=(
    --height "${FIG4_HEIGHT}"
    --parameterization "${FIG4_PARAMETERIZATION}"
    --frank-k "${FIG4_FRANK_K}"
    --coefficient-min "${FIG4_COEFFICIENT_MIN}"
    --coefficient-max "${FIG4_COEFFICIENT_MAX}"
    --lx "${FIG4_LX}"
    --ly "${FIG4_LY}"
    --nx "${FIG4_NX}"
    --ny "${FIG4_NY}"
    --nz "${FIG4_NZ}"
    --dt "${FIG4_DT}"
    --steps "${FIG4_STEPS}"
    --save-start-step "${FIG4_SAVE_START_STEP}"
    --save-interval "${FIG4_SAVE_INTERVAL}"
    --diagnostic-interval "${FIG4_DIAGNOSTIC_INTERVAL}"
    --seed "${FIG4_SEED}"
    --ldg-a "${FIG4_LDG_A}"
    --ldg-b "${FIG4_LDG_B}"
    --ldg-c "${FIG4_LDG_C}"
    --gamma "${FIG4_GAMMA}"
    --flow-alignment "${FIG4_FLOW_ALIGNMENT}"
    --eta "${FIG4_ETA}"
    --zero-mode-policy "${FIG4_ZERO_MODE_POLICY}"
    --friction-mode-fric "${FIG4_FRICTION_MODE_FRIC}"
    --dealias-rule "${FIG4_DEALIAS_RULE}"
    --beta "${FIG4_BETA}"
    --num-defect-pairs "${FIG4_NUM_DEFECT_PAIRS}"
    --defect-min-separation "${FIG4_DEFECT_MIN_SEPARATION}"
    --defect-core-radius "${FIG4_DEFECT_CORE_RADIUS}"
    --background-angle "${FIG4_BACKGROUND_ANGLE}"
    --twist-amplitude "${FIG4_TWIST_AMPLITUDE}"
    --device "${FIG4_DEVICE}"
)
FIG4_COMMON_ARGS+=(--initial-s "${FIG4_S_INITIAL}")
read -r -a FIG4_TWIST_MODE_VALUES <<< "${FIG4_TWIST_MODES}"
FIG4_COMMON_ARGS+=(--twist-modes "${FIG4_TWIST_MODE_VALUES[@]}")

if [[ "${FIG4_DIAGNOSTICS}" == "1" ]]; then
    FIG4_COMMON_ARGS+=(--diagnostics)
fi
if [[ "${FIG4_SAVE_HYDRODYNAMICS}" == "1" ]]; then
    FIG4_COMMON_ARGS+=(--save-hydrodynamics)
fi
if [[ "${FIG4_DRY_RUN}" == "1" ]]; then
    FIG4_COMMON_ARGS+=(--dry-run)
fi

for FIG4_A in "${FIG4_A_VALUES[@]}"; do
    FIG4_A_LABEL="${FIG4_A//./p}"
    FIG4_RUN_NAME="H${FIG4_HEIGHT}_A${FIG4_A_LABEL}"
    FIG4_RUN_DIR="${FIG4_OUTPUT_ROOT}/${FIG4_RUN_NAME}"
    FIG4_LOG_PATH="${FIG4_OUTPUT_ROOT}/${FIG4_RUN_NAME}.log"

    if [[ "${FIG4_DRY_RUN}" != "1" && -d "${FIG4_RUN_DIR}" ]] \
        && [[ -n "$(find "${FIG4_RUN_DIR}" -mindepth 1 -maxdepth 1 -print -quit)" ]]; then
        if [[ -f "${FIG4_RUN_DIR}/COMPLETE" && "${FIG4_SKIP_EXISTING}" == "1" ]]; then
            echo "Skipping completed run: ${FIG4_RUN_DIR}"
            continue
        fi
        echo "Refusing to overwrite existing or incomplete run: ${FIG4_RUN_DIR}" >&2
        exit 3
    fi

    FIG4_COMMAND=(
        "${FIG4_PYTHON_BIN}"
        "${FIG4_PROJECT_ROOT}/Plane_fig4_benchmark.py"
        --activity-number "${FIG4_A}"
        --output-dir "${FIG4_RUN_DIR}"
        "${FIG4_COMMON_ARGS[@]}"
    )

    echo "Running ${FIG4_RUN_NAME}"
    printf '  %q' "${FIG4_COMMAND[@]}"
    printf '\n'
    "${FIG4_COMMAND[@]}" 2>&1 | tee "${FIG4_LOG_PATH}"
done

if [[ "${FIG4_DRY_RUN}" != "1" && "${FIG4_ANALYZE_AFTER_SCAN}" == "1" ]]; then
    "${FIG4_PYTHON_BIN}" \
        "${FIG4_PROJECT_ROOT}/scripts_plane/analyze_fig4_sigma.py" \
        --scan-root "${FIG4_OUTPUT_ROOT}" \
        --threshold "${FIG4_DEFECT_THRESHOLD}"
fi
