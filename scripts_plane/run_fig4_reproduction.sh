#!/usr/bin/env bash
set -euo pipefail

# Reproduce the three-height free-slip/free-Q branch of Shendruk et al. Fig. 4
# with zero-mean tangential plug modes and cubic-half spectral dealiasing.
# Each child Python process runs exactly one activity number A.

FIG4_REPRO_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FIG4_REPRO_PROJECT_ROOT="$(cd "${FIG4_REPRO_SCRIPT_DIR}/.." && pwd)"
FIG4_REPRO_SCAN_SCRIPT="${FIG4_REPRO_SCRIPT_DIR}/run_fig4_scan.sh"

FIG4_REPRO_OUTPUT_ROOT="${FIG4_REPRO_OUTPUT_ROOT:-${FIG4_REPRO_PROJECT_ROOT}/data_fig4_free_slip_free_Q_cubic_half_extruded_defects}"
FIG4_REPRO_DRY_RUN="${FIG4_REPRO_DRY_RUN:-0}"
FIG4_REPRO_ANALYZE="${FIG4_REPRO_ANALYZE:-1}"

# Time-budgeted Fig. 4 sampling.  The low/high endpoints constrain each curve,
# while A=17,18,20 retain resolution across the reported crossover near A=18.
# At measured RTX 3060 Ti throughput these 23 runs take approximately 7.7 h.
FIG4_REPRO_A_H10="${FIG4_REPRO_A_H10:-5 10 15 17 18 20 22}"
FIG4_REPRO_A_H15="${FIG4_REPRO_A_H15:-7 12 16 17 18 20 25 33}"
FIG4_REPRO_A_H20="${FIG4_REPRO_A_H20:-9 15 17 18 20 25 35 44.7}"

run_height() {
    local height="$1"
    local nz="$2"
    local activity_numbers="$3"
    local analyze_after="$4"

    FIG4_OUTPUT_ROOT="${FIG4_REPRO_OUTPUT_ROOT}" \
    FIG4_PARAMETERIZATION="paper-window" \
    FIG4_HEIGHT="${height}" \
    FIG4_NZ="${nz}" \
    FIG4_ACTIVITY_NUMBERS="${activity_numbers}" \
    FIG4_DRY_RUN="${FIG4_REPRO_DRY_RUN}" \
    FIG4_ANALYZE_AFTER_SCAN="${analyze_after}" \
    bash "${FIG4_REPRO_SCAN_SCRIPT}"
}

run_height 10 32 "${FIG4_REPRO_A_H10}" 0
run_height 15 48 "${FIG4_REPRO_A_H15}" 0
run_height 20 64 "${FIG4_REPRO_A_H20}" "${FIG4_REPRO_ANALYZE}"
