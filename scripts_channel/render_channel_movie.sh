#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
VIS_SCRIPT="${REPO_ROOT}/visualize_nematics3d_snapshot_channel.py"

DATA_DIR="${DATA_DIR:-${REPO_ROOT}/data_control_box_Nx256_Lx64}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${DATA_DIR}/nematics3d_channel_movie}"
PYTHON_BIN="${PYTHON_BIN:-/home/fansenwei/anaconda3/envs/Nematics3D/bin/python}"
FPS="${FPS:-12}"
VIDEO_NAME="${VIDEO_NAME:-nematics3d_channel_defects_velocity.mp4}"
CONTROL_HISTORY="${CONTROL_HISTORY:-${DATA_DIR}/control_history.csv}"
FONT_FILE="${FONT_FILE:-/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf}"
USE_XVFB="${USE_XVFB:-1}"
OVERWRITE="${OVERWRITE:-0}"
DRY_RUN="${DRY_RUN:-0}"
START_INDEX="${START_INDEX:-}"
END_INDEX="${END_INDEX:-}"
STEP_MOD="${STEP_MOD:-10}"

DATA_DIR="$(realpath -m "${DATA_DIR}")"
OUTPUT_ROOT="$(realpath -m "${OUTPUT_ROOT}")"

SNAPSHOT_DIR="${OUTPUT_ROOT}/snapshot_images"
FRAME_DIR="${OUTPUT_ROOT}/frames"
LOG_DIR="${OUTPUT_ROOT}/logs"
VIDEO_PATH="${OUTPUT_ROOT}/${VIDEO_NAME}"
VIS_BACKUP=""
INDEX_LIST="${OUTPUT_ROOT}/snapshot_indices.txt"
CONCAT_LIST="${OUTPUT_ROOT}/frames.txt"

usage() {
  cat <<'USAGE'
Render every saved PSSolver/data snapshot with visualize_nematics3d_snapshot_channel.py.
For each frame, this script rewrites SNAPSHOT_INDEX in that Python file, runs it,
combines the defect-line and velocity-field images, then encodes a video.

Environment variables:
  DATA_DIR      Snapshot folder with Q_<index>.npy and u_<index>.npy.
                Default: PSSolver/data
  OUTPUT_ROOT   Output folder. Default: DATA_DIR/nematics3d_channel_movie
  PYTHON_BIN    Python with Nematics3D installed.
                Default: /home/fansenwei/anaconda3/envs/Nematics3D/bin/python
  FPS           Output video frame rate. Default: 12
  VIDEO_NAME    Output mp4 filename. Default: nematics3d_channel_defects_velocity.mp4
  CONTROL_HISTORY
                Optional control-history CSV used to label step, defects, alpha,
                and control state. Default: DATA_DIR/control_history.csv
  USE_XVFB      1 to render through xvfb-run for headless VTK/PyVista. Default: 1
  OVERWRITE     1 to delete old PNG frames before rendering. Default: 0
  DRY_RUN       1 to only scan snapshots and print the planned output path.
  START_INDEX   Optional first snapshot index to render.
  END_INDEX     Optional last snapshot index to render.
  STEP_MOD      Snapshot-index modulus filter. Default: 10, matching the current
                data cadence and rendering all 1000 saved snapshots.

Examples:
  bash scripts_channel/render_channel_movie.sh
  START_INDEX=0 END_INDEX=500 STEP_MOD=50 FPS=8 bash scripts_channel/render_channel_movie.sh
  PYTHON_BIN=python OUTPUT_ROOT=data/movie_test OVERWRITE=1 bash scripts_channel/render_channel_movie.sh
USAGE
}

die() {
  printf 'error: %s\n' "$*" >&2
  exit 1
}

require_command() {
  command -v "$1" >/dev/null 2>&1 || die "required command not found: $1"
}

numeric_or_empty() {
  local value="$1"
  local name="$2"
  [[ -z "${value}" || "${value}" =~ ^[0-9]+$ ]] || die "${name} must be a non-negative integer"
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

[[ -f "${VIS_SCRIPT}" ]] || die "missing visualization script: ${VIS_SCRIPT}"
[[ -d "${DATA_DIR}" ]] || die "missing data directory: ${DATA_DIR}"
[[ -x "${PYTHON_BIN}" ]] || die "PYTHON_BIN is not executable: ${PYTHON_BIN}"
[[ "${FPS}" =~ ^[0-9]+([.][0-9]+)?$ ]] || die "FPS must be numeric"
[[ "${OVERWRITE}" == "0" || "${OVERWRITE}" == "1" ]] || die "OVERWRITE must be 0 or 1"
[[ "${DRY_RUN}" == "0" || "${DRY_RUN}" == "1" ]] || die "DRY_RUN must be 0 or 1"
[[ "${USE_XVFB}" == "0" || "${USE_XVFB}" == "1" ]] || die "USE_XVFB must be 0 or 1"
numeric_or_empty "${START_INDEX}" "START_INDEX"
numeric_or_empty "${END_INDEX}" "END_INDEX"
numeric_or_empty "${STEP_MOD}" "STEP_MOD"
if [[ -n "${START_INDEX}" && -n "${END_INDEX}" && "${START_INDEX}" -gt "${END_INDEX}" ]]; then
  die "START_INDEX cannot be greater than END_INDEX"
fi
if [[ -n "${STEP_MOD}" && "${STEP_MOD}" -lt 1 ]]; then
  die "STEP_MOD must be >= 1"
fi

require_command find
require_command awk
require_command sort
require_command sed
require_command ffmpeg
require_command mktemp
if [[ "${USE_XVFB}" == "1" ]]; then
  require_command xvfb-run
fi

mkdir -p "${SNAPSHOT_DIR}" "${FRAME_DIR}" "${LOG_DIR}"
if [[ "${OVERWRITE}" == "1" ]]; then
  find "${SNAPSHOT_DIR}" -maxdepth 1 -type f -name 'Q_*_*.png' -delete
  find "${FRAME_DIR}" -maxdepth 1 -type f -name 'frame_*.png' -delete
fi

find "${DATA_DIR}" -maxdepth 1 -type f -name 'Q_*.npy' -printf '%f\n' \
  | sed -E 's/^Q_([0-9]+)\.npy$/\1/' \
  | sort -n \
  | awk -v data_dir="${DATA_DIR}" \
        -v start="${START_INDEX}" \
        -v end="${END_INDEX}" \
        -v step_mod="${STEP_MOD}" '
      {
        idx = $1 + 0
        if (start != "" && idx < start) next
        if (end != "" && idx > end) next
        if (step_mod != "" && idx % step_mod != 0) next
        u_path = data_dir "/u_" idx ".npy"
        if ($1 ~ /^[0-9]+$/ && system("[ -f \"" u_path "\" ]") == 0) print idx
      }
    ' > "${INDEX_LIST}"

snapshot_count="$(wc -l < "${INDEX_LIST}" | tr -d ' ')"
[[ "${snapshot_count}" -gt 0 ]] || die "no matched Q/u snapshots found in ${DATA_DIR}"

printf 'data_dir=%s\n' "${DATA_DIR}"
printf 'output_root=%s\n' "${OUTPUT_ROOT}"
printf 'snapshots=%s\n' "${snapshot_count}"
printf 'fps=%s\n' "${FPS}"
if [[ "${DRY_RUN}" == "1" ]]; then
  printf 'first_indices='
  sed -n '1,10p' "${INDEX_LIST}" | paste -sd, -
  printf 'video=%s\n' "${VIDEO_PATH}"
  exit 0
fi

VIS_BACKUP="$(mktemp "${OUTPUT_ROOT}/visualize_nematics3d_snapshot_channel.py.backup.XXXXXX")"
cp "${VIS_SCRIPT}" "${VIS_BACKUP}"
cleanup() {
  if [[ -n "${VIS_BACKUP}" && -f "${VIS_BACKUP}" ]]; then
    cp "${VIS_BACKUP}" "${VIS_SCRIPT}"
    rm -f "${VIS_BACKUP}"
  fi
}
trap cleanup EXIT

sed -i -E \
  -e "s#^DATA_DIR = .*#DATA_DIR = Path(r\"${DATA_DIR}\")#" \
  -e "s#^OUTPUT_DIR = .*#OUTPUT_DIR = Path(r\"${SNAPSHOT_DIR}\")#" \
  -e "s#^SKIP_EMPTY_DEFECT_FIGURE = .*#SKIP_EMPTY_DEFECT_FIGURE = False#" \
  "${VIS_SCRIPT}"

frame_number=0
while IFS= read -r -u 3 index; do
  if [[ ! "${index}" =~ ^[0-9]+$ ]]; then
    die "internal error: invalid snapshot index '${index}' in ${INDEX_LIST}"
  fi
  frame_number=$((frame_number + 1))
  defects_png="${SNAPSHOT_DIR}/Q_${index}_defects_lines.png"
  velocity_png="${SNAPSHOT_DIR}/Q_${index}_velocity_vectors.png"
  frame_png="${FRAME_DIR}/frame_$(printf '%06d' "${frame_number}").png"
  log_file="${LOG_DIR}/snapshot_${index}.log"

  if [[ ! -s "${defects_png}" || ! -s "${velocity_png}" ]]; then
    sed -i -E "s#^SNAPSHOT_INDEX = .*#SNAPSHOT_INDEX = ${index}#" "${VIS_SCRIPT}"
    printf '[%s/%s] rendering snapshot %s\n' "${frame_number}" "${snapshot_count}" "${index}"
    if [[ "${USE_XVFB}" == "1" ]]; then
      xvfb-run -a "${PYTHON_BIN}" "${VIS_SCRIPT}" >"${log_file}" 2>&1 </dev/null
    else
      "${PYTHON_BIN}" "${VIS_SCRIPT}" >"${log_file}" 2>&1 </dev/null
    fi
  else
    printf '[%s/%s] reusing snapshot %s\n' "${frame_number}" "${snapshot_count}" "${index}"
  fi

  [[ -s "${defects_png}" ]] || die "missing defect-line image after rendering snapshot ${index}; see ${log_file}"
  [[ -s "${velocity_png}" ]] || die "missing velocity image after rendering snapshot ${index}; see ${log_file}"

  if [[ ! -s "${frame_png}" || "${OVERWRITE}" == "1" ]]; then
    control_label="step=${index}"
    if [[ -f "${CONTROL_HISTORY}" ]]; then
      control_row="$(
        awk -F, -v idx="${index}" '
          NR == 1 {
            is_box = ($2 == "global_defect_points")
            next
          }
          NR > 1 && ($1 + 0) == idx {
            if (is_box) {
              printf "%s|%s|%s|%s", $2, $3, $5, $6
            } else {
              printf "%s|%s|%s", $2, $4, $5
            }
            exit
          }
        ' "${CONTROL_HISTORY}"
      )"
      if [[ -n "${control_row}" ]]; then
        pipe_count="${control_row//[^|]/}"
        if [[ "${#pipe_count}" -eq 3 ]]; then
          IFS='|' read -r global_defects box_defects alpha_value control_state <<< "${control_row}"
          control_label="step=${index}  global=${global_defects}  box=${box_defects}  alpha_box=${alpha_value}  state=${control_state}"
        else
          IFS='|' read -r defect_points alpha_value control_state <<< "${control_row}"
          control_label="step=${index}  defects=${defect_points}  alpha=${alpha_value}  state=${control_state}"
        fi
      fi
    fi
    ffmpeg -hide_banner -loglevel error -y \
      -i "${defects_png}" \
      -i "${velocity_png}" \
      -filter_complex "[0:v]scale=1920:-2[top];[1:v]scale=1920:-2[bottom];[top][bottom]vstack=inputs=2,drawtext=fontfile=${FONT_FILE}:text='${control_label}':x=35:y=30:fontsize=34:fontcolor=black:box=1:boxcolor=white@0.82:boxborderw=12,format=rgb24" \
      "${frame_png}" </dev/null
  fi
done 3< "${INDEX_LIST}"

find "${FRAME_DIR}" -maxdepth 1 -type f -name 'frame_*.png' -printf '%f\n' \
  | sort \
  | awk -v frame_dir="${FRAME_DIR}" "{print \"file '\" frame_dir \"/\" \$0 \"'\"}" > "${CONCAT_LIST}"

frame_count="$(wc -l < "${CONCAT_LIST}" | tr -d ' ')"
[[ "${frame_count}" -eq "${snapshot_count}" ]] || die "expected ${snapshot_count} frames, found ${frame_count}"

ffmpeg -hide_banner -loglevel error -y \
  -r "${FPS}" \
  -f concat \
  -safe 0 \
  -i "${CONCAT_LIST}" \
  -vf "pad=ceil(iw/2)*2:ceil(ih/2)*2" \
  -c:v libx264 \
  -pix_fmt yuv420p \
  -movflags +faststart \
  "${VIDEO_PATH}" </dev/null

printf 'video=%s\n' "${VIDEO_PATH}"
