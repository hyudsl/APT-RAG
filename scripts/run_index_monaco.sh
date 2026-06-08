#!/usr/bin/env bash
set -euo pipefail

# root paths (derived from script location) 
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
APT_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"

readonly PYTHON_BIN="${PYTHON_BIN:-python}"
readonly MAIN_SCRIPT="$APT_ROOT/src/artifacts_construction/vectorDB/indexer/corpus_indexer.py"
readonly CORPUS="$APT_ROOT/data/artifacts/corpus/monaco/corpus_1M.jsonl"
readonly MODEL_NAME="Qwen/Qwen3-Embedding-0.6B"
readonly OUTPUT_DIR_MAIN="$APT_ROOT/data/artifacts/vectorDB/monaco"
readonly OUTPUT_DIR_SMOKE="${OUTPUT_DIR_MAIN}_smoke"

# resource limits 
readonly MEM_MAX_MAIN="220G"
readonly MEM_HIGH_MAIN="190G"
readonly MEM_MAX_SMOKE="64G"
readonly MEM_HIGH_SMOKE="48G"
readonly CPU_WEIGHT="50"
readonly IO_WEIGHT="50"

if [[ $# -lt 1 ]]; then
  cat >&2 <<EOF
[ERROR] gpu_id is required.
Usage:
  $0 <gpu_id>
  $0 <gpu_id> --cold
  $0 <gpu_id> --smoke
  $0 <gpu_id> --smoke --cold
EOF
  exit 1
fi

GPU_ID="$1"
shift

if ! [[ "${GPU_ID}" =~ ^[0-9]+$ ]]; then
  echo "[ERROR] gpu_id must be a non-negative integer (got: '${GPU_ID}')" >&2
  exit 1
fi

COLD_START=0
SMOKE=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --cold)   COLD_START=1; shift ;;
    --smoke)  SMOKE=1;      shift ;;
    --help|-h) sed -n '1,35p' "$0"; exit 0 ;;
    *) echo "[ERROR] Unknown argument: $1" >&2; exit 1 ;;
  esac
done

# mode-specific settings 
if [[ "${SMOKE}" == "1" ]]; then
  UNIT_NAME="indexing-monaco-smoke"
  OUTPUT_DIR="${OUTPUT_DIR_SMOKE}"
  MEM_MAX="${MEM_MAX_SMOKE}"
  MEM_HIGH="${MEM_HIGH_SMOKE}"
  EXTRA_ENV="SMOKE=1"
  MODE_LABEL="SMOKE"
else
  UNIT_NAME="indexing-monaco"
  OUTPUT_DIR="${OUTPUT_DIR_MAIN}"
  MEM_MAX="${MEM_MAX_MAIN}"
  MEM_HIGH="${MEM_HIGH_MAIN}"
  EXTRA_ENV=""
  MODE_LABEL="MAIN"
fi

# stop existing scope if running 
if systemctl --user is-active --quiet "${UNIT_NAME}.scope" 2>/dev/null; then
  echo "[INFO] Stopping existing scope '${UNIT_NAME}.scope'..."
  systemctl --user stop "${UNIT_NAME}.scope" || true
  for _ in 1 2 3 4 5; do
    systemctl --user is-active --quiet "${UNIT_NAME}.scope" 2>/dev/null || break
    sleep 1
  done
fi

# --cold: backup output_dir 
if [[ "${COLD_START}" == "1" ]]; then
  if [[ -e "${OUTPUT_DIR}" ]]; then
    TS="$(date +%Y%m%d_%H%M%S)"
    BACKUP_DIR="${OUTPUT_DIR}_backup_${TS}"
    echo "[INFO] --cold: backing up output_dir"
    echo "       from: ${OUTPUT_DIR}"
    echo "       to  : ${BACKUP_DIR}"
    mv "${OUTPUT_DIR}" "${BACKUP_DIR}"
  else
    echo "[INFO] --cold: no existing output_dir to back up (${OUTPUT_DIR})"
  fi
fi

# already-finished detection 
FINAL_INDEX="${OUTPUT_DIR}/hnsw_sq8/hnsw.index"
FINAL_PKL="${OUTPUT_DIR}/index.pkl"
if [[ -f "${FINAL_INDEX}" ]]; then
  cat <<EOF
[INFO] A completed index already exists.
       final hnsw : ${FINAL_INDEX}
       final pkl  : ${FINAL_PKL}$( [[ -f "${FINAL_PKL}" ]] || echo "  (missing - possible abnormal exit)" )

Options:
  1) Use the existing index as-is (do nothing)
  2) Rebuild from scratch: $0 ${GPU_ID}$( [[ "${SMOKE}" == "1" ]] && echo " --smoke" ) --cold
  3) Change OUTPUT_DIR_MAIN in this script for a different output location
EOF
  exit 0
fi

# resume detection 
if [[ -f "${OUTPUT_DIR}/progress.json" ]] || [[ -f "${OUTPUT_DIR}/hnsw_sq8/hnsw.index.partial" ]]; then
  echo "[INFO] Resume mode: existing checkpoint detected."
  echo "       progress     : ${OUTPUT_DIR}/progress.json"
  echo "       hnsw partial : ${OUTPUT_DIR}/hnsw_sq8/hnsw.index.partial"
else
  echo "[INFO] Fresh start (no checkpoint found)"
fi

# SMOKE_LIMIT passthrough 
if [[ "${SMOKE}" == "1" && -n "${SMOKE_LIMIT:-}" ]]; then
  if ! [[ "${SMOKE_LIMIT}" =~ ^[0-9]+$ ]]; then
    echo "[ERROR] SMOKE_LIMIT must be a non-negative integer (got: '${SMOKE_LIMIT}')" >&2
    exit 1
  fi
  EXTRA_ENV="${EXTRA_ENV} SMOKE_LIMIT=${SMOKE_LIMIT}"
  echo "[INFO] Passing SMOKE_LIMIT=${SMOKE_LIMIT} to Python."
fi

# launch 
cat <<EOF
[LAUNCH] mode        : ${MODE_LABEL}
[LAUNCH] gpu_id      : ${GPU_ID} (CUDA_VISIBLE_DEVICES)
[LAUNCH] unit        : ${UNIT_NAME}.scope
[LAUNCH] corpus      : ${CORPUS}
[LAUNCH] output_dir  : ${OUTPUT_DIR}
[LAUNCH] model       : ${MODEL_NAME}
[LAUNCH] MemoryMax   : ${MEM_MAX}
[LAUNCH] MemoryHigh  : ${MEM_HIGH}
[LAUNCH] CPUWeight   : ${CPU_WEIGHT}
[LAUNCH] IOWeight    : ${IO_WEIGHT}
[LAUNCH] python      : ${PYTHON_BIN}
[LAUNCH] script      : ${MAIN_SCRIPT}
EOF

exec systemd-run --user --scope \
  --unit="${UNIT_NAME}" \
  -p MemoryMax="${MEM_MAX}" \
  -p MemoryHigh="${MEM_HIGH}" \
  -p MemorySwapMax=0 \
  -p CPUWeight="${CPU_WEIGHT}" \
  -p IOWeight="${IO_WEIGHT}" \
  -p TasksMax=8192 \
  nice -n 10 ionice -c 2 -n 7 \
  env \
    CUDA_VISIBLE_DEVICES="${GPU_ID}" \
    PYTHONUNBUFFERED=1 \
    ${EXTRA_ENV} \
    "${PYTHON_BIN}" "${MAIN_SCRIPT}" \
      --corpus     "${CORPUS}" \
      --output-dir "${OUTPUT_DIR}" \
      --model      "${MODEL_NAME}" \
      --gpu-id     0
