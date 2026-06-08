#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
APT_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"

readonly PYTHON_BIN="${PYTHON_BIN:-python}"
readonly SERVER_SCRIPT="$APT_ROOT/src/artifacts_construction/vectorDB/server/api.py"
readonly VECTORDB_PATH="$APT_ROOT/data/artifacts/vectorDB/monaco"
readonly MODEL_NAME="Qwen/Qwen3-Embedding-0.6B"
readonly INDEX_TYPE="hnsw"
readonly HNSW_SUBDIR="hnsw_sq8"
readonly DEFAULT_PORT=8007
readonly DEFAULT_GPU_ID=1

readonly MEM_MAX="80G"
readonly MEM_HIGH="64G"
readonly CPU_WEIGHT="50"
readonly IO_WEIGHT="50"

if [[ $# -lt 1 ]]; then
  cat >&2 <<EOF
[ERROR] gpu_id is required.
Usage:
  $0 <gpu_id>
  $0 <gpu_id> --port <port>
EOF
  exit 1
fi

GPU_ID="$1"
shift

if ! [[ "${GPU_ID}" =~ ^[0-9]+$ ]]; then
  echo "[ERROR] gpu_id must be a non-negative integer (got: '${GPU_ID}')" >&2
  exit 1
fi

PORT="${DEFAULT_PORT}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --port)
      PORT="$2"
      shift 2
      ;;
    --help|-h) sed -n '1,25p' "$0"; exit 0 ;;
    *) echo "[ERROR] Unknown argument: $1" >&2; exit 1 ;;
  esac
done

readonly UNIT_NAME="vectorstore-monaco"

if systemctl --user is-active --quiet "${UNIT_NAME}.scope" 2>/dev/null; then
  echo "[INFO] Stopping existing scope '${UNIT_NAME}.scope'..."
  systemctl --user stop "${UNIT_NAME}.scope" || true
  for _ in 1 2 3 4 5; do
    systemctl --user is-active --quiet "${UNIT_NAME}.scope" 2>/dev/null || break
    sleep 1
  done
fi

if [[ ! -f "${SERVER_SCRIPT}" ]]; then
  echo "[ERROR] Server script not found: ${SERVER_SCRIPT}" >&2
  exit 1
fi

if [[ ! -d "${VECTORDB_PATH}" ]]; then
  echo "[ERROR] VectorDB directory not found: ${VECTORDB_PATH}" >&2
  echo "        Run run_index_monaco.sh first to build the index." >&2
  exit 1
fi

cat <<EOF
[LAUNCH] gpu_id      : ${GPU_ID}
[LAUNCH] unit        : ${UNIT_NAME}.scope
[LAUNCH] vectordb    : ${VECTORDB_PATH}
[LAUNCH] model       : ${MODEL_NAME}
[LAUNCH] index_type  : ${INDEX_TYPE}
[LAUNCH] hnsw_subdir : ${HNSW_SUBDIR}
[LAUNCH] port        : ${PORT}
[LAUNCH] MemoryMax   : ${MEM_MAX}
[LAUNCH] MemoryHigh  : ${MEM_HIGH}
[LAUNCH] python      : ${PYTHON_BIN}
[LAUNCH] script      : ${SERVER_SCRIPT}
EOF

exec systemd-run --user --scope \
  --unit="${UNIT_NAME}" \
  -p MemoryMax="${MEM_MAX}" \
  -p MemoryHigh="${MEM_HIGH}" \
  -p MemorySwapMax=0 \
  -p CPUWeight="${CPU_WEIGHT}" \
  -p IOWeight="${IO_WEIGHT}" \
  -p TasksMax=8192 \
  env \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH="$APT_ROOT:${PYTHONPATH:-}" \
    "${PYTHON_BIN}" "${SERVER_SCRIPT}" \
      --vectordb-path "${VECTORDB_PATH}" \
      --model         "${MODEL_NAME}" \
      --gpu-id        "${GPU_ID}" \
      --port          "${PORT}" \
      --index-type    "${INDEX_TYPE}" \
      --hnsw-subdir   "${HNSW_SUBDIR}"