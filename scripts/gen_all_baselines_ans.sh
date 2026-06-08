#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
APT_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
SRC_ROOT="$APT_ROOT/src"
RESULTS_ROOT="$APT_ROOT/results"
DATA_ROOT="$APT_ROOT/data"
PYTHON_BIN="${PYTHON_BIN:-python}"
export PYTHONPATH="$SRC_ROOT/baselines:$SRC_ROOT/evaluation:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-3}"

MODELS=(
  "Qwen/Qwen3-4B-Instruct-2507"
  "Qwen/Qwen3-30B-A3B-Instruct-2507"
)

DATASETS=("monaco" "qampari")
METHODS=("llm_only" "naive-rag" "plan-rag" "tree-of-question" "rt-rag" "apt-rag" "logic-rag")
SAMPLE="${SAMPLE:-full}"

selected() {
  local value="$1"
  local filter="$2"
  [[ -z "$filter" ]] && return 0
  IFS=',' read -ra items <<< "$filter"
  for item in "${items[@]}"; do
    item="${item//[[:space:]]/}"
    [[ "$value" == "$item" ]] && return 0
  done
  return 1
}

model_name() {
  local model="$1"
  printf '%s\n' "${model##*/}"
}

script_for_method() {
  case "$1" in
    llm_only) printf '%s\n' "$SRC_ROOT/baselines/llm_only/LLM_only.py" ;;
    naive-rag) printf '%s\n' "$SRC_ROOT/baselines/naive-rag/LLM_DPR.py" ;;
    plan-rag) printf '%s\n' "$SRC_ROOT/baselines/plan-rag/main.py" ;;
    tree-of-question) printf '%s\n' "$SRC_ROOT/baselines/tree-of-question/main.py" ;;
    rt-rag) printf '%s\n' "$SRC_ROOT/baselines/rt-rag/main.py" ;;
    apt-rag) printf '%s\n' "$SRC_ROOT/baselines/apt-rag/main.py" ;;
    logic-rag) printf '%s\n' "$SRC_ROOT/baselines/logic-rag/main.py" ;;
    *) echo "[ERROR] unknown method: $1" >&2; exit 1 ;;
  esac
}

corpus_for_dataset() {
  case "$1" in
    monaco) printf '%s\n' "$DATA_ROOT/artifacts/corpus/monaco/corpus_1M.jsonl" ;;
    qampari) printf '%s\n' "$DATA_ROOT/artifacts/corpus/qampari/corpus.jsonl" ;;
    *) echo "[ERROR] unsupported dataset: $1" >&2; exit 1 ;;
  esac
}

api_for_dataset() {
  case "$1" in
    monaco) printf '%s\n' "${MONACO_API_URL:-http://localhost:8007}" ;;
    qampari) printf '%s\n' "${QAMPARI_API_URL:-http://localhost:8008}" ;;
    *) echo "[ERROR] unsupported dataset: $1" >&2; exit 1 ;;
  esac
}

expand_flag_for_dataset() {
  case "$1" in
    monaco) printf '%s\n' "--expand" ;;
    qampari) printf '%s\n' "--no-expand" ;;
    *) echo "[ERROR] unsupported dataset: $1" >&2; exit 1 ;;
  esac
}

final_top_k_for_method() {
  case "$1" in
    naive-rag|rt-rag) printf '%s\n' "20" ;;
    *) printf '%s\n' "10" ;;
  esac
}

run_job() {
  local label="$1"
  shift
  echo
  echo "============================================================"
  echo ">>> $label"
  echo "============================================================"
  if [[ "${DRY_RUN:-0}" == "1" ]]; then
    printf '[dry-run]'
    printf ' %q' "$PYTHON_BIN" "$@"
    printf '\n'
    return 0
  fi
  "$PYTHON_BIN" "$@"
}

run_basic_job() {
  local dataset="$1"
  local method="$2"
  local llm_type="$3"
  local answer_model_name="$4"
  local answer_dir="$RESULTS_ROOT/$dataset/$method/answer/$answer_model_name"

  run_job "[$dataset][$answer_model_name] $method" \
    "$(script_for_method "$method")" \
    --llm-type "$llm_type" \
    --dataset-name "$dataset" \
    --sample "$SAMPLE" \
    --answer-dir "$answer_dir"
}

run_retrieval_job() {
  local dataset="$1"
  local method="$2"
  local llm_type="$3"
  local answer_model_name="$4"
  local answer_dir="$RESULTS_ROOT/$dataset/$method/answer/$answer_model_name"
  local corpus_path api_url expand_flag final_top_k
  corpus_path="$(corpus_for_dataset "$dataset")"
  api_url="$(api_for_dataset "$dataset")"
  expand_flag="$(expand_flag_for_dataset "$dataset")"
  final_top_k="$(final_top_k_for_method "$method")"

  run_job "[$dataset][$answer_model_name] $method" \
    "$(script_for_method "$method")" \
    --llm-type "$llm_type" \
    --dataset-name "$dataset" \
    --sample "$SAMPLE" \
    --answer-dir "$answer_dir" \
    --corpus-path "$corpus_path" \
    --api-server-url "$api_url" \
    --use-api \
    "$expand_flag" \
    --retrieval-type "dense" \
    --top-k "20" \
    --final-top-k "$final_top_k" \
    --no-post-processing
}

for llm_type in "${MODELS[@]}"; do
  answer_model_name="$(model_name "$llm_type")"
  selected "$answer_model_name" "${MODEL_FILTER:-}" || continue

  for dataset in "${DATASETS[@]}"; do
    selected "$dataset" "${DATASET_FILTER:-}" || continue

    for method in "${METHODS[@]}"; do
      selected "$method" "${METHOD_FILTER:-}" || continue
      case "$method" in
        llm_only) run_basic_job "$dataset" "$method" "$llm_type" "$answer_model_name" ;;
        *) run_retrieval_job "$dataset" "$method" "$llm_type" "$answer_model_name" ;;
      esac
    done
  done
done

echo
echo "All baseline generation runs completed."
