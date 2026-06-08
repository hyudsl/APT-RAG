#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
APT_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
SRC_ROOT="$APT_ROOT/src"
RESULTS_ROOT="$APT_ROOT/results"
PYTHON_BIN="${PYTHON_BIN:-python}"
EVAL_PY="$SRC_ROOT/evaluation/cost_evaluation.py"
export PYTHONPATH="$SRC_ROOT/evaluation:$SRC_ROOT/baselines:${PYTHONPATH:-}"

MODELS=(
  "Qwen/Qwen3-4B-Instruct-2507"
  "Qwen/Qwen3-30B-A3B-Instruct-2507"
)
DATASETS=("monaco" "qampari")
METHODS=("plan-rag" "tree-of-question" "rt-rag" "apt-rag" "logic-rag")

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

for llm_type in "${MODELS[@]}"; do
  answer_model_name="$(model_name "$llm_type")"
  selected "$answer_model_name" "${MODEL_FILTER:-}" || continue

  for dataset in "${DATASETS[@]}"; do
    selected "$dataset" "${DATASET_FILTER:-}" || continue

    for method in "${METHODS[@]}"; do
      selected "$method" "${METHOD_FILTER:-}" || continue
      answer_dir="$RESULTS_ROOT/$dataset/$method/answer/$answer_model_name"
      save_path="$RESULTS_ROOT/$dataset/$method/eval/$answer_model_name/cost_results.json"
      if [[ ! -d "$answer_dir/execution_traces" ]]; then
        echo "[skip] missing execution trace directory: $answer_dir/execution_traces"
        continue
      fi
      mkdir -p "$(dirname "$save_path")"
      run_job "[$dataset][$answer_model_name] $method" \
        "$EVAL_PY" \
        --answer-dir "$answer_dir" \
        --save-path "$save_path"
    done
  done
done

echo
echo "All cost evaluation runs completed."
