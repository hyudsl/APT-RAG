#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
APT_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
SRC_ROOT="$APT_ROOT/src"
RESULTS_ROOT="$APT_ROOT/results"
PYTHON_BIN="${PYTHON_BIN:-python}"
export PYTHONPATH="$SRC_ROOT/evaluation:$SRC_ROOT/baselines:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-3}"

MODELS=(
  "Qwen/Qwen3-4B-Instruct-2507"
  "Qwen/Qwen3-30B-A3B-Instruct-2507"
)
DATASETS=("monaco" "qampari")
METHODS=("llm_only" "naive-rag" "plan-rag" "tree-of-question" "rt-rag" "apt-rag" "logic-rag")
MONACO_EVAL_MODELS=("gpt-5.4-2026-03-05")
MONACO_EVAL_MAX_TOKENS="${MONACO_EVAL_MAX_TOKENS:-2048}"

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

is_basic_method() {
  case "$1" in
    llm_only|naive-rag) return 0 ;;
    *) return 1 ;;
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

run_eval() {
  local dataset="$1"
  local method="$2"
  local answer_model_name="$3"
  local eval_model="$4"
  local answer_root="$RESULTS_ROOT/$dataset/$method/answer/$answer_model_name"
  local eval_root="$RESULTS_ROOT/$dataset/$method/eval/$answer_model_name"
  local script_path input_path save_path eval_model_name
  eval_model_name="$(model_name "$eval_model")"

  if [[ "$dataset" == "monaco" ]]; then
    if is_basic_method "$method"; then
      script_path="$SRC_ROOT/evaluation/monaco/answer/basic_ans_evaluation.py"
      input_path="$answer_root/results.json"
    else
      script_path="$SRC_ROOT/evaluation/monaco/answer/structured_ans_evaluation.py"
      input_path="$answer_root/execution_traces"
    fi
    save_path="$eval_root/$eval_model_name/test_results.json"
    [[ -e "$input_path" ]] || { echo "[skip] missing input: $input_path"; return 0; }
    mkdir -p "$(dirname "$save_path")"
    run_job "[$dataset][$answer_model_name][$eval_model_name] $method" \
      "$script_path" \
      --input-path "$input_path" \
      --save-path "$save_path" \
      --eval-llm-type "$eval_model" \
      --max-tokens "$MONACO_EVAL_MAX_TOKENS" \
      --detailed-log-path "$(dirname "$save_path")/detailed_calls.jsonl"
  elif [[ "$dataset" == "qampari" ]]; then
    if is_basic_method "$method"; then
      script_path="$SRC_ROOT/evaluation/qampari/answer/basic_ans_evaluation.py"
      input_path="$answer_root/results.json"
    else
      script_path="$SRC_ROOT/evaluation/qampari/answer/structured_ans_evaluation.py"
      input_path="$answer_root/execution_traces"
    fi
    save_path="$eval_root/test_results.json"
    [[ -e "$input_path" ]] || { echo "[skip] missing input: $input_path"; return 0; }
    mkdir -p "$(dirname "$save_path")"
    run_job "[$dataset][$answer_model_name] $method" \
      "$script_path" \
      --input-path "$input_path" \
      --save-path "$save_path"
  else
    echo "[ERROR] unsupported dataset: $dataset" >&2
    exit 1
  fi
}

for llm_type in "${MODELS[@]}"; do
  answer_model_name="$(model_name "$llm_type")"
  selected "$answer_model_name" "${MODEL_FILTER:-}" || continue

  for dataset in "${DATASETS[@]}"; do
    selected "$dataset" "${DATASET_FILTER:-}" || continue

    for method in "${METHODS[@]}"; do
      selected "$method" "${METHOD_FILTER:-}" || continue
      if [[ "$dataset" == "monaco" ]]; then
        for eval_model in "${MONACO_EVAL_MODELS[@]}"; do
          run_eval "$dataset" "$method" "$answer_model_name" "$eval_model"
        done
      else
        run_eval "$dataset" "$method" "$answer_model_name" ""
      fi
    done
  done
done

echo
echo "All answer evaluation runs completed."
