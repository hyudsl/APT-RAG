import argparse
import os
import json
from pathlib import Path
from tqdm import tqdm

from module import LLMJudgeScorerV2
from prompt import SINGLE_ANSWER_LLM_JUDGE, MULTI_ANSWER_LLM_JUDGE
from utils.model_utils import load_model, LLM
from utils.utils import load_json, save_json, set_seed

APT_ROOT = Path(__file__).resolve().parents[4]


def load_existing_results(save_path):
    if not save_path or not os.path.exists(save_path):
        return []

    data = load_json(save_path)
    if isinstance(data, dict):
        results = data.get("results", [])
        if isinstance(results, list):
            return results
    return []


def build_evaluated_query_set(results):
    evaluated = set()
    for item in results:
        evaluated.add(item["query"].strip())
    return evaluated


def resolve_detailed_log_path(eval_llm_type, save_path, detailed_log_path=None):
    if detailed_log_path:
        return detailed_log_path
    if save_path and "gpt" in eval_llm_type.lower():
        return os.path.join(os.path.dirname(save_path), "detailed_calls.jsonl")
    return None


def append_detailed_log(log_path, log_entry):
    if not log_path:
        return

    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    with open(log_path, "a", encoding="utf-8") as saver:
        saver.write(json.dumps(log_entry, ensure_ascii=False) + "\n")


def display_source_file(source_file):
    if not source_file:
        return source_file
    path = Path(source_file).expanduser()
    if not path.is_absolute():
        return path.as_posix()
    resolved = path.resolve()
    try:
        return resolved.relative_to(APT_ROOT).as_posix()
    except ValueError:
        return resolved.as_posix()


def build_detailed_log_entry(eval_llm_type, source_file, ex_num, question, generation, q_label, gold_answers, prompt_text, eval_generation, usage_info, scores_dic):
    return {
        "ex_num": ex_num,
        "source_file": display_source_file(source_file),
        "eval_llm_type": eval_llm_type,
        "query": question,
        "output": generation,
        "query_type": q_label,
        "validated_answer": gold_answers,
        "judge_input": {
            "system_prompt": "",
            "input_prompt": prompt_text,
        },
        "judge_output": eval_generation,
        "usage": usage_info if isinstance(usage_info, dict) else None,
        "scores": scores_dic,
    }


def score_evaluator(eval_llm_type, tokenizer, model, max_tokens, eval_targets, save_path=None, detailed_log_path=None, detailed_log_source=None):
    test_results = []

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        test_results = load_existing_results(save_path)

    evaluated_queries = build_evaluated_query_set(test_results)
    detailed_log_path = resolve_detailed_log_path(eval_llm_type, save_path, detailed_log_path)

    for eval_target in tqdm(eval_targets):
        ex_num = eval_target["ex_num"]
        question = eval_target["query"]
        generation = eval_target["generation"]
        identity = question.strip()

        if identity in evaluated_queries:
            continue

        if "validated_answer" not in eval_target:
            gold_answers = eval_target['info']['validated_answer']
        else:
            gold_answers = eval_target["validated_answer"]

        gold_len = len(gold_answers)

        if gold_len == 1:
            q_label = "single"
            prompt_text = SINGLE_ANSWER_LLM_JUDGE.format(
                question=question,
                response=generation,
                correct_answer=gold_answers,
            )
        else:
            q_label = "multi"
            prompt_text = MULTI_ANSWER_LLM_JUDGE.format(
                question=question,
                response=generation,
                correct_answer=gold_answers,
            )

        usage_info, eval_generation = LLM("", prompt_text, eval_llm_type, max_tokens, tokenizer, model, False, None, 1)

        scorer = LLMJudgeScorerV2(eval_generation, gold_len)
        scores_dic = scorer.Evaluate()

        if scores_dic is None:
            scores_dic = {
                "judge_score": 0.0,
                "precision": 0.0,
                "recall": 0.0,
                "f1": 0.0,
                "error": "Parsing Failed"
            }

        result_entry = {
            "ex_num": ex_num,
            "query": question,
            "output": generation,
            "query_type": q_label,
            "validated_answer": gold_answers,
            "llm_judgement": eval_generation,
            **scores_dic
        }
        test_results.append(result_entry)

        append_detailed_log(detailed_log_path, build_detailed_log_entry(
            eval_llm_type,
            detailed_log_source,
            ex_num,
            question,
            generation,
            q_label,
            gold_answers,
            prompt_text,
            eval_generation,
            usage_info,
            scores_dic,
        ))

        if save_path:
            intermediate_result = calculate_metrics(test_results)
            save_json(save_path, intermediate_result)

    return calculate_metrics(test_results)


def calculate_metrics(results):
    precision_list = [r["precision"] for r in results if "precision" in r]
    recall_list    = [r["recall"]    for r in results if "recall"    in r]
    f1_list        = [r["judge_score"] for r in results if "judge_score" in r]

    avg_precision = sum(precision_list) / len(precision_list) if precision_list else 0.0
    avg_recall    = sum(recall_list)    / len(recall_list)    if recall_list    else 0.0
    avg_f1        = sum(f1_list)        / len(f1_list)        if f1_list        else 0.0
    
    return {
        "avg_precision": avg_precision,
        "avg_recall": avg_recall,
        "avg_f1": avg_f1,
        "results": results
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Run Monaco basic evaluation.")
    parser.add_argument(
        "--eval-llm-type",
        "--eval_llm_type",
        dest="eval_llm_type",
        default="Qwen/Qwen3-30B-A3B-Instruct-2507",
    )
    parser.add_argument(
        "--input-path",
        "--input_path",
        dest="input_path",
        default=None,
    )
    parser.add_argument(
        "--save-path",
        "--save_path",
        dest="save_path",
        default=None,
    )
    parser.add_argument(
        "--max-tokens",
        "--max_tokens",
        dest="max_tokens",
        type=int,
        default=2048,
    )
    parser.add_argument(
        "--detailed-log-path",
        "--detailed_log_path",
        dest="detailed_log_path",
        default=None,
    )
    return parser.parse_args()


if __name__ == "__main__":
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "3")
    set_seed()
    args = parse_args()
    tokenizer, model = load_model(args.eval_llm_type)
    eval_targets = load_json(args.input_path)
    final_result = score_evaluator(
        args.eval_llm_type,
        tokenizer,
        model,
        args.max_tokens,
        eval_targets,
        args.save_path,
        args.detailed_log_path,
        args.input_path,
    )
