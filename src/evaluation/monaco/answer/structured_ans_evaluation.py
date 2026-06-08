import argparse
import os
import glob
import json
from tqdm import tqdm

from basic_ans_evaluation import (
    append_detailed_log,
    build_detailed_log_entry,
    calculate_metrics,
    load_existing_results,
    build_evaluated_query_set,
    resolve_detailed_log_path,
)
from module import LLMJudgeScorerV2
from prompt import SINGLE_ANSWER_LLM_JUDGE, MULTI_ANSWER_LLM_JUDGE
from utils.model_utils import load_model, LLM
from utils.utils import load_json, save_json, set_seed


def tree_score_evaluator(eval_llm_type, tokenizer, model, max_tokens, input_path, save_path=None, limit=None, detailed_log_path=None):    

    test_results = []
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        test_results = load_existing_results(save_path)

    trace_files = sorted(glob.glob(os.path.join(input_path, "ex_*.json")))
    if limit is not None:
        trace_files = trace_files[:limit]
    evaluated_queries = build_evaluated_query_set(test_results)
    detailed_log_path = resolve_detailed_log_path(eval_llm_type, save_path, detailed_log_path)

    for file_path in tqdm(trace_files):

        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        ex_num = data.get("ex_num")
        question = data.get("query")
        generation = data.get("generation")
        gold_answers = data.get("validated_answer")
        identity = question.strip()

        if identity in evaluated_queries:
            continue

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
            file_path,
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


def parse_args():
    parser = argparse.ArgumentParser(description="Run Monaco trace evaluation.")
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
        "--limit",
        dest="limit",
        type=int,
        default=None,
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
    final_result = tree_score_evaluator(
        args.eval_llm_type,
        tokenizer,
        model,
        args.max_tokens,
        args.input_path,
        args.save_path,
        args.limit,
        args.detailed_log_path,
    )
