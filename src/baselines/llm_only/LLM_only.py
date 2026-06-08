import argparse
import sys
import os
import json
import time
from pathlib import Path
from tqdm import tqdm

BASELINES_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASELINES_ROOT))
sys.path.insert(0, str(BASELINES_ROOT / "apt-rag"))

from utils.utils import set_seed, load_query_dict
from utils.model_utils import load_model, get_total_usage
from module.module import Generator


def main(query_dict, generator, save, save_path, max_token, llm_type, load=True):
    g_cost = {"call": 0, "input": 0, "output": 0, "latency": 0}

    if save:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

    if load and os.path.exists(save_path):
        with open(save_path, "r", encoding="utf-8") as loader:
            final_result = json.load(loader)
    else:
        final_result = []
    usage_enabled = "gpt" in llm_type.lower()

    keys = list(query_dict.keys())
    preprocessed_ex_nums = [item['ex_num'] for item in final_result]

    for key in tqdm(keys):
        info = query_dict[key]
        ex_num = info['ex_num']
        query = info['question']
        validated_answer = info['validated_answer']

        if ex_num in preprocessed_ex_nums:
            continue

        start_time = time.perf_counter()
        generation_result, g_cost, _ = generator.LLM_Generate(query, g_cost, max_token)
        total_latency = time.perf_counter() - start_time

        final_result.append({
            "ex_num": ex_num,
            "query": query,
            "validated_answer": validated_answer,
            "generation": generation_result,
            "g_cost": g_cost,
            "total_latency": total_latency,
            **{k: v for k, v in info.items() if k not in ["ex_num", "question", "validated_answer", "gold_doc", "decomposition"]}
        })

        if save:
            with open(save_path, "w", encoding="utf-8") as saver:
                json.dump(final_result, saver, indent=4, ensure_ascii=False)

    if save and usage_enabled:
        usage = get_total_usage()
        usage_path = os.path.join(os.path.dirname(save_path), "usage.txt")
        with open(usage_path, "w", encoding="utf-8") as saver:
            saver.write(f"input_tokens: {usage.get('input_tokens', 0)}\n")
            saver.write(f"cached_input_tokens: {usage.get('cached_input_tokens', 0)}\n")
            saver.write(f"output_tokens: {usage.get('output_tokens', 0)}\n")
            saver.write(f"total_tokens: {usage.get('total_tokens', 0)}\n")
            saver.write(f"cost_usd: {usage.get('cost_usd', 0.0)}\n")


def parse_args():
    parser = argparse.ArgumentParser(description="Run LLM_only baseline.")
    parser.add_argument("--llm-type", "--llm_type", dest="llm_type", default="Qwen/Qwen3-4B-Instruct-2507")
    parser.add_argument("--dataset-name", "--dataset_name", dest="dataset_name", default="monaco")
    parser.add_argument("--sample", default="full")
    parser.add_argument("--answer-dir", "--answer_dir", dest="answer_dir", default=None)
    parser.add_argument("--max-token", "--max_token", dest="max_token", type=int, default=3000)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "3")
    set_seed()

    llm_type = args.llm_type
    dataset_name = args.dataset_name.lower()

    save = True
    if args.answer_dir is None:
        raise ValueError("--answer-dir is required")
    save_path = os.path.join(args.answer_dir, "results.json")
    max_token = args.max_token
    
    tokenizer, model = load_model(llm_type)
    generator = Generator(llm_type, tokenizer, model, dataset_name)

    query_dict = load_query_dict(dataset_name, args.sample)

    main(query_dict, generator, save, save_path, max_token, llm_type)
