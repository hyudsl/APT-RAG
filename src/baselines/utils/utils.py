import time
import json
import re
import torch
import os
import random
import numpy as np
import pandas as pd
import requests
from tqdm import tqdm
from itertools import islice
from pathlib import Path
from utils.model_utils import LLM

APT_ROOT = Path(__file__).resolve().parents[3]
DATA_ROOT = APT_ROOT / "data"


def set_seed(seed_value=42):
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)  # if you are using multi-GPU.
    random.seed(seed_value)
    os.environ['PYTHONHASHSEED'] = str(seed_value)
    
    # The below two lines are for deterministic algorithm behavior in CUDA
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_jsonl(file_path, option="dict"):
    """
    option:
      - "dict": return {index: item}
      - "list": return [item, item, ...]
    """
    data = []

    with open(file_path, 'r', encoding='utf-8') as f:
        for line in tqdm(f):
            if not line.strip():
                continue
            data.append(json.loads(line))

    if option == "dict":
        data_dict = {i: item for i, item in enumerate(data)}
        return data_dict

    elif option == "list":
        return data

    else:
        raise ValueError(f"Invalid option: {option}. Use 'dict' or 'list'.")


def cap_retrieved_documents_by_chars(documents, max_chars=None):
    """
    Limit retrieved documents by cumulative content characters.
    Once adding the next document would exceed the cap, that document and all
    following documents are excluded.
    """
    if max_chars is None:
        max_chars = int(os.getenv("RETRIEVED_DOC_CHAR_CAP", "300000"))

    if not documents:
        return []

    capped_documents = []
    total_chars = 0

    for doc in documents:
        content = doc.get("content", "") if isinstance(doc, dict) else ""
        content_chars = len(content or "")
        if total_chars + content_chars > max_chars:
            break
        capped_documents.append(doc)
        total_chars += content_chars

    return capped_documents



# ===== server =====

def check_server_health(api_url):
    try:
        response = requests.get(f"{api_url}/health", timeout=120)
        if response.status_code == 200:
            health_info = response.json()
            index_type = health_info.get('index_type', 'Unknown')
            print(f"{index_type}Store API server connected")
            print(f"   - status: {health_info.get('status')}")
            print(f"   - documents: {health_info.get('document_count'):,}")
            print(f"   - model: {health_info.get('model_name')}")
        else:
            raise Exception(f"Server response error: {response.status_code}")
    except Exception as e:
        raise Exception(f"Store API server connection failed: {e}\nPlease check whether the server is running.")


# ===== cost utils =====

def count_tokens(tokenizer, text: str) -> int:
    return len(tokenizer.encode(text, add_special_tokens=False))



# ===== benchmark utils =====

def load_query_dict(benchmark_name, sample="full"):
    benchmark_key = benchmark_name.lower()
    dataset = {
        "monaco": {
            "full": DATA_ROOT / "benchmarks/monaco/monaco_full.jsonl",
            "sample_300": DATA_ROOT / "benchmarks/monaco/monaco_300.jsonl",
        },
        "qampari": {
            "full": DATA_ROOT / "benchmarks/qampari/qampari_full.jsonl",
            "sample_300": DATA_ROOT / "benchmarks/qampari/qampari_300.jsonl",
        },
    }
    if benchmark_key not in dataset:
        raise ValueError(f"Unsupported benchmark for this repo: {benchmark_name}")
    if sample not in dataset[benchmark_key]:
        raise ValueError(f"Unsupported sample for {benchmark_name}: {sample}")
    return load_jsonl(str(dataset[benchmark_key][sample]))


def update_g_cost(tokenizer, g_cost, sys_prompt, input_prompt, generation_result, elapsed_sec):
    input_tokens = count_tokens(tokenizer, sys_prompt) + count_tokens(tokenizer, input_prompt)
    output_tokens = count_tokens(tokenizer, generation_result)

    return {
        "call": g_cost.get("call", 0) + 1,
        "input": input_tokens + g_cost.get("input", 0),
        "output": output_tokens + g_cost.get("output", 0),
        "latency": elapsed_sec + g_cost.get("latency", 0),
    }, {
        "call": 1,
        "input": input_tokens,
        "output": output_tokens,
        "latency": elapsed_sec,
    }

def run_llm_with_cost(llm_type, tokenizer, model, sys_prompt, input_prompt, g_cost, max_token):
    t0 = time.perf_counter()
    try:
        _, generation_result = LLM(
            sys_prompt,
            input_prompt,
            llm_type,
            max_token,
            tokenizer,
            model,
        )
        elapsed_sec = time.perf_counter() - t0

        updated_cost, updated_cost_individual = update_g_cost(
            tokenizer=tokenizer,
            g_cost=g_cost,
            sys_prompt=sys_prompt,
            input_prompt=input_prompt,
            generation_result=generation_result,
            elapsed_sec=elapsed_sec
        )
        return generation_result, updated_cost, updated_cost_individual

    except Exception as e:
        # Preserve output and latency totals when generation fails.
        input_tokens = count_tokens(tokenizer, sys_prompt) + count_tokens(tokenizer, input_prompt)

        updated_cost = {
            "call": g_cost.get("call", 0) + 1,
            "input": input_tokens + g_cost.get("input", 0),
            "output": g_cost.get("output", 0),
            "latency": g_cost.get("latency", 0),
        }

        return None, updated_cost, updated_cost