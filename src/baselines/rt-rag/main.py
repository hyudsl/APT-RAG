import argparse
import os
import sys
import json
import time
from pathlib import Path

# GPU must be set before any torch/cuda import (used by shared modules)
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "3")

from tqdm import tqdm

BASELINES_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASELINES_ROOT))

from utils.utils import load_query_dict, set_seed
from utils.model_utils import load_model
from utils.module import Retriever
from rt_utils import get_logger
from module.tree import *
from module.module import *
from module.framework import *
from config import *


def apply_query_shard(query_dict):
    num_shards = int(os.getenv("RT_RAG_NUM_SHARDS", "1"))
    shard_index = int(os.getenv("RT_RAG_SHARD_INDEX", "0"))

    if num_shards <= 1:
        return query_dict

    if shard_index < 0 or shard_index >= num_shards:
        raise ValueError(
            f"Invalid RT_RAG_SHARD_INDEX={shard_index} for RT_RAG_NUM_SHARDS={num_shards}"
        )

    # Shard by ex_num so parallel workers use a stable partition.
    items = sorted(
        query_dict.items(),
        key=lambda item: item[1].get("ex_num", 0)
    )
    total_items = len(items)
    start_idx = total_items * shard_index // num_shards
    end_idx = total_items * (shard_index + 1) // num_shards
    sharded_items = items[start_idx:end_idx]

    print(
        f"[RT-RAG shard] shard {shard_index + 1}/{num_shards} "
        f"selected {len(sharded_items)} of {total_items} examples "
        f"(slice: {start_idx}:{end_idx})"
    )
    return dict(sharded_items)


def apply_query_ex_nums(query_dict, ex_nums_text):
    if not ex_nums_text:
        return query_dict

    ex_nums = []
    for raw_ex_num in ex_nums_text.split(","):
        raw_ex_num = raw_ex_num.strip()
        if not raw_ex_num:
            continue
        ex_nums.append(int(raw_ex_num))

    if not ex_nums:
        return query_dict

    items_by_ex_num = {}
    for key, item in query_dict.items():
        items_by_ex_num[int(item["ex_num"])] = (key, item)

    selected_items = []
    missing_ex_nums = []
    for ex_num in ex_nums:
        if ex_num in items_by_ex_num:
            selected_items.append(items_by_ex_num[ex_num])
        else:
            missing_ex_nums.append(ex_num)

    if missing_ex_nums:
        print(f"[WARN] RT-RAG ex_nums not found in sample: {missing_ex_nums}")

    print(
        f"[RT-RAG ex_nums] selected {len(selected_items)} of {len(query_dict)} examples"
    )
    return dict(selected_items)


def traverse_and_log(root, ex_num, query, validated_answer, save_path, cost_dict=None):
    """
    BFS traverse the binary tree and build provenance from node.meta (RT_RAG framework).
    Saves to save_path/execution_traces/ex_{ex_num}.json
    """
    decomposition = {}
    provenance = {}
    issue = ""

    # BFS with node IDs: root="0", left="1", right="2", then "1.1", "1.2", "2.1", "2.2", ...
    queue = [(root, "0")]
    while queue:
        curr_node, curr_id = queue.pop(0)
        display_q = getattr(curr_node, "display_question", None) or curr_node.question

        # Base entry from framework meta; override with canonical query/generation
        # Base entry for this node; meta already includes detailed fields
        entry = {
            "query": display_q,
            "generation": curr_node.answer,
            "plan_type": curr_node.type or "",
            "depends_on": getattr(curr_node, "depends_on", None),
            # Flattened retrieval list for convenience (also inside meta["retrieved"])
            "retrieved_documents": curr_node.meta.get("retrieved", []),
            # Full metadata including structure, decomposition, iterative logs, etc.
            "meta": dict(curr_node.meta),
        }
        
        provenance[curr_id] = entry

        # Enqueue left then right (Monaco-style: first child "1", second "2")
        if curr_node.left:
            left_id = f"{curr_id}.1" if curr_id != "0" else "1"
            decomposition[left_id] = getattr(curr_node.left, "display_question", None) or curr_node.left.question
            queue.append((curr_node.left, left_id))
        if curr_node.right:
            right_id = f"{curr_id}.2" if curr_id != "0" else "2"
            decomposition[right_id] = getattr(curr_node.right, "display_question", None) or curr_node.right.question
            queue.append((curr_node.right, right_id))

    log_data = {
        "ex_num": ex_num,
        "query": query,
        "validated_answer": validated_answer,
        "generation": root.answer,
        "decomposition": decomposition,
        "issue": issue,
        "provenance": provenance,
    }
    if cost_dict is not None:
        log_data["cost"] = cost_dict
    os.makedirs(f"{save_path}/execution_traces", exist_ok=True)
    with open(f"{save_path}/execution_traces/ex_{ex_num}.json", "w", encoding="utf-8") as f:
        json.dump(log_data, f, indent=4, ensure_ascii=False)


def main(query_dict, framework, save_dir):
    """
    Main function that directly calls the question decomposition and answering functionality
    """


    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(f"{save_dir}/execution_traces", exist_ok=True)


    keys = list(query_dict.keys())
    for key in tqdm(keys):

        query = query_dict[key]['question']
        ex_num = query_dict[key]['ex_num']


        if os.path.exists(f"{save_dir}/execution_traces/ex_{ex_num}.json"):
            continue

        validated_answer = query_dict[key]['validated_answer']
        cost = {
            "analyze_q_str_cost"  : {"call": 0, "input": 0, "output": 0, "latency": 0},
            "gen_q_var_cost"      : {"call": 0, "input": 0, "output": 0, "latency": 0},
            "decomp_cost"         : {"call": 0, "input": 0, "output": 0, "latency": 0},
            "answer_cost"         : {"call": 0, "input": 0, "output": 0, "latency": 0},
            "refine_cost"         : {"call": 0, "input": 0, "output": 0, "latency": 0},
            "gen_right_q_cost"    : {"call": 0, "input": 0, "output": 0, "latency": 0},
            "answer_reason_cost"  : {"call": 0, "input": 0, "output": 0, "latency": 0},
            "final_ans_gen_cost"  : {"call": 0, "input": 0, "output": 0, "latency": 0},
            "r_cost"              : {"call": 0, "embed_latency": 0, "search_latency": 0}
        }
        stats_file_path = None
        logger = get_logger()
        start_time = time.perf_counter()
        answer, cost, tree_root = framework.decompose_and_answer_with_variants(
            question=query,
            cost=cost,
            trees_per_question=TREES_PER_QUESTION,
            max_tokens=9000,
            temperature=0,
            top_p=1.0,
            frequency_penalty=0.0,
            presence_penalty=0.0,
            num_examples=25,
            max_height=MAX_HEIGHT,
            enhanced_right_subtree=ENHANCED_RIGHT_SUBTREE,
            right_subtree_variants=RIGHT_SUBTREE_VARIANTS,
            right_subtree_trees_per_variant=RIGHT_SUBTREE_TREES_PER_VARIANT,
            max_variants=0,
            stats_file_path=stats_file_path,
            logger=logger,
        )
        total_latency = time.perf_counter() - start_time
        cost["total_latency"] = total_latency

        # Ensure tree_root is never None so downstream consumers (e.g., to_dict) are safe
        if tree_root is None:
            # Minimal fallback tree node with the final answer attached
            tree_root = QuestionNode(
                question=query,
                q_type="None",
                subq1=query,
                subq2="",
            )
            tree_root.answer = answer
            tree_root.meta["fallback"] = True
            

        logger.debug(f"\nFinal answer: {answer}")

        result = {
            "ex_num": ex_num,
            "question": query,
            "validated_answer": validated_answer,
            "generation": answer,
            "cost": cost,
            "cost_trace": getattr(framework, "latest_cost_trace", []),
            "trees": getattr(framework, "latest_tree_bundle", {}),
            "tree": tree_root.to_dict(),
        }

        save = True
        if save:
            with open(f"{save_dir}/ex_{ex_num}_answer.json", "w", encoding="utf-8") as saver:
                json.dump(result, saver, indent=4, ensure_ascii=False)
            traverse_and_log(tree_root, ex_num, query, validated_answer, save_dir, cost)


def parse_args():
    parser = argparse.ArgumentParser(description="Run RT_RAG baseline.")
    parser.add_argument("--llm-type", "--llm_type", dest="llm_type", default="Qwen/Qwen3-4B-Instruct-2507")
    parser.add_argument("--dataset-name", "--dataset_name", dest="dataset_name", default="monaco")
    parser.add_argument("--sample", default="sample_100")
    parser.add_argument("--answer-dir", "--answer_dir", dest="answer_dir", default=None)
    parser.add_argument("--api-server-url", "--api_server_url", dest="api_server_url", default="http://localhost:8006")
    parser.add_argument("--corpus-path", "--corpus_path", dest="corpus_path", default=None)
    parser.add_argument("--retrieval-type", "--retrieval_type", dest="retrieval_type", default="dense")
    parser.add_argument("--top-k", "--top_k", dest="top_k", type=int, default=20)
    parser.add_argument("--final-top-k", "--final_top_k", dest="final_top_k", type=int, default=20)
    parser.add_argument("--use-api", "--use_api", dest="use_api", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--expand", dest="expand", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--post-processing", "--post_processing", dest="post_processing", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--ex-nums", "--ex_nums", dest="ex_nums", default=os.getenv("RT_RAG_EX_NUMS", ""))
    return parser.parse_args()


if __name__ == "__main__": 
    args = parse_args()
    set_seed()

    llm_type = args.llm_type
    re_llm_type = "Qwen3-Reranker-0.6B" 
    embedding_model_name = "Qwen/Qwen3-Embedding-4B"
    tokenizer, model = load_model(llm_type)


    use_api = args.use_api
    use_local = not use_api
    api_server_url = args.api_server_url
    expand = args.expand
    post_processing = args.post_processing
    top_k = args.top_k
    final_top_k = args.final_top_k
    retrieval_type = args.retrieval_type

    dataset_name = args.dataset_name.lower()
    query_dict = load_query_dict(dataset_name, args.sample)
    if args.ex_nums:
        query_dict = apply_query_ex_nums(query_dict, args.ex_nums)
    else:
        query_dict = apply_query_shard(query_dict)
    
    if args.answer_dir is None:
        raise ValueError("--answer-dir is required")
    save_dir = args.answer_dir
    os.makedirs(save_dir, exist_ok=True)

    if args.corpus_path is None:
        raise ValueError("--corpus-path is required")
    corpus_path = args.corpus_path

    if expand:
        base_to_offsets, docid_to_suffix = Retriever.build_inverted_index(corpus_path)
    else:
        base_to_offsets, docid_to_suffix = {}, {}

    retriever = Retriever(corpus_path, base_to_offsets, top_k, final_top_k,
        db_path=None, embedding_model=None, tokenizer=tokenizer,
        use_local=use_local, api_url=api_server_url,
        retrieval_type=retrieval_type, expand=expand, post_processing=post_processing
    )

    q_struc_analyzer = QuestionStructureAnalyzer(llm_type, tokenizer, model)
    q_variant_generator = QuestionVariantGenerator(llm_type, tokenizer, model)
    similar_ex_finder = SimilarExamplesFinder()
    decomposer = Decomposer(llm_type, tokenizer, model)
    iter_ans_generator = IterAnswerGenerator(llm_type, tokenizer, model, retriever)
    generator = Generator(llm_type, tokenizer, model)

    framework = RTRAGFramework(
        q_struc_analyzer=q_struc_analyzer,
        q_variant_generator=q_variant_generator,
        retriever=retriever,
        similar_ex_finder=similar_ex_finder,
        decomposer=decomposer,
        iter_ans_generator=iter_ans_generator,
        generator=generator,
    )

    main(query_dict, framework, save_dir)   
