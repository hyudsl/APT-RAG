import argparse
import sys, os
import gc
import json
import time
from pathlib import Path

import torch
from tqdm import tqdm

BASELINES_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASELINES_ROOT))

from utils.utils import set_seed, load_query_dict
from utils.model_utils import load_model
from module.framework import APTRAGFramework
from module.tree import TreeNode
from module.module import (
    EvidenceGuidedCluster,
    Generator,
    MemoryController,
    QueryRefiner,
    QueryRewriter,
    QueryStrategyManager,
)
from utils.module import Retriever


def apply_query_shard(query_dict):
    num_shards = int(os.getenv("APT_RAG_NUM_SHARDS", "1"))
    shard_index = int(os.getenv("APT_RAG_SHARD_INDEX", "0"))

    if num_shards <= 1:
        return query_dict

    if shard_index < 0 or shard_index >= num_shards:
        raise ValueError(
            f"Invalid APT_RAG_SHARD_INDEX={shard_index} for APT_RAG_NUM_SHARDS={num_shards}"
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
        f"[APT-RAG shard] shard {shard_index + 1}/{num_shards} "
        f"selected {len(sharded_items)} of {total_items} examples "
        f"(slice: {start_idx}:{end_idx})"
    )
    return dict(sharded_items)


def traverse_and_log(root, ex_num, query, validated_answer, cost_dict, save_path):
    decomposition = {}
    
    provenance = {}
    is_infinite_loop = False
    
    queue = [(root, "0")]
    
    while queue:
        curr_node, curr_id = queue.pop(0)
        
        node_issue = ""
        if curr_node.meta.get("max_depth_prevented"):
            node_issue = "Max depth reached"
            is_infinite_loop = True

        entry = {
            "question": curr_node.question,
            "refined_question": curr_node.refined_question,
            "original_question": curr_node.meta.get("original_question", curr_node.question),
            "rewritten_query": curr_node.query,
            "generation": curr_node.answer,
            "memory": curr_node.memory,
            "context": curr_node.meta.get("input_context", []),
            "plan_type": curr_node.meta.get("plan_type", ""),
            "retrieve_plan": curr_node.meta.get("retrieve_plan", {}),
            "retrieved_documents": curr_node.meta.get("retrieved", []),
            "issue": node_issue
        }
        provenance[curr_id] = entry
        
        for i, child in enumerate(curr_node.children):
            if curr_id == "0":
                child_id = str(i + 1)
            else:
                child_id = f"{curr_id}.{i + 1}"
            
            decomposition[child_id] = child.question
            
            queue.append((child, child_id))

    log_data = {
        "ex_num": ex_num,
        "query": query,
        "validated_answer": validated_answer,
        "generation": root.answer,
        "decomposition": decomposition,
        "issue": "Infinite branching problem" if is_infinite_loop else "",
        "provenance": provenance,
        "cost": cost_dict
    }
    
    os.makedirs(f"{save_path}/execution_traces", exist_ok=True)
    with open(f"{save_path}/execution_traces/ex_{ex_num}.json", "w", encoding='utf-8') as saver:
        json.dump(log_data, saver, indent=4, ensure_ascii=False)


def _json_contains_clustered_key(obj) -> bool:
    if isinstance(obj, dict):
        if "clustered" in obj:
            return True
        return any(_json_contains_clustered_key(v) for v in obj.values())
    if isinstance(obj, list):
        return any(_json_contains_clustered_key(x) for x in obj)
    return False


def _sum_meta_latency_list(node: TreeNode, meta_key: str) -> float:
    """Sum a numeric metadata list over the full tree."""
    total = sum(node.meta.get(meta_key, []))
    for child in node.children:
        total += _sum_meta_latency_list(child, meta_key)
    return total


def main(framework, query_dict, save_dir, max_context_size):

    keys = list(query_dict.keys())
    for key in tqdm(keys):

        query = query_dict[key]['question']
        ex_num = query_dict[key]['ex_num']
        validated_answer = query_dict[key]['validated_answer']
        print(ex_num)

        if os.path.exists(f"{save_dir}/execution_traces/ex_{ex_num}.json"):
            continue


        if torch.cuda.is_available():
            gc.collect()
            torch.cuda.empty_cache()

        start_time = time.perf_counter()
        root = TreeNode(query)
        total_retrieved_call, r_plan_cost, q_plan_cost, q_rewrite_cost, cm_cost, g_cost, r_cost = framework.run(node=root, max_context_size=max_context_size)
        end_time = time.perf_counter()
        total_latency = end_time - start_time
        cluster_no_length_latency = _sum_meta_latency_list(root, "cluster_latency_no_length")
        total_latency_excl_cluster_no_length = total_latency - cluster_no_length_latency

        result = {
                "ex_num": ex_num,
                "question": query,
                "final answer": root.answer,
                "tree": root.to_dict(),
                'retrieved_call': total_retrieved_call,
                'q_plan_cost': q_plan_cost,
                'q_rewrite_cost': q_rewrite_cost,
                "r_plan_cost": r_plan_cost,
                "cm_cost": cm_cost,
                "g_cost": g_cost,
                "r_cost": r_cost,
                'total_latency': total_latency,
                'cluster_no_length_latency': cluster_no_length_latency,
                'total_latency_excl_cluster_no_length': total_latency_excl_cluster_no_length,
        }
        cost_dict = {
                "r_plan_cost": r_plan_cost,
                "q_plan_cost": q_plan_cost,
                "q_rewrite_cost": q_rewrite_cost,
                "cm_cost": cm_cost,
                "g_cost": g_cost,
                "r_cost": r_cost,
                "total_latency": total_latency,
                "cluster_no_length_latency": cluster_no_length_latency,
                "total_latency_excl_cluster_no_length": total_latency_excl_cluster_no_length,
        }

        retrieved = {"retrieved":root.to_retrieved_dict()}


        save = True
        if save:
            with open(f"{save_dir}/ex_{ex_num}_answer.json", "w", encoding='utf-8') as saver:
                json.dump(result, saver, indent=4, ensure_ascii=False)

            with open(f"{save_dir}/ex_{ex_num}_retrieved.json", "w", encoding='utf-8') as saver:
                json.dump(retrieved, saver, indent=4, ensure_ascii=False)
            
            traverse_and_log(root, ex_num, query, validated_answer, cost_dict, save_dir)

def parse_args():
    parser = argparse.ArgumentParser(description="Run APT-RAG baseline.")
    parser.add_argument("--llm-type", "--llm_type", dest="llm_type", default="Qwen/Qwen3-30B-A3B-Instruct-2507")
    parser.add_argument("--dataset-name", "--dataset_name", dest="dataset_name", default="monaco")
    parser.add_argument("--sample", default="full")
    parser.add_argument("--answer-dir", "--answer_dir", dest="answer_dir", default=None)
    parser.add_argument("--api-server-url", "--api_server_url", dest="api_server_url", default="http://localhost:8067")
    parser.add_argument("--corpus-path", "--corpus_path", dest="corpus_path", default=None)
    parser.add_argument("--retrieval-type", "--retrieval_type", dest="retrieval_type", default="dense")
    parser.add_argument("--top-k", "--top_k", dest="top_k", type=int, default=20)
    parser.add_argument("--final-top-k", "--final_top_k", dest="final_top_k", type=int, default=10)
    parser.add_argument("--use-api", "--use_api", dest="use_api", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--expand", dest="expand", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--post-processing", "--post_processing", dest="post_processing", action=argparse.BooleanOptionalAction, default=False)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "3")
    set_seed()

    llm_type = args.llm_type
    tokenizer, model = load_model(llm_type)

    if llm_type == "Qwen/Qwen3-30B-A3B-Instruct-2507":
        max_context_size = 50000
    else:
        max_context_size = 150000

    use_api = args.use_api
    use_local = not use_api
    api_server_url = args.api_server_url
    retrieval_type = args.retrieval_type
    top_k = args.top_k
    final_top_k = args.final_top_k
    expand = args.expand
    post_processing = args.post_processing

    dataset_name = args.dataset_name.lower()
    query_dict = load_query_dict(dataset_name, args.sample)
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

    dpr_path = ""
    bm25_path = ""

    query_refiner = QueryRefiner(llm_type, tokenizer, model)
    q_manager = QueryStrategyManager(llm_type, tokenizer, model)
    q_rewriter = QueryRewriter(llm_type, tokenizer, model)
    generator = Generator(llm_type, tokenizer, model, dataset_name)
    m_controller = MemoryController()
    evidence_clusterer = EvidenceGuidedCluster(llm_type, tokenizer, model)
    
    db_path = None
    if use_local:
        if retrieval_type == "dense":
            db_path = dpr_path
        elif retrieval_type == "bm25":
            db_path = bm25_path

    retriever = Retriever(corpus_path, base_to_offsets, top_k, final_top_k,
        db_path=None, embedding_model=None, tokenizer=tokenizer,
        use_local=use_local, api_url=api_server_url,
        retrieval_type=retrieval_type, expand=expand, post_processing=post_processing
    )
    
    framework = APTRAGFramework(
            q_manager=q_manager,
            q_rewriter=q_rewriter,
            retriever=retriever,
            generator=generator,
            query_refiner=query_refiner,
            m_controller=m_controller,
            reranker=None,
            evidence_clusterer=evidence_clusterer)

    main(framework, query_dict, save_dir, max_context_size)
