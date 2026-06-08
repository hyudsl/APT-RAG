import argparse
import os
import sys
import time
from pathlib import Path

BASELINES_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASELINES_ROOT))

from utils.utils import *
from utils.model_utils import *
from module.module import *
from module.framework import *
from utils.module import *


def build_partial_cost_dict(root, total_latency):
    cost_dict = {"total_latency": total_latency}
    for key in ["q_plan_cost", "rewrite_cost", "g_cost", "r_cost"]:
        value = root.meta.get(key)
        if isinstance(value, dict):
            cost_dict[key] = value
    return cost_dict


def traverse_and_log(root, ex_num, query, validated_answer, cost_dict, save_dir):
    decomposition = {}
    
    provenance = {}
    queue = [root]
    visited = set()

    def query_id_sort_key(query_id):
        if query_id == "Q":
            return (0, 0)
        if query_id.startswith("Q"):
            parts = query_id[1:].split(".")
            major = int(parts[0]) if parts[0] else 0
            minor = int(parts[1]) if len(parts) > 1 else 0
            return (major, minor)
        return (float("inf"), float("inf"))

    while queue:
        curr_node = queue.pop(0)
        node_key = curr_node.query_id

        if node_key in visited:
            continue
        visited.add(node_key)

        issue_messages = []
        for issue_key in ["plan_parse_error", "depth_validation_error", "dag_build_error", "query_id_warning"]:
            if issue_key in curr_node.meta:
                issue_messages.append(f"{issue_key}: {curr_node.meta[issue_key]}")

        entry = {
            "original_query": curr_node.original_query,
            "query": curr_node.query,
            "search_query": curr_node.search_query,
            "generation": curr_node.answer,
            "retrieved_documents": curr_node.search_results or [],
            "q_plan_cost": curr_node.meta.get("q_plan_cost"),
            "rewrite_cost": curr_node.meta.get("rewrite_cost"),
            "g_cost": curr_node.meta.get("g_cost"),
            "r_cost": curr_node.meta.get("r_cost"),
            "issue": "; ".join(issue_messages)
        }
        provenance[node_key] = entry

        children = sorted(curr_node.child_nodes, key=lambda n: query_id_sort_key(n.query_id))
        for child in children:
            child_key = child.query_id
            if child_key not in decomposition:
                decomposition[child_key] = child.original_query
            queue.append(child)

    issue_messages = []
    for issue_key in ["plan_parse_error", "depth_validation_error", "dag_build_error", "run_error"]:
        if issue_key in root.meta:
            issue_messages.append(f"{issue_key}: {root.meta[issue_key]}")

    log_data = {
        "ex_num": ex_num,
        "query": query,
        "validated_answer": validated_answer,
        "generation": root.answer,
        "decomposition": decomposition,
        "issue": "; ".join(issue_messages) if issue_messages else "",
        "provenance": provenance,
        "cost": cost_dict
    }

    os.makedirs(f"{save_dir}/execution_traces", exist_ok=True)
    with open(f"{save_dir}/execution_traces/ex_{ex_num}.json", "w", encoding='utf-8') as saver:
        json.dump(log_data, saver, indent=4, ensure_ascii=False)
    

def main(framework, query_dict, save_dir):
    keys = list(query_dict.keys())

    for key in tqdm(keys):
        query = query_dict[key]['question']
        ex_num = query_dict[key]['ex_num']
        validated_answer = query_dict[key]['validated_answer']
        
        if os.path.exists(f"{save_dir}/execution_traces/ex_{ex_num}.json"):
            continue

        start_time = time.perf_counter()
        root = DAGNode(query)
        try:
            q_plan_cost, rewrite_cost, g_cost, r_cost = framework.plan_star_rag(root)
            end_time = time.perf_counter()
            cost_dict = {
                'q_plan_cost': q_plan_cost,
                'rewrite_cost': rewrite_cost,
                'g_cost': g_cost,
                'r_cost': r_cost,
                'total_latency': end_time - start_time
            }
            traverse_and_log(root, ex_num, query, validated_answer, cost_dict, save_dir)
        except Exception as e:
            end_time = time.perf_counter()
            root.meta["run_error"] = str(e)
            cost_dict = build_partial_cost_dict(root, end_time - start_time)
            traverse_and_log(root, ex_num, query, validated_answer, cost_dict, save_dir)
            continue


def parse_args():
    parser = argparse.ArgumentParser(description="Run Plan_Star_RAG baseline.")
    parser.add_argument("--llm-type", "--llm_type", dest="llm_type", default="Qwen/Qwen3-4B-Instruct-2507")
    parser.add_argument("--dataset-name", "--dataset_name", dest="dataset_name", default="monaco")
    parser.add_argument("--sample", default="sample_100")
    parser.add_argument("--answer-dir", "--answer_dir", dest="answer_dir", default=None)
    parser.add_argument("--api-server-url", "--api_server_url", dest="api_server_url", default="http://localhost:8006")
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

    generator = Generator(llm_type, tokenizer, model, dataset_name.lower())
    retriever = Retriever(corpus_path, base_to_offsets, top_k, final_top_k,
        db_path=None, embedding_model=None, tokenizer=tokenizer,
        use_local=use_local, api_url=api_server_url,
        retrieval_type=retrieval_type, expand=expand, post_processing=post_processing
    )

    framework = Plan_Star_RAG_Framework(
        generator=generator,
        retriever=retriever
    )

    main(framework, query_dict, save_dir)
