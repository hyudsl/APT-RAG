import os
import sys
import argparse
from pathlib import Path

BASELINES_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASELINES_ROOT))

from utils.utils import set_seed, load_query_dict
from utils.model_utils import *
from utils.module import Retriever
from module.framework import LogicRAG
from module.module import *


def apply_query_shard(query_dict):
    num_shards = int(os.getenv("LOGIC_RAG_NUM_SHARDS", "1"))
    shard_index = int(os.getenv("LOGIC_RAG_SHARD_INDEX", "0"))

    if num_shards <= 1:
        return query_dict

    if shard_index < 0 or shard_index >= num_shards:
        raise ValueError(
            f"Invalid LOGIC_RAG_SHARD_INDEX={shard_index} for LOGIC_RAG_NUM_SHARDS={num_shards}"
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
        f"[LogicRAG shard] shard {shard_index + 1}/{num_shards} "
        f"selected {len(sharded_items)} of {total_items} examples "
        f"(slice: {start_idx}:{end_idx})"
    )
    return dict(sharded_items)


def main(framework, query_dict, save_dir):

    keys = list(query_dict.keys())
    for key in tqdm(keys):

        query = query_dict[key]['question']
        ex_num = query_dict[key]['ex_num']
        validated_answer = query_dict[key]['validated_answer']
        
        if os.path.exists(f"{save_dir}/execution_traces/ex_{ex_num}.json"):
            continue
    
        start_time = time.perf_counter()
        answer, contexts, rounds, summary_cost, warm_up_cost, dependency_aware_cost, dependency_sorter_cost, g_cost, r_cost = framework.answer_question(query)
        end_time = time.perf_counter()

        result = {
            "ex_num": ex_num,
            "question": query,
            "final answer": answer,
            "contexts": contexts,
            "rounds": rounds,
            "summary_cost": summary_cost,
            "warm_up_cost": warm_up_cost,
            "dependency_aware_cost": dependency_aware_cost,
            "dependency_sorter_cost": dependency_sorter_cost,
            "g_cost": g_cost,
            "r_cost": r_cost,
            "total_latency": end_time - start_time,
        }
        cost_dict = {
            "summary_cost": summary_cost,
            "warm_up_cost": warm_up_cost,
            "dependency_aware_cost": dependency_aware_cost,
            "dependency_sorter_cost": dependency_sorter_cost,
            "g_cost": g_cost,
            "r_cost": r_cost,
            "total_latency": end_time - start_time,
        }
        execution_trace = framework.get_execution_trace(
            ex_num=ex_num,
            query=query,
            validated_answer=validated_answer,
            cost=cost_dict,
        )

        provenance = execution_trace.get("provenance", {}) if isinstance(execution_trace, dict) else {}
        retrieved = {
            "retrieved": {
                node_id: node_info.get("retrieved_documents", [])
                for node_id, node_info in provenance.items()
                if isinstance(node_info, dict)
            }
        }

        save = True
        if save:
            with open(f"{save_dir}/ex_{ex_num}_answer.json", "w", encoding='utf-8') as saver:
                json.dump(result, saver, indent=4, ensure_ascii=False)
            with open(f"{save_dir}/ex_{ex_num}_retrieved.json", "w", encoding='utf-8') as saver:
                json.dump(retrieved, saver, indent=4, ensure_ascii=False)

            os.makedirs(f"{save_dir}/execution_traces", exist_ok=True)
            with open(f"{save_dir}/execution_traces/ex_{ex_num}.json", "w", encoding='utf-8') as saver:
                json.dump(execution_trace, saver, indent=4, ensure_ascii=False)

def parse_args():
    parser = argparse.ArgumentParser(description="Run Logic-RAG baseline.")
    parser.add_argument("--llm-type", "--llm_type", dest="llm_type", default="Qwen/Qwen3-4B-Instruct-2507")
    parser.add_argument("--dataset-name", "--dataset_name", dest="dataset_name", default="monaco")
    parser.add_argument("--sample", default="full")
    parser.add_argument("--answer-dir", "--answer_dir", dest="answer_dir", default=None)
    parser.add_argument("--api-server-url", "--api_server_url", dest="api_server_url", default="http://localhost:8007")
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

    summary_generator = SummaryGenerator(llm_type, tokenizer, model)
    warm_up_analyzer = WarmUpAnalyzer(llm_type, tokenizer, model)
    dependency_analyzer = DependencyAnalyzer(llm_type, tokenizer, model)
    answer_generator = AnswerGenerator(llm_type, tokenizer, model)
    dependency_sorter = DependencySorter(llm_type, tokenizer, model)

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

    framework = LogicRAG(
        summary_generator=summary_generator,
        warm_up_analyzer=warm_up_analyzer,
        dependency_analyzer=dependency_analyzer,
        answer_generator=answer_generator,
        dependency_sorter=dependency_sorter,
        retriever=retriever
    )
    
    main(framework, query_dict, save_dir)
