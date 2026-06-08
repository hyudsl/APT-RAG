import argparse
import os
import sys
import json
import time
from pathlib import Path
from tqdm import tqdm
from typing import Dict, Any

BASELINES_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASELINES_ROOT))

from utils.utils import *
from utils.model_utils import *
from module.framework import TreeOfQuestion
from module.tree import TreeOfQuestionState
from module.module import *
from utils.module import Retriever


def apply_query_shard(query_dict: Dict[str, Any]) -> Dict[str, Any]:
    num_shards = int(os.getenv("TOQ_NUM_SHARDS", "1"))
    shard_index = int(os.getenv("TOQ_SHARD_INDEX", "0"))

    if num_shards <= 1:
        return query_dict

    if shard_index < 0 or shard_index >= num_shards:
        raise ValueError(
            f"Invalid TOQ_SHARD_INDEX={shard_index} for TOQ_NUM_SHARDS={num_shards}"
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
        f"[ToQ shard] shard {shard_index + 1}/{num_shards} "
        f"selected {len(sharded_items)} of {total_items} examples "
        f"(slice: {start_idx}:{end_idx})"
    )
    return dict(sharded_items)


def traverse_and_log(
    state: TreeOfQuestionState,
    ex_num: int,
    query: str,
    validated_answer: str,
    final_answer: str,
    cost_dict: dict,
    save_dir: str
):
    """Traverse the ToQ tree and write the canonical execution trace."""
    decomposition = {}
    provenance = {}
    is_infinite_loop = False
    
    root = state.root
    
    queue = [(root, "0")]
    
    while queue:
        curr_node, curr_id = queue.pop(0)
        
        node_issue = ""
        if curr_node.level > 4:  # max_depth
            node_issue = "Max depth reached"
            is_infinite_loop = True
        
        entry = {
            "node_id": curr_node.node_id,
            "level": curr_node.level,
            "question": curr_node.question,
            "query": curr_node.query,
            "generation": curr_node.response,
            "answer_span": curr_node.answer_span,
            "is_resolved": curr_node.is_resolved,
            "depends_on": curr_node.depends_on,
            "node_costs": curr_node.node_costs,
            "eval_metrics": curr_node.eval_metrics.to_dict() if curr_node.eval_metrics else None,
            "retrieved_documents": [
                {
                    "title": doc.title,
                    "content": doc.content,
                    "url": doc.url,
                    "doc_id": doc.doc_id,
                    "page_id": doc.page_id,
                    "section": doc.section,
                    "type": doc.type_,
                    "chunk_id": doc.chunk_id,
                    "sub_chunk_id": doc.sub_chunk_id,
                    "total_sub_chunk": doc.total_sub_chunk,
                    "score": doc.score
                }
                for doc in curr_node.documents
            ] if curr_node.documents else [],
            "issue": node_issue
        }
        provenance[curr_id] = entry
        
        for i, child in enumerate(curr_node.children):
            if curr_id == "0":
                child_id = str(i + 1)
            else:
                child_id = f"{curr_id}.{i + 1}"
            
            decomposition[child_id] = {
                "question": child.question,
                "depends_on": child.depends_on
            }
            
            queue.append((child, child_id))
    
    log_data = {
        "ex_num": ex_num,
        "query": query,
        "validated_answer": validated_answer,
        "generation": final_answer,
        "decomposition": decomposition,
        "statistics": {
            "total_nodes": len(state.all_nodes),
            "total_queries": state.total_queries,
            "total_retrievals": state.total_retrievals,
            "resolved_nodes": sum(1 for n in state.all_nodes if n.is_resolved),
            "max_depth_reached": max(n.level for n in state.all_nodes)
        },
        "issue": "Infinite branching problem" if is_infinite_loop else "",
        "provenance": provenance,
        "cost": cost_dict
    }
    
    os.makedirs(f"{save_dir}/execution_traces", exist_ok=True)
    with open(f"{save_dir}/execution_traces/ex_{ex_num}.json", "w", encoding='utf-8') as f:
        json.dump(log_data, f, indent=4, ensure_ascii=False)
    
    return log_data


def main(
    toq_framework: TreeOfQuestion,
    query_dict: Dict[str, Any],
    save_dir: str,
    save: bool = True
):
    # keys = list(query_dict.keys())[51:]
    keys = list(query_dict.keys())

    for key in tqdm(keys, desc="Processing queries"):
        item = query_dict[key]
        query = item['question']
        ex_num = item['ex_num']
        validated_answer = item.get('validated_answer', '')

        if os.path.exists(f"{save_dir}/execution_traces/ex_{ex_num}.json"):
            print(f"skip ex_{ex_num}")
            continue

        start_time = time.perf_counter()
        final_answer, info, state, cost_dict = toq_framework.toq(query)
        end_time = time.perf_counter()

        result = {
            "ex_num": ex_num,
            "query": query,
            "final answer": final_answer,
            "tree": state.root.to_dict(),
            "etc": {
                "total_nodes": info['total_nodes'],
                "total_queries": info['total_queries'],
                "success": True
            }
        }
        retrieved = {"retrieved": state.root.to_retrieved_dict()}
        cost_dict['total_latency'] = end_time - start_time

        if save:
            with open(f"{save_dir}/ex_{ex_num}_answer.json", "w", encoding='utf-8') as saver:
                json.dump(result, saver, indent=4, ensure_ascii=False)

            with open(f"{save_dir}/ex_{ex_num}_retrieved.json", "w", encoding='utf-8') as saver:
                json.dump(retrieved, saver, indent=4, ensure_ascii=False)
            
            traverse_and_log(
                state=state,
                ex_num=ex_num,
                query=query,
                validated_answer=validated_answer,
                final_answer=final_answer,
                cost_dict=cost_dict,
                save_dir=save_dir
            )

def parse_args():
    parser = argparse.ArgumentParser(description="Run Tree_of_Question baseline.")
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

    q_decomposer = QuestionDecomposer(llm_type, tokenizer, model)
    q_generator = QueryGenerator(llm_type, tokenizer, model)
    q_evaluator = QueryEvaluator(llm_type, tokenizer, model)
    a_generator = ResponseGenerator(llm_type, tokenizer, model, dataset_name)
    a_integrator = AnswerIntegrator(llm_type, tokenizer, model)
    
    toq = TreeOfQuestion(
        q_decomposer=q_decomposer,
        q_generator=q_generator,
        q_evaluator=q_evaluator,
        a_generator=a_generator,
        a_integrator=a_integrator,
        retriever=retriever,
        max_depth=4
    )
    
    main(
        toq_framework=toq,
        query_dict=query_dict,
        save_dir=save_dir
    )
