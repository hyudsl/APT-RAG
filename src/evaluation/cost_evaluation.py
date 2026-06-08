#!/usr/bin/env python3
"""Aggregate per-example cost metrics from execution traces."""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from statistics import mean
from typing import Any, Dict, Iterable, Optional

APT_ROOT = Path(__file__).resolve().parents[2]


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def display_path(path: Path) -> str:
    resolved = path.expanduser().resolve()
    try:
        return resolved.relative_to(APT_ROOT).as_posix()
    except ValueError:
        return resolved.as_posix()


def safe_get(data: Dict[str, Any], key: str, default: float = 0.0) -> float:
    value = data.get(key, default)
    if value is None:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def ex_num_from_path(path: Path) -> Optional[int]:
    match = re.search(r"ex_(\d+)", path.name)
    if not match:
        return None
    return int(match.group(1))


def tree_structure_stats(trace: Dict[str, Any]) -> Dict[str, Any]:
    decomposition = trace.get("decomposition", {})
    if not isinstance(decomposition, dict):
        decomposition = {}

    node_ids = [str(node_id) for node_id in decomposition.keys()]
    node_count = len(node_ids) + 1
    depth = max((node_id.count(".") + 1 for node_id in node_ids), default=0)

    child_counts: Dict[str, int] = {}
    for node_id in node_ids:
        parent_id = node_id.rsplit(".", 1)[0] if "." in node_id else "root"
        child_counts[parent_id] = child_counts.get(parent_id, 0) + 1

    decompose_count = len(child_counts)
    subquestion_count = sum(child_counts.values())
    avg_subquestions_per_decompose = (
        subquestion_count / decompose_count
        if decompose_count
        else None
    )

    return {
        "node_count": node_count,
        "depth": depth,
        "decompose_count": decompose_count,
        "subquestion_count": subquestion_count,
        "avg_subquestions_per_decompose": avg_subquestions_per_decompose,
    }


def node_resolution_stats(trace: Dict[str, Any]) -> Dict[str, Any]:
    provenance = trace.get("provenance", {})
    if not isinstance(provenance, dict):
        provenance = {}

    sibling_count = 0
    direct_count = 0

    for node in provenance.values():
        if not isinstance(node, dict):
            continue

        retrieve_plan = node.get("retrieve_plan", {})
        if not isinstance(retrieve_plan, dict):
            retrieve_plan = {}

        retrieval_decision = retrieve_plan.get("retrieval")
        if retrieval_decision is False:
            sibling_count += 1

        if (
            node.get("plan_type") == "maintain"
            and retrieval_decision is True
        ):
            direct_count += 1

    return {
        "sibling_count": sibling_count,
        "direct_count": direct_count,
    }


def aggregate_trace_cost(trace: Dict[str, Any], ex_num: int) -> Dict[str, Any]:
    cost = trace.get("cost", {})
    latency = cost.get("total_latency_excl_cluster_no_length")
    if latency is None:
        latency = cost.get("total_latency")

    llm_calls = 0.0
    retrieval_calls = None
    input_tokens = 0.0
    output_tokens = 0.0

    for cost_name, cost_dict in cost.items():
        if not isinstance(cost_dict, dict):
            continue

        if cost_name == "r_cost":
            retrieval_calls = safe_get(cost_dict, "call", 0.0)
        else:
            llm_calls += safe_get(cost_dict, "call", 0.0)

        input_tokens += safe_get(cost_dict, "input", 0.0)
        output_tokens += safe_get(cost_dict, "output", 0.0)

    return {
        "ex_num": ex_num,
        "llm_calls": llm_calls,
        "retrieval_calls": retrieval_calls,
        "latency": latency,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        **tree_structure_stats(trace),
        **node_resolution_stats(trace),
    }


def average(rows: Iterable[Dict[str, Any]], key: str) -> Optional[float]:
    values = [row[key] for row in rows if row.get(key) is not None]
    return mean(values) if values else None


def evaluate(answer_dir: Path) -> Dict[str, Any]:
    trace_dir = answer_dir / "execution_traces"
    if not trace_dir.is_dir():
        raise FileNotFoundError(f"missing execution trace directory: {trace_dir}")

    results = []
    for path in sorted(trace_dir.glob("ex_*.json"), key=lambda p: ex_num_from_path(p) or -1):
        ex_num = ex_num_from_path(path)
        if ex_num is None:
            continue
        trace = load_json(path)
        results.append(aggregate_trace_cost(trace, ex_num))

    if not results:
        raise RuntimeError(f"no execution traces found under {trace_dir}")

    return {
        "answer_dir": display_path(answer_dir),
        "trace_dir": display_path(trace_dir),
        "count": len(results),
        "avg_latency": average(results, "latency"),
        "avg_llm_calls": average(results, "llm_calls"),
        "avg_retrieval_calls": average(results, "retrieval_calls"),
        "avg_input_tokens": average(results, "input_tokens"),
        "avg_output_tokens": average(results, "output_tokens"),
        "avg_node_count": average(results, "node_count"),
        "avg_depth": average(results, "depth"),
        "avg_sibling_count": average(results, "sibling_count"),
        "avg_direct_count": average(results, "direct_count"),
        "avg_decompose_count": average(results, "decompose_count"),
        "avg_total_subquestions": average(results, "subquestion_count"),
        "avg_subquestions_per_decompose": average(results, "avg_subquestions_per_decompose"),
        "results": results,
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--answer-dir", type=Path, required=True)
    parser.add_argument("--save-path", type=Path, required=True)
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    payload = evaluate(args.answer_dir.expanduser().resolve())
    args.save_path.parent.mkdir(parents=True, exist_ok=True)
    args.save_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Evaluated {payload['count']} traces.")
    print(
        f"latency={payload['avg_latency']:.4f} "
        f"llm={payload['avg_llm_calls']:.4f} "
        f"retrieval={payload['avg_retrieval_calls']:.4f} "
        f"nodes={payload['avg_node_count']:.4f} "
        f"depth={payload['avg_depth']:.4f} "
        f"sibling={payload['avg_sibling_count']:.4f} "
        f"direct={payload['avg_direct_count']:.4f} "
        f"decompose={payload['avg_decompose_count']:.4f}"
    )
    print(f"Saved: {args.save_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
