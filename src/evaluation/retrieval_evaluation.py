#!/usr/bin/env python3
"""Compute retrieval metrics for the public APT-RAG result layout."""
from __future__ import annotations

import argparse
import glob
import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple
from urllib.parse import unquote

APT_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = APT_ROOT / "data"
DEFAULT_MONACO_TRACE_DIR = DATA_ROOT / "benchmarks/monaco/execution_traces"
DEFAULT_QAMPARI_GOLD_PATH = DATA_ROOT / "benchmarks/qampari/qampari_full.jsonl"
DEFAULT_METRIC_LEVEL = {
    "monaco": "page_title",
    "qampari": "title_content",
}


def display_path(path: Path) -> str:
    resolved = path.expanduser().resolve()
    try:
        return resolved.relative_to(APT_ROOT).as_posix()
    except ValueError:
        return resolved.as_posix()


def normalize_title(text: str) -> str:
    if not text:
        return ""
    if "wikipedia.org/wiki/" in text:
        text = text.split("/wiki/")[-1]
    elif "/wiki/" in text:
        text = text.split("/wiki/")[-1]
    text = unquote(text).replace("_", " ")
    return " ".join(text.split()).strip().lower()


def normalize_text(text: str) -> str:
    if not text:
        return ""
    return " ".join(text.split()).strip()


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def iter_jsonl(path: Path) -> Iterable[Mapping[str, Any]]:
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def collect_docs_from_obj(obj: Any) -> List[Mapping[str, Any]]:
    docs: List[Mapping[str, Any]] = []
    if isinstance(obj, dict):
        if obj.get("title") and (obj.get("content") or obj.get("text")):
            docs.append(obj)
        for key, value in obj.items():
            if key in {"retrieved", "retrieved_documents", "children", "provenance", "tree", "meta"}:
                docs.extend(collect_docs_from_obj(value))
            elif isinstance(value, (dict, list)):
                docs.extend(collect_docs_from_obj(value))
    elif isinstance(obj, list):
        for item in obj:
            docs.extend(collect_docs_from_obj(item))
    return docs


def load_retrieved_docs_by_ex(answer_dir: Path) -> Dict[int, List[Mapping[str, Any]]]:
    if not answer_dir.is_dir():
        return {}

    results_path = answer_dir / "results.json"
    if results_path.is_file():
        rows = load_json(results_path)
        out: Dict[int, List[Mapping[str, Any]]] = {}
        for row in rows:
            ex_num = row.get("ex_num")
            if ex_num is None:
                continue
            out[int(ex_num)] = list(row.get("retrieved_documents") or [])
        return out

    out: Dict[int, List[Mapping[str, Any]]] = {}
    patterns = [
        str(answer_dir / "ex_*_retrieved.json"),
        str(answer_dir / "execution_traces" / "ex_*.json"),
    ]
    for pattern in patterns:
        for raw_path in glob.glob(pattern):
            path = Path(raw_path)
            match = re.search(r"ex_(\d+)", path.name)
            if not match:
                continue
            out[int(match.group(1))] = collect_docs_from_obj(load_json(path))
    return out


def load_title_content_gold(path: Path) -> Dict[int, Set[Tuple[str, str]]]:
    gold: Dict[int, Set[Tuple[str, str]]] = {}
    for idx, row in enumerate(iter_jsonl(path)):
        ex_num = int(row.get("ex_num", idx))
        items = set()
        for doc in row.get("gold_doc", []) or []:
            if not isinstance(doc, dict):
                continue
            title = normalize_title(doc.get("title", "") or "")
            text = normalize_text(doc.get("content", "") or doc.get("text", "") or "")
            if title and text:
                items.add((title, text))
        gold[ex_num] = items
    return gold


def load_monaco_gold(ex_num: int, trace_dir: Path) -> Set[str]:
    path = trace_dir / f"dataset_ex_{ex_num}.json"
    if not path.is_file():
        return set()
    data = load_json(path)
    question_data = next(iter(data.values()))
    provenance = question_data.get("provenance", {})
    titles: Set[str] = set()
    for step_list in provenance.values():
        if not isinstance(step_list, list):
            continue
        for step in step_list:
            if not isinstance(step, dict):
                continue
            if step.get("answer_in_wiki") == "no" or step.get("answer_in_wiki") is None:
                continue
            for url in step.get("source_url") or []:
                title = normalize_title(url or "")
                if title:
                    titles.add(title)
    return titles


def prf_hit(match_count: int, retrieved_count: int, gold_count: int) -> Dict[str, float]:
    precision = match_count / retrieved_count if retrieved_count else 0.0
    recall = match_count / gold_count if gold_count else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    hit = 1.0 if match_count > 0 else 0.0
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "hit": hit,
        "retrieved_count": retrieved_count,
        "gold_count": gold_count,
        "match_count": match_count,
    }


def page_title_metrics(retrieved_docs: Sequence[Mapping[str, Any]], gold_titles: Set[str]) -> Dict[str, float]:
    retrieved_titles = set()
    for doc in retrieved_docs:
        title = normalize_title(doc.get("url", "") or doc.get("title", "") or "")
        if title:
            retrieved_titles.add(title)
    return prf_hit(len(retrieved_titles & gold_titles), len(retrieved_titles), len(gold_titles))


def title_content_metrics(
    retrieved_docs: Sequence[Mapping[str, Any]],
    gold_set: Set[Tuple[str, str]],
) -> Dict[str, float]:
    retrieved = set()
    for doc in retrieved_docs:
        title = normalize_title(doc.get("title", "") or "")
        text = normalize_text(doc.get("content", "") or doc.get("text", "") or "")
        if title and text:
            retrieved.add((title, text))
    return prf_hit(len(retrieved & gold_set), len(retrieved), len(gold_set))


def macro_average(results: Sequence[Mapping[str, Any]], key: str) -> float:
    if not results:
        return 0.0
    return sum(float(row[key]) for row in results) / len(results)


def evaluate(args: argparse.Namespace) -> Dict[str, Any]:
    benchmark = args.benchmark
    metric_level = args.metric_level or DEFAULT_METRIC_LEVEL[benchmark]
    answer_dir = args.answer_dir.expanduser().resolve()
    docs_by_ex = load_retrieved_docs_by_ex(answer_dir)
    if not docs_by_ex:
        raise FileNotFoundError(f"no retrieval docs found under {answer_dir}")

    per_example = []
    total_retrieved_docs = 0
    min_retrieved_docs: Optional[int] = None
    max_retrieved_docs: Optional[int] = None

    if benchmark == "monaco":
        if metric_level != "page_title":
            raise ValueError("Monaco public reproduction supports metric-level=page_title")
        trace_dir = args.monaco_trace_dir or DEFAULT_MONACO_TRACE_DIR
        for ex_num in sorted(docs_by_ex):
            gold_titles = load_monaco_gold(ex_num, trace_dir)
            if not gold_titles:
                continue
            docs = docs_by_ex[ex_num]
            total_retrieved_docs += len(docs)
            min_retrieved_docs = len(docs) if min_retrieved_docs is None else min(min_retrieved_docs, len(docs))
            max_retrieved_docs = len(docs) if max_retrieved_docs is None else max(max_retrieved_docs, len(docs))
            row = {"ex_num": ex_num, "metric_level": metric_level}
            row.update(page_title_metrics(docs, gold_titles))
            per_example.append(row)
    else:
        if metric_level != "title_content":
            raise ValueError("QAMPARI public reproduction supports metric-level=title_content")
        gold_path = args.qampari_gold_path or args.gold_path or DEFAULT_QAMPARI_GOLD_PATH
        gold_map = load_title_content_gold(gold_path)
        for ex_num in sorted(docs_by_ex):
            gold_set = gold_map.get(ex_num, set())
            if not gold_set:
                continue
            docs = docs_by_ex[ex_num]
            total_retrieved_docs += len(docs)
            min_retrieved_docs = len(docs) if min_retrieved_docs is None else min(min_retrieved_docs, len(docs))
            max_retrieved_docs = len(docs) if max_retrieved_docs is None else max(max_retrieved_docs, len(docs))
            row = {"ex_num": ex_num, "metric_level": metric_level}
            row.update(title_content_metrics(docs, gold_set))
            per_example.append(row)

    if not per_example:
        raise RuntimeError(f"no valid retrieval evaluation targets found under {answer_dir}")

    avg_precision = macro_average(per_example, "precision")
    avg_recall = macro_average(per_example, "recall")
    avg_f1 = macro_average(per_example, "f1")
    return {
        "benchmark": benchmark,
        "answer_dir": display_path(answer_dir),
        "metric_level": metric_level,
        "avg_precision": avg_precision,
        "avg_recall": avg_recall,
        "avg_f1": avg_f1,
        "avg_hit_rate": macro_average(per_example, "hit"),
        "count": len(per_example),
        "avg_retrieved_docs": total_retrieved_docs / len(per_example),
        "min_retrieved_docs": min_retrieved_docs or 0,
        "max_retrieved_docs": max_retrieved_docs or 0,
        "metrics": {
            metric_level: {
                "precision": avg_precision,
                "recall": avg_recall,
                "f1": avg_f1,
                "hit_rate": macro_average(per_example, "hit"),
                "hit_count": int(sum(float(row["hit"]) for row in per_example)),
            }
        },
        "results": per_example,
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", choices=("monaco", "qampari"), required=True)
    parser.add_argument("--answer-dir", type=Path, required=True)
    parser.add_argument("--save-path", type=Path, required=True)
    parser.add_argument("--metric-level", choices=("page_title", "title_content"), default=None)
    parser.add_argument("--monaco-trace-dir", type=Path, default=None)
    parser.add_argument("--qampari-gold-path", type=Path, default=None)
    parser.add_argument("--gold-path", type=Path, default=None)
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    payload = evaluate(args)
    args.save_path.parent.mkdir(parents=True, exist_ok=True)
    args.save_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Evaluated {payload['count']} examples.")
    print(
        f"[{payload['metric_level']}] recall={payload['avg_recall']:.4f} "
        f"precision={payload['avg_precision']:.4f} f1={payload['avg_f1']:.4f}"
    )
    print(f"Saved: {args.save_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
