import os
import json
import glob
import random
import argparse
from typing import Set, Tuple
from tqdm import tqdm


def _ensure_list(x) -> list:

    """
    Coerce None / scalar / list to a list.
    """

    if x is None:
        return []
    return x if isinstance(x, list) else [x]


def _normalize(s) -> str | None:

    """
    Lowercase and collapse whitespace; return None for empty input.
    """

    if s is None:
        return None
    s = " ".join(str(s).strip().split()).lower()
    return s or None


def collect_provenance(trace_dir: str):
    
    """
    Walk execution-trace JSON files and build three provenance lookups.

    Returns:
        tuples_3:       set of (title, answers_section, context_type)
        title_type_set: set of (title, context_type)  — used for level-1 sections
        gold_sections:  dict (title, context_type) -> set of normalised section names
    """

    tuples_3      = set()
    title_type_set = set()
    gold_sections  = {}

    for path in sorted(glob.glob(os.path.join(trace_dir, "dataset_ex_*.json"))):
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:
            print(f"[WARN] Failed to read {path}: {e}")
            continue

        if not isinstance(data, dict):
            continue

        for payload in data.values():
            if not isinstance(payload, dict):
                continue
            prov = payload.get("provenance", {})
            if not isinstance(prov, dict):
                continue

            for prov_entries in prov.values():
                if not isinstance(prov_entries, list):
                    continue
                for entry in prov_entries:
                    su    = _ensure_list(entry.get("source_url"))
                    sec   = _ensure_list(entry.get("answers_section"))
                    ctype = _ensure_list(entry.get("context_type"))

                    for i in range(max(len(su), len(sec), len(ctype))):
                        source_url      = su[i]    if i < len(su)    else None
                        answers_section = sec[i]   if i < len(sec)   else None
                        context_type    = ctype[i] if i < len(ctype) else None

                        if source_url is None:
                            continue

                        title = source_url.split("/")[-1].replace("_", " ")

                        tuples_3.add((title, answers_section, context_type))
                        title_type_set.add((title, context_type))

                        ns = _normalize(answers_section)
                        if ns:
                            gold_sections.setdefault((title, context_type), set()).add(ns)

    return tuples_3, title_type_set, gold_sections


def extract_heading_names(text: str) -> list[str]:
    
    """
    Return the section titles from the Markdown heading prefix of a chunk.
    """

    if not text or not isinstance(text, str):
        return []

    out = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            break
        if line.startswith("#"):
            title = line.lstrip("#").strip()
            if title:
                out.append(title)
        else:
            break
    return out


def is_gold(item: dict, tuples_3: set, title_type_set: set, gold_sections: dict) -> bool:
    
    """
    Return True if a corpus item matches any provenance entry.
    """

    title        = item.get("title")
    section      = item.get("section")
    content_type = item.get("content_type")

    if item.get("section_level") == 1:
        return (title, content_type) in title_type_set

    if (title, section, content_type) in tuples_3:
        return True

    gs = gold_sections.get((title, content_type), set())
    if gs:
        headings_norm = {_normalize(h) for h in extract_heading_names(item.get("text", "")) if _normalize(h)}
        if headings_norm & gs:
            return True

    return False


def _make_key(item: dict) -> Tuple:
    
    """
    Build a grouping key from an item, stripping sub-chunk suffixes from chunk_index.
    """

    parts = []
    for k in ("title", "section", "content_type", "chunk_index"):
        val = str(item.get(k, ""))
        if k == "chunk_index":
            val = val.split(".")[0]
        parts.append(val)
    return tuple(parts)


def collect_unique_keys(items: list[dict]) -> Set[Tuple]:

    """
    Return the set of unique grouping keys present in items.
    """

    return {_make_key(item) for item in items}


def build_corpus(
    corpus_path: str,
    trace_dir: str,
    output_path: str,
    target_non_gold: int,
) -> None:

    """
    Filter and sample a chunked corpus into a single output file.

    All gold items (matched by provenance traces) are written unconditionally.
    Non-gold items are randomly sampled down to target_non_gold unique document
    groups. The output is written in the original corpus order.
    """

    print("Collecting provenance from trace files...")
    tuples_3, title_type_set, gold_sections = collect_provenance(trace_dir)
    print(f"  Provenance tuples:  {len(tuples_3):,}")
    print(f"  Title/type pairs:   {len(title_type_set):,}")

    print("Partitioning corpus into gold / non-gold...")
    gold_items     = []
    non_gold_items = []

    with open(corpus_path, "r", encoding="utf-8") as f:
        for line in tqdm(f, desc="Reading corpus"):
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            if is_gold(item, tuples_3, title_type_set, gold_sections):
                gold_items.append(item)
            else:
                non_gold_items.append(item)

    print(f"  Gold items:     {len(gold_items):,}")
    print(f"  Non-gold items: {len(non_gold_items):,}")

    print("Sampling non-gold items...")
    all_keys     = collect_unique_keys(non_gold_items)
    sample_size  = min(target_non_gold, len(all_keys))
    sampled_keys = set(random.sample(sorted(all_keys), sample_size))
    sampled_non_gold = [item for item in non_gold_items if _make_key(item) in sampled_keys]
    print(f"  Sampled non-gold items: {len(sampled_non_gold):,} (from {len(all_keys):,} unique groups)")

    print(f"Writing output to {output_path}...")
    total = 0
    with open(output_path, "w", encoding="utf-8") as f:
        for item in gold_items + sampled_non_gold:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
            total += 1

    print(f"Done. Total items written: {total:,}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build a filtered corpus from gold + sampled non-gold items.")
    parser.add_argument("--corpus",      required=True, help="Path to chunked corpus .jsonl")
    parser.add_argument("--trace-dir",   required=True, help="Directory containing dataset_ex_*.json trace files")
    parser.add_argument("--output",      required=True, help="Output path (e.g. corpus_1M.jsonl)")
    parser.add_argument("--target-non-gold", type=int, default=1_000_000, help="Number of unique non-gold document groups to sample (default: 1000000)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    args = parser.parse_args()

    random.seed(args.seed)
    build_corpus(
        corpus_path=args.corpus,
        trace_dir=args.trace_dir,
        output_path=args.output,
        target_non_gold=args.target_non_gold,
    )
