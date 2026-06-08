import re
import json
import time
from json_repair import repair_json
from itertools import combinations
from prompt.prompt import (
    ANSWERABILITY_CHECK_PROMPT,
    CONTEXTUALIZATION_PROMPT,
    DECOMPOSITION_PROMPT,
    EVIDENCE_ANSWER_SYS_PROMPT,
    EXTERNAL_ANSWER_PROMPT,
    FINAL_ANSWER_PROMPT,
    FINAL_ANSWER_SYS_PROMPT,
    LATERAL_ANSWER_PROMPT,
    Q_REWRITE_MULTI,
    Q_REWRITE_SINGLE,
    VERTICAL_ANSWER_PROMPT,
    VERTICAL_ANSWER_SYS_PROMPT,
    build_evidence_cluster_prompt,
)
from utils.utils import count_tokens
from utils.model_utils import LLM


def _count_tokens(tokenizer, text: str) -> int:
    return len(tokenizer.encode(text, add_special_tokens=False))

def extract_json_block(defulat_fallback, text: str): 
    start = text.find("{")
    if start == -1:
        return defulat_fallback.copy()

    brace_count = 0
    for i in range(start, len(text)):
        if text[i] == "{":
            brace_count += 1
        elif text[i] == "}":
            brace_count -= 1
            if brace_count == 0:
                json_str = text[start:i + 1]
                try:
                    parsed = json.loads(json_str)
                except json.JSONDecodeError:
                    parsed = repair_json(json_str, return_objects=True)
                if isinstance(parsed, dict):
                    result = defulat_fallback.copy()
                    result.update(parsed)
                    return result

    return defulat_fallback.copy()


def update_g_cost(tokenizer, g_cost, sys_prompt, input_prompt, generation_result, elapsed_sec):
    input_tokens = _count_tokens(tokenizer, sys_prompt) + _count_tokens(tokenizer, input_prompt)
    output_tokens = _count_tokens(tokenizer, generation_result)

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

# -------------------- Module for query processing -------------------- 

class QueryStrategyManager:
    def __init__(self, llm_type, tokenizer, model):
        self.llm_type = llm_type
        self.tokenizer = tokenizer
        self.model = model
    

    def answerability_check(self, query, context, r_plan_cost, max_token):
        system_prompt = "You are an answerability checker. Decide whether the question requires external search according to the provided instructions."
        input_prompt = ANSWERABILITY_CHECK_PROMPT.format(query=query)
        
        for key, value in context.items():
            if "Q" in key:
                input_prompt += f"\n\tQ:{value}"
            if "A" in key:
                input_prompt += f"\n\tA:{value}"
        input_prompt += "\n\noutput:"

        DEFAULT_FALLBACK = {
            "step 1": "",
            "step 2": "",
            "step 3": "",
            "step 4": False,
        }

        generation_result, r_plan_cost, r_plan_cost_individual = run_llm_with_cost(
            llm_type=self.llm_type, 
            tokenizer=self.tokenizer,
            model=self.model,
            sys_prompt=system_prompt,
            input_prompt=input_prompt,
            g_cost=r_plan_cost,
            max_token=max_token,
        )
        json_result = extract_json_block(DEFAULT_FALLBACK, generation_result)
        retrieval = bool(json_result.get("step 4", False))
        return {"retrieval": retrieval}, json_result, r_plan_cost, r_plan_cost_individual

    
    def decomposition_planner(self, query, q_plan_cost, max_token):
        system_prompt = "You are a planner that creates query plans for the given question. Make appropriate decisions according to the provided instructions."
        input_prompt = DECOMPOSITION_PROMPT.format(query=query)
        
        DEFAULT_FALLBACK = {
            "decision": "maintain",
            "subqueries": [],
        }

        generation_result, q_plan_cost, q_plan_cost_individual = run_llm_with_cost(
            llm_type=self.llm_type, 
            tokenizer=self.tokenizer,
            model=self.model,
            sys_prompt=system_prompt,
            input_prompt=input_prompt,
            g_cost=q_plan_cost,
            max_token=max_token,
        )
        json_result = extract_json_block(DEFAULT_FALLBACK, generation_result)
        return json_result, q_plan_cost, q_plan_cost_individual

# -------------------- Query Rewriter --------------------

class QueryRewriter:
    def __init__(self, llm_type, tokenizer, model):
        self.llm_type = llm_type
        self.tokenizer = tokenizer
        self.model = model

    def Rewrite_single(self, question, q_rewrite_cost, max_token):
        generation_result, q_rewrite_cost, q_rewrite_cost_individual = run_llm_with_cost(
            llm_type=self.llm_type,
            tokenizer=self.tokenizer,
            model=self.model,
            sys_prompt="",
            input_prompt=Q_REWRITE_SINGLE.format(question=question),
            g_cost=q_rewrite_cost,
            max_token=max_token,
        )
        rewritten_query = (generation_result or question).replace("Query: ", "").strip()
        return rewritten_query, q_rewrite_cost, q_rewrite_cost_individual

    def Rewrite_multi(self, questions, q_idx_list, q_rewrite_cost, max_token):
        questions_str = "\n".join(questions)
        generation_result, q_rewrite_cost, q_rewrite_cost_individual = run_llm_with_cost(
            llm_type=self.llm_type,
            tokenizer=self.tokenizer,
            model=self.model,
            sys_prompt="",
            input_prompt=Q_REWRITE_MULTI.format(questions=questions_str),
            g_cost=q_rewrite_cost,
            max_token=max_token,
        )

        DEFAULT_FALLBACK = {"raw": generation_result, **{(str(i)): "" for i in q_idx_list}}
        rewritten_query = extract_json_block(DEFAULT_FALLBACK, generation_result)

        for i in q_idx_list:
            if str(i) not in rewritten_query.keys():
                rewritten_query[str(i)] = ""
            else:
                rewritten_query[str(i)] = rewritten_query[str(i)].replace("Query: ", "").strip()
            
        return rewritten_query, q_rewrite_cost, q_rewrite_cost_individual


# -------------------- Query Refiner --------------------
class QueryRefiner: 
    Q_TAG_RE = re.compile(r"<Q(\d+)>")

    def __init__(self, llm_type, tokenizer, model):
        self.llm_type = llm_type
        self.tokenizer= tokenizer
        self.model = model
    
    def context_specify(self, query, context, cm_cost, max_token):
        input_prompt = CONTEXTUALIZATION_PROMPT.format(t_query=query)
        for key, value in context.items():
            if "Q" in key:
                input_prompt += f"\n\tQ:{value}"
            if "A" in key:
                input_prompt += f"\n\tA:{value}"
        input_prompt += "\n\noutput:"

        DEFAULT_FALLBACK = {
            "step 1": "",
            "step 2": "",
            "step 3": [],
            "step 4": query
        }

        generation_result, cm_cost, cm_cost_individual = run_llm_with_cost(
            llm_type=self.llm_type, 
            tokenizer=self.tokenizer,
            model=self.model,
            sys_prompt="",
            input_prompt=input_prompt,
            g_cost=cm_cost,
            max_token=max_token,
        )
        json_result = extract_json_block(DEFAULT_FALLBACK, generation_result)
        return json_result, cm_cost, cm_cost_individual


# -------------------- Generator -------------------- 



class Generator:  
    def __init__(self, llm_type, tokenizer, model, dataset_name):
        self.llm_type = llm_type
        self.tokenizer= tokenizer
        self.model = model
        self.dataset_name = dataset_name

    def _format_documents(self, documents):
        formatted = []
        for i, doc in enumerate(documents, 1):
            formatted.append(f"[doc {i}] {doc['title']}\n{doc['content']}")
        return "\n\n".join(formatted)

    def Retrieval_Generate(self, question, query, documents, g_cost, max_token):
        docs_text = self._format_documents(documents)
        input_prompt = EXTERNAL_ANSWER_PROMPT.format(question=question, query=query, documents=docs_text)

        return run_llm_with_cost(
            llm_type=self.llm_type, 
            tokenizer=self.tokenizer,
            model=self.model,
            sys_prompt=FINAL_ANSWER_SYS_PROMPT[self.dataset_name],
            input_prompt=input_prompt,
            g_cost=g_cost,
            max_token=max_token,
        )

    def generate_external_answer(self, question, query, documents, g_cost, max_token): 
        docs_text = self._format_documents(documents)
        input_prompt = EXTERNAL_ANSWER_PROMPT.format(question=question, query=query, documents=docs_text)

        return run_llm_with_cost(
            llm_type=self.llm_type, 
            tokenizer=self.tokenizer,
            model=self.model,
            sys_prompt=EVIDENCE_ANSWER_SYS_PROMPT,
            input_prompt=input_prompt,
            g_cost=g_cost,
            max_token=max_token,
        )

    def generate_lateral_answer(self, query, context, g_cost, max_token):
        input_prompt = f"Question: {query}\n\n"

        sq_keys = sorted([k for k in context.keys() if k.startswith('SQ')], key=lambda x: int(x[2:]))
        
        if sq_keys:
            input_prompt += "QA pairs generated along the path leading to the current question:\n"
            for sq_key in sq_keys:
                sa_key = sq_key.replace('SQ', 'SA')
                num = sq_key[2:]

                sq = context.get(sq_key, '').replace('<seq>', '').strip()
                sa = context.get(sa_key, '-')
                
                input_prompt += f"Q{num}: {sq}\n"
                input_prompt += f"A{num}: {sa}\n\n"
        
            input_prompt += LATERAL_ANSWER_PROMPT
        input_prompt += "\nAnswer:"

        return run_llm_with_cost(
            llm_type=self.llm_type, 
            tokenizer=self.tokenizer,
            model=self.model,
            sys_prompt=EVIDENCE_ANSWER_SYS_PROMPT,
            input_prompt=input_prompt,
            g_cost=g_cost,
            max_token=max_token,
        )
        
    def generate_vertical_answer(self, query, memory, g_cost, max_token):
        input_prompt = f"Current Question {query}\n\nsub-QA pairs for the current question:\n"

        sq_keys = sorted([k for k in memory.keys() if k.startswith('SQ')], key=lambda x: int(x[2:]))

        for sq_key in sq_keys:
            sa_key = sq_key.replace('SQ', 'SA')
            num = sq_key[2:]

            sq = memory.get(sq_key, '').replace('<seq>', '').strip()
            sa = memory.get(sa_key, '-')
            
            input_prompt += f"Q{num}: {sq}\n"
            input_prompt += f"A{num}: {sa}\n\n"

        input_prompt += VERTICAL_ANSWER_PROMPT
        input_prompt += "\nAnswer:"

        return run_llm_with_cost(
            llm_type=self.llm_type, 
            tokenizer=self.tokenizer,
            model=self.model,
            sys_prompt=VERTICAL_ANSWER_SYS_PROMPT,
            input_prompt=input_prompt,
            g_cost=g_cost,
            max_token=max_token,
        )

    def generate_final_answer(self, query, memory, g_cost, max_token):
        input_prompt = f"Current Question {query}\n\nsub-QA pairs for the current question:\n"

        sq_keys = sorted([k for k in memory.keys() if k.startswith('SQ')], key=lambda x: int(x[2:]))

        for sq_key in sq_keys:
            sa_key = sq_key.replace('SQ', 'SA')
            num = sq_key[2:]

            sq = memory.get(sq_key, '').replace('<seq>', '').strip()
            sa = memory.get(sa_key, '-')
            
            input_prompt += f"Q{num}: {sq}\n"
            input_prompt += f"A{num}: {sa}\n\n"

        input_prompt += FINAL_ANSWER_PROMPT[self.dataset_name]
        input_prompt += "\nAnswer:"

        return run_llm_with_cost(
            llm_type=self.llm_type, 
            tokenizer=self.tokenizer,
            model=self.model,
            sys_prompt=FINAL_ANSWER_SYS_PROMPT[self.dataset_name],
            input_prompt=input_prompt,
            g_cost=g_cost,
            max_token=max_token,
        )
    
# -------------------- Memory controller -------------------- 
class MemoryController:  
    def build_memory(self, query, sub_queries): 
        memory = {}
        memory['RQ'] = query
        memory['RA'] = "-"
        for k in range(len(sub_queries)):
            memory[f"SQ{k+1}"] = sub_queries[k]
            memory[f'SA{k+1}'] = "-"
        return memory
    
    def context_extract(self, memory, key_lst):
        context = {}
        for key_num in key_lst:
            context[f"SQ{key_num}"] = memory.get(f"SQ{key_num}", "")
            context[f"SA{key_num}"] = memory.get(f"SA{key_num}", "")

        return context
    
### Evidence-guided clustering module

class EvidenceGuidedCluster:
    def __init__(
        self,
        llm_type,
        tokenizer,
        model,
        sim_threshold: float = 0.0,
        max_group_size=None,
    ):
        self.llm_type       = llm_type
        self.tokenizer      = tokenizer
        self.model          = model
        self.sim_threshold  = sim_threshold
        self.max_group_size = max_group_size

        self.children       = None
        self.retrieved_list = None

    # -----------------------------
    # Evidence similarity
    # -----------------------------
    def to_pair_set(self, retrieved):
        result = set()
        for item in retrieved:
            if not isinstance(item, dict):
                continue
            title   = str(item.get("title",   "")).strip()
            content = str(item.get("content", "")).strip()
            result.add((title, content))
        return result

    def jaccard_similarity(self, retrieved_1, retrieved_2):
        set1  = self.to_pair_set(retrieved_1)
        set2  = self.to_pair_set(retrieved_2)
        union = set1 | set2
        if not union:
            return 0.0
        return len(set1 & set2) / len(union)

    def build_similarity_matrix(self, children, retrieved_list):
        n          = len(children)
        sim_matrix = [[0.0] * n for _ in range(n)]
        for i in range(n):
            sim_matrix[i][i] = 1.0
        for i in range(n):
            for j in range(i + 1, n):
                sim = self.jaccard_similarity(retrieved_list[i], retrieved_list[j])
                sim_matrix[i][j] = sim
                sim_matrix[j][i] = sim
        return sim_matrix

    def _get_score(self, group, sim_matrix):
        if len(group) <= 1:
            return 0.0
        return sum(sim_matrix[i][j] for i, j in combinations(group, 2))

    # -----------------------------
    # Cluster token counting (union-based)
    # -----------------------------
    def _union_prompt_tokens(self, children, safe_retrieved, group: tuple) -> int:
        """
        Count the LLM input tokens for a cluster using the union of its evidence.
        build_evidence_cluster_prompt deduplicates shared documents within the group.
        """
        sys_tokens  = count_tokens(self.tokenizer, EVIDENCE_ANSWER_SYS_PROMPT)
        user_prompt = build_evidence_cluster_prompt(children, safe_retrieved, list(group))
        return sys_tokens + count_tokens(self.tokenizer, user_prompt)

    # -----------------------------
    # Minimum-Group-Count Clustering
    #
    # Objective: minimize |G|, the number of LLM calls.
    # Constraints: pairwise evidence similarity and optional context length.
    #   ∀ i,j ∈ G_k : Jaccard(i,j) > S
    #        prompt_len(∪docs in G_k) < C       (context constraint, optional)
    #
    # Greedy places the least compatible nodes first.
    # -----------------------------
    def _greedy_cluster(self, n: int, order: list, is_valid) -> list:
        """First-Fit greedy clustering with a caller-supplied validity predicate."""
        cluster_lists: list[list[int]] = []
        for idx in order:
            placed = False
            for cl in cluster_lists:
                candidate = tuple(sorted(cl + [idx]))
                if is_valid(candidate):
                    cl.append(idx)
                    placed = True
                    break
            if not placed:
                cluster_lists.append([idx])
        result = [tuple(sorted(g)) for g in cluster_lists]
        result.sort(key=lambda g: (-len(g), g[0]))
        return result

    def cluster(self, children, retrieved_list, context_window_size=183296, sim_threshold=None):
        """
        min |G|  subject to
          (1) ∀ i,j ∈ G_k : Jaccard(i,j) > sim_threshold
          (2) prompt_len( ∪ docs in G_k ) < context_window_size  (with_length only)

        Returns dict:
          - with_length: clusters used for inference
          - no_length: clusters using only the similarity constraint
          - latency_no_length: no_length greedy latency
          - latency_with_length: with_length greedy latency, including token counting
        Shared preparation is treated as part of the constrained path and is not
        included in either isolated latency measurement.
        """
        if sim_threshold is None:
            sim_threshold = self.sim_threshold

        n = len(children)
        if n <= 1:
            single = [(0,)] if n == 1 else []
            return {
                "with_length": single,
                "no_length": single,
                "latency_no_length": 0.0,
                "latency_with_length": 0.0,
            }

        safe_retrieved = [r if r else [] for r in retrieved_list]

        sim_matrix = self.build_similarity_matrix(children, safe_retrieved)

        compat = {
            (i, j): sim_matrix[i][j] > sim_threshold
            for i in range(n) for j in range(i + 1, n)
        }

        def _sim_ok(group: tuple) -> bool:
            if len(group) <= 1:
                return True
            return all(
                compat[(min(i, j), max(i, j))]
                for i, j in combinations(group, 2)
            )

        def is_valid_no_length(group: tuple) -> bool:
            return _sim_ok(group)

        def is_valid_with_length(group: tuple) -> bool:
            if not _sim_ok(group):
                return False
            return self._union_prompt_tokens(children, safe_retrieved, group) < context_window_size

        incompat_degree = [
            sum(1 for j in range(n) if i != j and not compat[(min(i, j), max(i, j))])
            for i in range(n)
        ]
        order = sorted(range(n), key=lambda i: -incompat_degree[i])

        t0 = time.perf_counter()
        no_length = self._greedy_cluster(n, order, is_valid_no_length)
        latency_no_length = time.perf_counter() - t0

        t0 = time.perf_counter()
        with_length = self._greedy_cluster(n, order, is_valid_with_length)
        latency_with_length = time.perf_counter() - t0

        return {
            "with_length": with_length,
            "no_length": no_length,
            "latency_no_length": latency_no_length,
            "latency_with_length": latency_with_length,
        }

    # -----------------------------
    # Clustered inference
    # -----------------------------
    def get_prompt(self, group):
        return build_evidence_cluster_prompt(self.children, self.retrieved_list, group)

    def generate_batched_answer(self, input_prompt, g_cost, max_token):
        return run_llm_with_cost(
            llm_type=self.llm_type,
            tokenizer=self.tokenizer,
            model=self.model,
            sys_prompt=EVIDENCE_ANSWER_SYS_PROMPT,
            input_prompt=input_prompt,
            g_cost=g_cost,
            max_token=max_token,
        )

    def inference(self, group, g_cost, max_token=10000):
        input_prompt = self.get_prompt(group)
        generation, g_cost, g_cost_individual = self.generate_batched_answer(input_prompt, g_cost, max_token)

        DEFAULT_FALLBACK = {"raw": generation, **{str(i + 1): "" for i in group}}
        generation = extract_json_block(DEFAULT_FALLBACK, generation)

        for i in group:
            if str(i + 1) not in generation.keys():
                generation[str(i + 1)] = ""

        return generation, g_cost, g_cost_individual
