import re
from typing import Dict, Optional
from collections import defaultdict
from module.tree import TreeNode


class APTRAGFramework:
    def __init__(self, q_manager, q_rewriter, retriever, generator, query_refiner, m_controller, reranker, evidence_clusterer):
        self.q_manager      = q_manager
        self.q_rewriter     = q_rewriter
        self.retriever      = retriever
        self.generator      = generator
        self.query_refiner  = query_refiner
        self.m_controller   = m_controller
        self.reranker       = reranker
        self.evidence_clusterer = evidence_clusterer

    def _depth(self, node: TreeNode):
        d, cur = 0, node
        while cur.parent is not None:
            d += 1
            cur = cur.parent
        return d 

    def _extract_answers(self, text: str):
        m = list(re.finditer(r'\bAnswers:\s*', text))
        if not m:
            return text
        return text[m[-1].end():].strip()

    def _extract_tag(self, text:str):
        pattern = r"<Q(\d+)>"
        matches = re.findall(pattern, text)
        
        if matches:
            return True, [int(n) for n in matches]
        else:
            return False, []

    def _retrieval(self, node:TreeNode, q_rewrite_cost, r_cost, retrieved_call:int):
        if not node.query:
            search_query, q_rewrite_cost, q_rewrite_cost_individual = self.q_rewriter.Rewrite_single(node.question, q_rewrite_cost, 500)
            node.query = search_query
            node.meta['q_rewrite_cost'] = q_rewrite_cost_individual
        else:
            search_query = node.query

        dpr_retrieved, r_cost, r_cost_individual = self.retriever.Retrieve(search_query, r_cost)
        retrieved_call += 1
        node.meta['retrieved'] = dpr_retrieved
        node.meta['r_cost'] = r_cost_individual
        return retrieved_call, q_rewrite_cost, r_cost

    def _planning(self, node, r_plan_cost, q_plan_cost, max_depth:int=4):
        current_depth = self._depth(node)

        if current_depth+1 >= max_depth:
            node.meta["inf_decomp_prev"] = True
            node.meta['plan_type'] = 'maintain'
            sub_queries = []
            needs_external_search = True
            node.meta['retrieve_plan'] = {'retrieval': needs_external_search}
        else:
            context_tag = bool(node.meta.get("re_write"))
            if context_tag:
                context = node.meta.get("context_content")
                answerability_decision, raw_result, r_plan_cost, r_plan_cost_individual = self.q_manager.answerability_check(node.question, context, r_plan_cost, 500)
                needs_external_search = answerability_decision["retrieval"]
                node.meta['retrieve_plan'] = answerability_decision
                node.meta['r_plan_cost'] = r_plan_cost_individual
            else:
                needs_external_search = True
                node.meta['retrieve_plan'] = {'retrieval': needs_external_search}

            sub_queries = []
            if needs_external_search:
                decomp_plan, q_plan_cost, q_plan_cost_individual = self.q_manager.decomposition_planner(node.question, q_plan_cost, 1000)
                node.meta['plan_type'] = decomp_plan.get('decision')
                node.meta['q_plan_cost'] = q_plan_cost_individual
                sub_queries = decomp_plan.get('subqueries') or []
        
        if sub_queries:
            memory = self.m_controller.build_memory(node.question, sub_queries)
            node.set_memory(memory) 
        return sub_queries, needs_external_search, q_plan_cost, r_plan_cost
    
    def _parent_node(
        self, 
        node, 
        sub_queries, 
        r_plan_cost, q_plan_cost, q_rewrite_cost, cm_cost, g_cost, r_cost, 
        retrieved_call, max_depth, max_context_size=183296
    ):
        queue = defaultdict(lambda: {"cluster": False, "nodes": []})
        external_answer_queue = {}
        group_key = 0

        for i, sub_q in enumerate(sub_queries, start=1):
            
            # ---- node creation

            parent_id = node.node_id
            child_id = str(i) if parent_id == "0" else f"{parent_id}.{i}" 
            child = node.add_child(sub_q, node_id=child_id, step='split', memory_key=f'SA{i}')

            # ---- contextualization check
            
            context_tag, key_num_lst  = self._extract_tag(sub_q)
            child.meta["key_num_lst"] = key_num_lst
            child.meta["re_write"] = context_tag

            if context_tag:
                group_key += 1
                queue[group_key]["nodes"].append({
                    "status": "need_context",
                    "node": child,
                    "index": i,
                })
                continue
            
            # ---- planning

            child_sub_queries, needs_external_search, q_plan_cost, r_plan_cost = self._planning(child, r_plan_cost, q_plan_cost, max_depth)

            if child_sub_queries:
                group_key += 1
                queue[group_key]["nodes"].append({
                    "status": "split",
                    "node": child,
                    "index": i,
                    "child_sub_queries": child_sub_queries,
                })
            elif not needs_external_search:
                group_key += 1
                queue[group_key]["nodes"].append({
                    "status": "maintain_not_retrieval",
                    "node": child,
                    "index": i,
                })
            else:
                external_answer_queue[i] = child

                if group_key != 0 and queue[group_key]["nodes"][-1]["status"] != "maintain_retrieval":
                    group_key += 1
                    
                queue[group_key]["nodes"].append({
                    "status": "maintain_retrieval",
                    "node": child,
                    "index": i,
                })

                if len(queue[group_key]["nodes"]) > 1:
                    queue[group_key]["cluster"] = True
        
        
        # ---- batched retrieval for external child nodes
        
        if external_answer_queue:
            questions = [external_answer_queue[i].question for i in external_answer_queue]
            q_idx_list = list(external_answer_queue.keys())
            search_query_dict, q_rewrite_cost, q_rewrite_cost_individual = self.q_rewriter.Rewrite_multi(questions, q_idx_list, q_rewrite_cost, 3000)
            
            for i, child in external_answer_queue.items():
                child.query = search_query_dict[str(i)]
                retrieved_call, q_rewrite_cost, r_cost = self._retrieval(child, q_rewrite_cost, r_cost, retrieved_call)

            first_key = list(external_answer_queue.keys())[0]
            external_answer_queue[first_key].meta['q_rewrite_cost'] = q_rewrite_cost_individual


        # ---- evidence-guided clustered answer generation

        for group_key, item in queue.items():

            if item["cluster"]:
                all_children = [nodes["node"] for nodes in item["nodes"]]
                sub_index_by_position = [nodes["index"] for nodes in item["nodes"]]
                all_retrieved_list = [child.meta.get("retrieved") for child in all_children]
                
                cluster_result = self.evidence_clusterer.cluster(
                    all_children,
                    all_retrieved_list,
                    context_window_size=max_context_size,
                    sim_threshold=self.evidence_clusterer.sim_threshold,
                )
                cluster_with_length = cluster_result["with_length"]
                cluster_no_length = cluster_result["no_length"]
                latency_no_length = cluster_result["latency_no_length"]
                latency_with_length = cluster_result["latency_with_length"]

                node.meta.setdefault("cluster_group", []).append(cluster_with_length)
                node.meta.setdefault("cluster_group_no_length", []).append(cluster_no_length)
                node.meta.setdefault("cluster_latency_no_length", []).append(latency_no_length)
                node.meta.setdefault("cluster_latency_with_length", []).append(latency_with_length)
                print(
                    f"Cluster with length ({len(cluster_with_length)} groups, "
                    f"{latency_with_length:.4f}s): {cluster_with_length}"
                )
                print(
                    f"Cluster no length ({len(cluster_no_length)} groups, "
                    f"{latency_no_length:.4f}s): {cluster_no_length}"
                )

                self.evidence_clusterer.children = all_children
                self.evidence_clusterer.retrieved_list = all_retrieved_list
                
                for group in cluster_with_length:
                    first_key = group[0]

                    if len(group) > 1:
                        answers, g_cost, g_cost_individual = self.evidence_clusterer.inference(group, g_cost, max_token=10000)

                        for i in group:
                            answer = answers[str(i+1)]
                            all_children[i].meta["clustered"] = True
                            all_children[i].set_answer(answer)
                            sa_key = sub_index_by_position[i]
                            node.memory[f'SA{sa_key}'] = answer
                        
                        all_children[first_key].meta["g_cost"] = g_cost_individual
                        
                    else:
                        child = all_children[first_key]
                        question = child.question
                        search_query = child.query
                        dpr_retrieved = all_retrieved_list[first_key]
                        answer, g_cost, g_cost_individual = self.generator.generate_external_answer(question, search_query, documents=dpr_retrieved, g_cost=g_cost, max_token=2000)

                        answer = self._extract_answers(answer)
                        child.set_answer(answer)
                        child.meta["clustered"] = False
                        child.meta["g_cost"] = g_cost_individual
                        node.memory[f'SA{sub_index_by_position[first_key]}'] = answer

            else:
                group             = item["nodes"][0]
                status            = group["status"]
                child             = group["node"]
                idx               = group["index"]
                child_sub_queries = group["child_sub_queries"] if group.get("child_sub_queries") else []
                context           = {}

                # ---- contextualization
                
                if status == "need_context":
                    child_key_lst = child.meta.get("key_num_lst", [])
                    context = self.m_controller.context_extract(node.memory, child_key_lst)

                    result, cm_cost, cm_cost_individual = self.query_refiner.context_specify(child.question, context, cm_cost, 1000)
                    re_write_query = result.get('step 4')
                    child.meta["context_content"] = context
                    child.meta["re_write_process"] = result
                    child.meta["cm_cost"] = cm_cost_individual

                    child.meta["original_question"] = child.question
                    child.refined_question = re_write_query
                    child.question = re_write_query
                    node.memory[f'SQ{idx}'] = child.question

                    child_sub_queries, needs_external_search, q_plan_cost, r_plan_cost = self._planning(child, r_plan_cost, q_plan_cost, max_depth)

                    status = (
                        "split" if child_sub_queries
                        else "maintain_not_retrieval" if not needs_external_search
                        else "maintain_retrieval"
                    )

                    if status == "maintain_retrieval":
                        retrieved_call, q_rewrite_cost, r_cost = self._retrieval(child, q_rewrite_cost, r_cost, retrieved_call)
                
                # ---- answer generation
                
                if status == "split":
                    retrieved_call, r_plan_cost, q_plan_cost, q_rewrite_cost, cm_cost, g_cost, r_cost = self._parent_node(child, child_sub_queries, r_plan_cost, q_plan_cost, q_rewrite_cost, cm_cost, g_cost, r_cost, retrieved_call, max_depth, max_context_size)
                    answer, g_cost, g_cost_individual = self.generator.generate_vertical_answer(child.question, child.memory, g_cost, 2000)

                    # Keep incomplete child evidence visible in execution traces.
                    for key in child.memory.keys():
                        if key != "RA" and child.memory[key] == "-":
                            print(f"[{child.node_id}] {key} is not filled")
                            child.meta["empty_memory"] = True

                else:
                    if status == "maintain_not_retrieval":
                        answer, g_cost, g_cost_individual = self.generator.generate_lateral_answer(child.question, context, g_cost, max_token=2000)

                    elif status == "maintain_retrieval":
                        search_query = child.query
                        dpr_retrieved = child.meta.get("retrieved")
                        answer, g_cost, g_cost_individual = self.generator.generate_external_answer(child.question, search_query, documents=dpr_retrieved, g_cost=g_cost, max_token=2000)

                answer = self._extract_answers(answer)
                child.set_answer(answer, context=context)
                child.meta['g_cost'] = g_cost_individual
                node.memory[f'SA{idx}'] = answer

        return retrieved_call, r_plan_cost, q_plan_cost, q_rewrite_cost, cm_cost, g_cost, r_cost
    
    def run(
        self, 
        node:TreeNode, memory:Optional[Dict]=None, context:Optional[Dict]=None, 
        r_plan_cost:Optional[Dict]=None, q_plan_cost:Optional[Dict]=None, 
        q_rewrite_cost:Optional[Dict]=None, cm_cost:Optional[Dict]=None, 
        g_cost:Optional[Dict]=None, 
        r_cost:Optional[Dict]=None, retrieved_call:int=0, 
        max_depth:int=4, max_context_size:int=183296
    ):
        # ---- initialization

        if node.node_id is None: 
            node.node_id = "0"
        
        if r_plan_cost is None:
            r_plan_cost    = {"call": 0, "input": 0, "output": 0, "latency": 0}
            q_plan_cost    = {"call": 0, "input": 0, "output": 0, "latency": 0}
            q_rewrite_cost = {"call": 0, "input": 0, "output": 0, "latency": 0}
            cm_cost        = {"call": 0, "input": 0, "output": 0, "latency": 0}
            g_cost         = {"call": 0, "input": 0, "output": 0, "latency": 0}
            r_cost         = {"call": 0, "embed_latency": 0, "search_latency": 0}
        

        # ---- root node processing

        sub_queries, needs_external_search, q_plan_cost, r_plan_cost = self._planning(node, r_plan_cost, q_plan_cost, max_depth)
        
        if sub_queries:
            retrieved_call, r_plan_cost, q_plan_cost, q_rewrite_cost, cm_cost, g_cost, r_cost = self._parent_node(node, sub_queries, r_plan_cost, q_plan_cost, q_rewrite_cost, cm_cost, g_cost, r_cost, retrieved_call, max_depth, max_context_size)
            answer, g_cost, g_cost_individual = self.generator.generate_final_answer(node.question, node.memory, g_cost, 2000)

        else:
            retrieved_call, q_rewrite_cost, r_cost = self._retrieval(node, q_rewrite_cost, r_cost, retrieved_call)
            question = node.question
            search_query = node.query
            dpr_retrieved = node.meta.get("retrieved")

            answer, g_cost, g_cost_individual = self.generator.Retrieval_Generate(question, search_query, documents=dpr_retrieved, g_cost=g_cost, max_token=2000)
        
        answer = self._extract_answers(answer)
        node.set_answer(answer)
        node.meta['g_cost'] = g_cost_individual
        node.memory['RA'] = answer

        return retrieved_call, r_plan_cost, q_plan_cost, q_rewrite_cost, cm_cost, g_cost, r_cost
