from typing import List, Dict
from module.tree import TreeNode, TreeOfQuestionState
from module.module import *
from module.utils import Document


class TreeOfQuestion:
    def __init__(
        self,
        q_decomposer, q_generator, q_evaluator,
        a_generator, a_integrator,
        retriever, 
        max_depth=4
    ):
        self.q_decomposer = q_decomposer
        self.q_generator = q_generator
        self.q_evaluator = q_evaluator
        self.a_generator = a_generator
        self.a_integrator = a_integrator
        self.retriever = retriever
        self.max_depth = max_depth
        self._node_counter = 0
    
    def _next_node_id(self) -> int:
        """Return the next node ID."""
        node_id = self._node_counter
        self._node_counter += 1
        return node_id
    
    def toq(self, question: str):
        cost_dict = {
            "question_plan_cost": {"call": 0, "input": 0, "output": 0, "latency": 0},
            "query_gen_cost"    : {"call": 0, "input": 0, "output": 0, "latency": 0},
            "ans_span_cost"     : {"call": 0, "input": 0, "output": 0, "latency": 0},
            "eval_cost"         : {"call": 0, "input": 0, "output": 0, "latency": 0},
            "g_cost"            : {"call": 0, "input": 0, "output": 0, "latency": 0},
            "r_cost"            : {"call": 0, "embed_latency": 0, "search_latency": 0}
        }

        self._node_counter = 0
        root = TreeNode(
            question=question,
            level=1,
            node_id=self._next_node_id()
        )
        state = TreeOfQuestionState(root=root, current_nodes=[root])
        
        infos = []
        root_response, cost_dict = self._process_node(root, question, state, infos, cost_dict)
        
        if root.is_resolved:
            result = self._build_result(state, infos)
            return root_response, result, state, cost_dict

        final_response, cost_dict = self._final_answer(state, question, cost_dict)
        result = self._build_result(state, infos)
        return final_response, result, state, cost_dict

    def _build_result(self, state, infos):
        return {
            "steps": infos,
            "total_nodes": len(state.all_nodes),
            "total_queries": state.total_queries,
            "total_retrievals": state.total_retrievals
        }

    def _convert_to_documents(self, retrieved_list: List[dict]) -> List[Document]:
        """Convert retriever dictionaries into Document objects."""
        documents = []
        for item in retrieved_list:
            doc = Document(
                title=item.get("title", ""),
                content=item.get("content", ""),
                url=item.get("url", ""),
                doc_id=item.get("doc_id", ""),
                page_id=item.get("page_id", ""),
                section=item.get("section", ""),
                type_=item.get("type", ""),

                chunk_id=item.get("chunk_id", ""),
                sub_chunk_id=item.get("sub_chunk_id", ""),
                total_sub_chunk=item.get("total_sub_chunk", ""),                

                score=item.get("score")
            )
            documents.append(doc)
        return documents

    def _process_node(
        self,
        node: TreeNode,
        original_question: str,
        state: TreeOfQuestionState,
        infos: List[Dict],
        cost_dict: dict
    ) -> tuple[str, dict]:

        # Line 5: Determine Level
        if node.level > self.max_depth:
            return node.response or "", cost_dict
        
        # Line 6: Identify dependent Nodes, use Answer Integrator
        if node.depends_on:
            resolved_question = self._resolve_dependencies(node, state)
            node.question = resolved_question
        
        # Line 7: Generate query from the (modified) Node
        query, cost_dict, query_gen_cost_individual = self.q_generator.Generate(
            node.question,
            cost_dict=cost_dict,
            max_token=500
        )
        node.query = query
        node.node_costs["query_gen_cost"] = query_gen_cost_individual
        state.total_queries += 1
        
        # Document Retrieval
        r_cost = cost_dict["r_cost"]
        retrieved_list, r_cost, r_cost_individual = self.retriever.Retrieve(query, r_cost)
        cost_dict["r_cost"] = r_cost
        node.node_costs["r_cost"] = r_cost_individual
        
        documents = self._convert_to_documents(retrieved_list)
        node.documents = documents
        state.total_retrievals += 1
        
        # Line 8: Generate response
        is_root = node.parent is None
        response, cost_dict, g_cost_individual = self.a_generator.Generate(
            node.question, 
            query,
            documents,
            is_root=is_root,
            cost_dict=cost_dict,
            max_token=1000
        )
        node.response = response
        node.node_costs["g_cost"] = g_cost_individual
        
        # Line 9: Evaluate
        metrics, cost_dict, eval_cost_individual = self.q_evaluator.Evaluate(
            original_question,
            response,
            cost_dict=cost_dict,
            max_token=500
        )
        node.eval_metrics = metrics
        node.node_costs["eval_cost"] = eval_cost_individual
        
        infos.append({
            "node_id": node.node_id,
            "level": node.level,
            "question": node.question,
            "query": query,
            "generation": response,
            "answer_span": node.answer_span,
            "metrics": metrics.to_dict(),
            "is_resolved": False
        })
        
        # Line 10-12: If Eval is positive, return
        if self.q_evaluator.is_positive(metrics):
            node.is_resolved = True
            infos[-1]["is_resolved"] = True
            return response, cost_dict
        
        cost_dict = self._decompose_and_recurse(node, original_question, state, infos, cost_dict)
        return node.response or "", cost_dict

    def _resolve_dependencies(
        self, 
        node: TreeNode, 
        state: TreeOfQuestionState
    ) -> str:
        """
        Resolve dependencies by replacing [ANS_N] placeholders with prior answers.
        """
        question = node.question
        
        if not node.depends_on:
            return question
        
        first_child_id = node.parent.children[0].node_id

        answers: Dict[int, str] = {}
        for dep_id in node.depends_on:
            dep_node = state.get_node_by_id(dep_id)
            if dep_node:
                val = dep_node.answer_span or dep_node.response
                if val:
                    sibling_index = dep_id - first_child_id
                    answers[sibling_index] = val

        return self.a_integrator.fill_answer_placeholders(question, answers)
    
    def _decompose_and_recurse(
        self,
        node: TreeNode,
        original_question: str,
        state: TreeOfQuestionState,
        infos: List[Dict],
        cost_dict: dict
    ) -> dict:
        """
        Decompose the node and recursively process child sub-questions.
        """
        # Line 13: Decompose Node into sub-questions
        sub_questions, cost_dict, question_plan_cost_individual = self.q_decomposer.Decompose(node.question, cost_dict, max_token=1000)
        node.node_costs["question_plan_cost"] = question_plan_cost_individual
        
        if len(sub_questions) <= 1:
            return cost_dict
        
        child_nodes = []
        first_child_id = self._node_counter
        
        for i, sub_q in enumerate(sub_questions):
            actual_depends_on = None
            if sub_q.depends_on:
                actual_depends_on = [first_child_id + dep_idx for dep_idx in sub_q.depends_on]
            
            child_node = TreeNode(
                question=sub_q.text,
                level=node.level + 1,
                node_id=self._next_node_id(),
                depends_on=actual_depends_on
            )
            
            node.add_child(child_node)
            state.add_node(child_node)
            child_nodes.append(child_node)
        
        for child_node in child_nodes:
            _, cost_dict = self._process_node(child_node, original_question, state, infos, cost_dict)
            
            answer_span = None
            if child_node.documents:
                integrate_costs = []
                for i, doc in enumerate(child_node.documents):
                    result, cost_dict, ans_span_cost_individual = self.a_integrator.Integrate(
                        child_node.question,
                        doc,
                        cost_dict=cost_dict,
                        max_token=1000
                    )
                    integrate_costs.append({
                        "doc_index": i,
                        "cost": ans_span_cost_individual,
                        "relevance": result.relevance
                    })
                    
                    if result.relevance == "relevant" and result.answer_span:
                        answer_span = result.answer_span
                        break
                if integrate_costs:
                    child_node.node_costs["ans_span_cost"] = integrate_costs
            
            child_node.answer_span = answer_span

        return cost_dict

    def _final_answer(
        self,
        state: TreeOfQuestionState,
        original_question: str,
        cost_dict: dict
    ) -> tuple[str, dict]:
        """
        Synthesize the final answer from resolved nodes.
        """
        positive_nodes = [
            node for node in state.all_nodes
            if node.is_resolved and node.response
        ]
        
        if not positive_nodes:
            return state.root.response or "", cost_dict
        
        if len(positive_nodes) == 1:
            return positive_nodes[0].response, cost_dict
        
        positive_nodes.sort(key=lambda n: (n.level, n.node_id))
        
        context = "\n\n".join([
            f"Q{i+1}: {node.question}\nA{i+1}: {node.response}"
            for i, node in enumerate(positive_nodes)
        ])
        
        final_answer, cost_dict, final_g_cost_individual = self.a_generator.Final_Generate(
            original_question,
            context,
            cost_dict=cost_dict,
            max_token=1000
        )
        state.root.node_costs["final_g_cost"] = final_g_cost_individual
        return final_answer, cost_dict
