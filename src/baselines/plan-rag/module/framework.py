import ast
from module.dag import *


class Plan_Star_RAG_Framework:
    def __init__(self, generator, retriever):
        self.generator = generator
        self.retriever = retriever
        self.max_token = 16384

    def plan_star_rag(self, node: DAGNode):
        q_plan_cost = {"call": 0, "input": 0, "output": 0, "latency": 0}
        rewrite_cost = {"call": 0, "input": 0, "output": 0, "latency": 0}
        g_cost = {"call": 0, "input": 0, "output": 0, "latency": 0}
        r_cost = {"call": 0, "embed_latency": 0, "search_latency": 0}

        # Line 3: Generate Reasoning Plan
        dag_plan, q_plan_cost, q_plan_cost_individual = self.generator.generate_reasoning_plan(
            node.query, q_plan_cost, self.max_token
        )
        node.meta["q_plan_cost"] = q_plan_cost_individual
        try:
            dag_result = self._parse_reasoning_plan(dag_plan)
        except ValueError as e:
            node.meta["plan_parse_error"] = str(e)
            raise

        try:
            nodes, analysis = self._create_dag_from_plan(dag_result, node)
        except Exception as e:
            node.meta["dag_build_error"] = str(e)
            raise

        max_depth = analysis["max_depth"]
        nodes_by_depth = analysis["nodes_by_depth"]
        start_depth = 0 if max_depth == 0 else 1

        # Line 7: Execute Nodes
        for depth in range(start_depth, max_depth + 1):
            depth_nodes = nodes_by_depth[depth]
            depth_nodes = self._sort_nodes(depth_nodes)

            # Line 8: Execute Nodes
            for depth_node in depth_nodes:
                rewrite_cost, g_cost, r_cost = self._execute_node(
                    depth_node, rewrite_cost, g_cost, r_cost
                )

        parent_qa_pairs = self._build_all_parent_qa_pairs(analysis, nodes)
        
        # Line 15: Generate Final Answer
        final_answer, g_cost, g_cost_individual = self.generator.generate_final_answer(
            node.query, parent_qa_pairs, g_cost, 2000
        )
        node.answer = final_answer
        node.meta["g_cost"] = g_cost_individual

        return q_plan_cost, rewrite_cost, g_cost, r_cost


    def _parse_reasoning_plan(self, plan_text):
        text = plan_text.strip()
        if not text:
            raise ValueError("Reasoning plan is empty.")

        try:
            return ast.literal_eval(text)
        except (ValueError, SyntaxError):
            pass

        start_index = text.find('[')
        end_index = text.rfind(']')
        if start_index != -1 and end_index != -1 and start_index < end_index:
            try:
                return ast.literal_eval(text[start_index:end_index + 1])
            except (ValueError, SyntaxError):
                pass

        if "DAG:" in text:
            after = text.split("DAG:", 1)[1].strip()
            try:
                return ast.literal_eval(after)
            except (ValueError, SyntaxError):
                raise ValueError("Failed to parse reasoning plan.")

        raise ValueError("Failed to parse reasoning plan.")


    def _create_dag_from_plan(self, dag_result, root_node):
        if isinstance(dag_result, str):
            return self._create_dag_from_str(dag_result, root_node)

        return self._create_dag_from_tuples(dag_result, root_node)


    def _create_dag_from_str(self, dag_text, root_node):
        root_node.update_query(dag_text)
        nodes = {root_node.query: root_node}

        self._set_depths_from_roots(nodes)
        analysis = self._build_nodes_analysis(nodes)
        return nodes, analysis


    def _create_dag_from_tuples(self, dag_tuples, root_node):
        standard_dag_tuples = list(dag_tuples)

        nodes = {}

        all_queries = []
        for parent_query, child_query in standard_dag_tuples:
            if parent_query not in all_queries:
                all_queries.append(parent_query)
            if child_query not in all_queries:
                all_queries.append(child_query)

        child_queries = {child for _, child in standard_dag_tuples}
        root_queries = [query for query in all_queries if query not in child_queries]
        root_query = root_queries[0]

        for query in all_queries:
            if query == root_query:
                node = root_node
                node.update_query(query)
            else:
                node = DAGNode(query)
            nodes[node.query] = node

        for parent_query, child_query in standard_dag_tuples:
            parent_node = nodes.get(parent_query)
            child_node = nodes.get(child_query)
            if parent_node and child_node:
                child_node.add_parent(parent_node)

        self._set_depths_from_roots(nodes)
        analysis = self._build_nodes_analysis(nodes)
        return nodes, analysis


    def _set_depths_from_roots(self, nodes):
        roots = [node for node in nodes.values() if not node.parent_nodes]
        for root in roots:
            root.depth = 0

        queue = list(roots)
        visited = set()
        while queue:
            current = queue.pop(0)
            visited.add(current.query)
            for child in current.child_nodes:
                child.depth = current.depth + 1
                if child.query not in visited:
                    queue.append(child)


    def _build_nodes_analysis(self, nodes):
        nodes_by_depth = {}
        for node in nodes.values():
            depth = node.depth
            if depth not in nodes_by_depth:
                nodes_by_depth[depth] = []
            nodes_by_depth[depth].append(node)

        max_depth = max(nodes_by_depth.keys()) if nodes_by_depth else 0
        expected_depths = set(range(0, max_depth + 1))
        if set(nodes_by_depth.keys()) != expected_depths:
            raise ValueError("Depth levels are not contiguous.")

        return {
            'nodes_by_depth': nodes_by_depth,
            'max_depth': max_depth,
        }


    def _sort_nodes(self, nodes):
        return sorted(nodes, key=self._query_id_sort_key)


    def _query_id_sort_key(self, node):
        qid = node.query_id
        if qid == "Q":
            return (0, 0)
        if qid.startswith("Q"):
            parts = qid[1:].split(".")
            major = int(parts[0]) if parts[0] else 0
            minor = int(parts[1]) if len(parts) > 1 else 0
            return (major, minor)
        return (float("inf"), float("inf"))


    def _execute_node(self, node, rewrite_cost, g_cost, r_cost):
        parents = self._get_sorted_parents(node)

        if node.has_dynamic_tags():
            parent_qa_pairs = self._build_parent_qa_pairs(parents)
            generated_query, rewrite_cost, rewrite_cost_individual = self.generator.generate_subquery(
                node.original_query, parent_qa_pairs, rewrite_cost, self.max_token
            )
            node.query = generated_query.strip()
            node.meta["rewrite_cost"] = rewrite_cost_individual

        search_query = node.query
        node.search_query = search_query

        retrievals, r_cost, r_cost_individual = self.retriever.Retrieve(search_query, r_cost)
        node.search_results = retrievals
        node.meta["r_cost"] = r_cost_individual

        known_answers = self._build_known_answers(parents)
        node.answer, g_cost, g_cost_individual = self.generator.generate_subanswer(
            node.query, search_query, known_answers, retrievals, g_cost, 2000
        )
        node.meta["g_cost"] = g_cost_individual
        return rewrite_cost, g_cost, r_cost


    def _get_sorted_parents(self, node):
        parents = list(node.parent_nodes)
        return sorted(parents, key=self._query_id_sort_key)


    def _build_parent_qa_pairs(self, parents):
        pairs = []
        for parent in parents:
            query_line = parent.query
            answer_id = self._answer_id_from_query_id(parent.query_id)
            answer_line = f"{answer_id}: {parent.answer}"
            pairs.append(f"{query_line}\n{answer_line}")
        return "\n".join(pairs)


    def _answer_id_from_query_id(self, query_id):
        if query_id.startswith("Q") and len(query_id) > 1:
            return f"A{query_id[1:]}"
        return "A"


    def _build_known_answers(self, parents):
        if not parents:
            return ""
        parts = [f"Q={parent.query} A={parent.answer}" for parent in parents]
        return "Known answers: " + " ".join(parts)


    def _build_all_parent_qa_pairs(self, analysis, nodes):
        root_ids = {node.query for node in nodes.values() if not node.parent_nodes}
        sub_nodes = [node for node in nodes.values() if node.query not in root_ids]
        sub_nodes = sorted(sub_nodes, key=self._query_id_sort_key)
        return self._build_parent_qa_pairs(sub_nodes)
