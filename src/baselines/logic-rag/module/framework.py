import logging
import re
from typing import List, Tuple, Dict, Any
from colorama import Fore, Style, init


# Initialize colorama
init()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LogicRAG:
    def __init__(
        self, 
        summary_generator,
        warm_up_analyzer,
        dependency_analyzer,
        answer_generator,
        dependency_sorter,
        retriever, 
        filter_repeats: bool = False,
        top_k: int = 20,
        max_rounds: int = 3
    ):
        self.max_rounds = max_rounds  # Default max rounds for iterative retrieval
        self.filter_repeats = filter_repeats  # Option to filter repeated chunks across rounds
        self.top_k = top_k  # original : 5
        self.summary_generator = summary_generator
        self.warm_up_analyzer = warm_up_analyzer
        self.dependency_analyzer = dependency_analyzer
        self.answer_generator = answer_generator
        self.dependency_sorter = dependency_sorter
        self.retriever = retriever
        # Stores the most recent execution trace in the shared evaluation schema.
        self.last_execution_trace = {}
        self.last_dependency_analysis = []

    @staticmethod
    def _context_to_text(context: Any) -> str:
        # Keep original behavior (List[str]) for generation modules.
        if isinstance(context, str):
            return context
        if isinstance(context, dict):
            title = str(context.get("title", "")).strip()
            content = str(context.get("content", "")).strip()
            if title and content:
                return f"{title}\n{content}"
            return content or title
        return str(context)

    def _contexts_to_text_list(self, contexts: List[Any]) -> List[str]:
        return [self._context_to_text(ctx) for ctx in contexts]

    def _context_key(self, context: Any) -> str:
        # Stable key for deduplication across retrieval rounds.
        if isinstance(context, dict):
            title = str(context.get("title", "")).strip()
            content = str(context.get("content", "")).strip()
            return f"{title}::{content}"
        return self._context_to_text(context)

    @staticmethod
    def _safe_issue(analysis: Dict[str, Any]) -> str:
        if not isinstance(analysis, dict):
            return ""
        return analysis.get("missing_reason", "") if not analysis.get("can_answer", True) else ""

    def _build_execution_trace(
        self,
        question: str,
        answer: str,
        sorted_dependencies: List[str],
        retrieval_history: List[Dict[str, Any]],
        dependency_analysis_history: List[Dict[str, Any]],
        info_summary: str,
        warm_up_summary_cost_ind: Dict[str, Any],
        warm_up_cost_ind: Dict[str, Any],
    ) -> Dict[str, Any]:
        decomposition: Dict[str, str] = {}
        provenance: Dict[str, Dict[str, Any]] = {}

        # Root node (0)
        root_issue = ""
        if dependency_analysis_history:
            last_analysis = dependency_analysis_history[-1]
            root_issue = self._safe_issue(last_analysis.get("analysis", {}))

        provenance["0"] = {
            "question": question,
            "generation": answer,
            "memory": {},
            "context": [],
            "plan_type": "logic_dependency",
            "retrieved_documents": [],
            "summary_cost": warm_up_summary_cost_ind,
            "warm_up_cost": warm_up_cost_ind,
            "issue": root_issue,
        }

        # DAG-aware ids:
        # - no dependency tags: top-level ids (1, 2, ...)
        # - references to parent tags (<Q2>): child ids (2.1, 2.2, ...)
        # - if multiple parents exist, representative parent is the first sorted reference
        #   and all references are stored in provenance.
        def extract_refs(text: str) -> List[str]:
            if not isinstance(text, str):
                return []
            # Only accept DAG parent references in <Qn.m> form.
            # <Qn> is treated as invalid and ignored by design.
            refs = re.findall(r"<(Q\d+\.\d+)>", text)
            seen = set()
            ordered = []
            for ref in refs:
                if ref not in seen:
                    seen.add(ref)
                    ordered.append(ref)
            return ordered

        # depth-based ids:
        # depth 1 parallel nodes -> 1.1, 1.2, ...
        # node depending on 1.1    -> 2.1
        # node depending on 2.1/2.2 -> 3.1
        node_id_by_query: Dict[str, str] = {}
        depth_by_node_id: Dict[str, int] = {}
        count_by_depth: Dict[int, int] = {}

        retrieval_by_query = {item.get("query"): item for item in retrieval_history}
        analysis_by_query = {
            item.get("query"): item.get("analysis", {}) for item in dependency_analysis_history if "query" in item
        }

        for dep in sorted_dependencies:
            refs = extract_refs(dep)
            valid_parents = [ref for ref in refs if ref in depth_by_node_id]

            if valid_parents:
                node_depth = max(depth_by_node_id[parent] for parent in valid_parents) + 1
            else:
                node_depth = 1

            next_idx = count_by_depth.get(node_depth, 0) + 1
            count_by_depth[node_depth] = next_idx
            node_id = f"Q{node_depth}.{next_idx}"

            node_id_by_query[dep] = node_id
            depth_by_node_id[node_id] = node_depth
            decomposition[node_id] = dep
            retrieval = retrieval_by_query.get(dep, {})
            analysis = analysis_by_query.get(dep, {})
            if not isinstance(analysis, dict):
                analysis = {}
            provenance[node_id] = {
                "question": dep,
                "generation": analysis.get("current_understanding", ""),
                "memory": {
                    "summary_snapshot": info_summary,
                    "depends_on": valid_parents,
                },
                "context": [],
                "plan_type": "maintain",
                "retrieved_documents": retrieval.get("contexts", []),
                "summary_cost": retrieval.get("summary_cost"),
                "dependency_aware_cost": retrieval.get("dependency_aware_cost"),
                "r_cost": retrieval.get("r_cost"),
                "warm_up_cost": None,
                "issue": self._safe_issue(analysis),
            }

        return {
            "generation": answer,
            "decomposition": decomposition,
            "provenance": provenance,
            "issue": "",
        }

    def get_execution_trace(self, ex_num: int, query: str, validated_answer: Any, cost: Dict[str, Any]) -> Dict[str, Any]:
        """
        Build an export-ready execution trace with shared evaluation keys.
        """
        base = dict(self.last_execution_trace) if isinstance(self.last_execution_trace, dict) else {}
        # Keep output key order consistent with Plan_Star trace style.
        ordered_trace = {
            "ex_num": ex_num,
            "query": query,
            "validated_answer": validated_answer,
            "generation": base.get("generation", ""),
            "decomposition": base.get("decomposition", {}),
            "issue": base.get("issue", ""),
            "provenance": base.get("provenance", {}),
            "cost": cost,
        }
        return ordered_trace
    
    def _retrieve_with_filter(self, query: str, retrieved_chunks_set: set, r_cost: Dict[str, Any]) -> Tuple[list, Dict[str, Any], Dict[str, Any]]:
        all_results, r_cost, r_cost_ind = self.retriever.Retrieve(query, r_cost)
        unique_results = []
        idx = self.top_k
        # If not enough unique in top_k, keep expanding
        while len(unique_results) < self.top_k and idx <= 1000:
            # Expand retrieval window
            all_results, r_cost, r_cost_ind = self.retriever.Retrieve(query, r_cost) if idx == self.top_k else self._retrieve_top_n(query, idx, r_cost)
            unique_results = [chunk for chunk in all_results if self._context_key(chunk) not in retrieved_chunks_set]
            idx += self.top_k
        return unique_results[:self.top_k], r_cost, r_cost_ind

    def _retrieve_top_n(self, query: str, n: int, r_cost: Dict[str, Any]) -> Tuple[list, Dict[str, Any], Dict[str, Any]]:
        # Temporarily override top_k
        old_top_k = self.top_k
        self.top_k = n
        results, r_cost, r_cost_ind = self.retriever.Retrieve(query, r_cost)
        self.top_k = old_top_k
        return results, r_cost, r_cost_ind

    def answer_question(self, question: str) -> Tuple[str, List[str], int]:
        summary_cost             = {"call": 0, "input": 0, "output": 0, "latency": 0}
        warm_up_cost             = {"call": 0, "input": 0, "output": 0, "latency": 0}
        dependency_aware_cost    = {"call": 0, "input": 0, "output": 0, "latency": 0}
        dependency_sorter_cost   = {"call": 0, "input": 0, "output": 0, "latency": 0}
        g_cost                   = {"call": 0, "input": 0, "output": 0, "latency": 0}
        r_cost                   = {"call": 0, "embed_latency": 0, "search_latency": 0}
        warm_up_summary_cost_ind = {"call": 0, "input": 0, "output": 0, "latency": 0}
        warm_up_cost_ind         = {"call": 0, "input": 0, "output": 0, "latency": 0}
        
        info_summary = "" 
        round_count = 0
        current_query = question
        retrieval_history = []
        last_contexts = []  
        dependency_analysis_history = []  
        retrieved_chunks_set = set() if self.filter_repeats else None  # Track retrieved chunks if filtering
        
        print(f"\n\n{Fore.CYAN} answering: {question}{Style.RESET_ALL}\n\n")
        
        #===============================================
        #== Stage 1: warm up retrieval ==
        if self.filter_repeats:
            new_contexts_raw, r_cost, r_cost_ind = self._retrieve_with_filter(question, retrieved_chunks_set, r_cost)
            for chunk in new_contexts_raw:
                retrieved_chunks_set.add(self._context_key(chunk))
        else:
            new_contexts_raw, r_cost, r_cost_ind = self.retriever.Retrieve(question, r_cost)
        new_contexts = self._contexts_to_text_list(new_contexts_raw)
        last_contexts = new_contexts
        info_summary, summary_cost, summary_cost_ind = self.summary_generator.refine_summary_with_context(
            question, 
            new_contexts, 
            info_summary,
            summary_cost,
            250
        )
        warm_up_summary_cost_ind = summary_cost_ind
        analysis, warm_up_cost, warm_up_cost_ind = self.warm_up_analyzer.warm_up_analysis(
            question, info_summary,
            warm_up_cost,
            250
        )

        if analysis["can_answer"]:
            # In this case, the question can be answered with simple fact retrieval, without any dependency analysis
            print(f"Warm-up analysis indicate the question can be answered with simple fact retrieval, without any dependency analysis.")
            answer, g_cost, g_cost_ind = self.answer_generator.generate_answer(question, info_summary, g_cost, 250)
            # Reset dependency analysis history for simple questions
            self.last_dependency_analysis = []
            self.last_execution_trace = {
                "generation": answer,
                "decomposition": {},
                "issue": "",
                "provenance": {
                    "0": {
                        "question": question,
                        "generation": answer,
                        "memory": {},
                        "context": [],
                        "plan_type": "warm_up",
                        "retrieved_documents": new_contexts_raw,
                        "summary_cost": warm_up_summary_cost_ind,
                        "warm_up_cost": warm_up_cost_ind,
                        "g_cost": g_cost_ind,
                        "r_cost": r_cost_ind,
                        "issue": "",
                    }
                },
            }
            return (answer, last_contexts, round_count,
            summary_cost, warm_up_cost, dependency_aware_cost, dependency_sorter_cost, g_cost, r_cost)
        else:
            logger.info(f"Warm-up analysis indicate the requirement of deeper reasoning-enhanced RAG. Now perform analysis with logical dependency graph.")
            logger.info(f"Dependencies: {', '.join(analysis.get('dependencies', []))}")

            # sort the dependencies, by first constructing the dependency graphs, then use topological sort to get the sorted dependencies
            try:
                sorted_dependencies, dependency_sorter_cost, dependency_sorter_cost_ind = self.dependency_sorter.sort_dependencies(
                    analysis["dependencies"], question,
                    dependency_sorter_cost,
                    250
                )
            except Exception as e:
                dependency_sorter_cost = getattr(e, "dependency_sorter_cost", dependency_sorter_cost)
                dependency_sorter_cost_ind = getattr(e, "dependency_sorter_cost_ind", None)
                error_message = f"dependency_sorter_error: {type(e).__name__}: {e}"
                logger.exception(
                    f"{Fore.RED}Dependency sorting failed; marking LogicRAG answer as None. "
                    f"question={question}, dependencies={analysis.get('dependencies', [])}{Style.RESET_ALL}"
                )
                self.last_dependency_analysis = dependency_analysis_history
                self.last_execution_trace = {
                    "generation": None,
                    "decomposition": {},
                    "issue": error_message,
                    "provenance": {
                        "0": {
                            "question": question,
                            "generation": None,
                            "memory": {
                                "dependencies": analysis.get("dependencies", []),
                            },
                            "context": [],
                            "plan_type": "logic_dependency",
                            "retrieved_documents": new_contexts_raw,
                            "summary_cost": warm_up_summary_cost_ind,
                            "warm_up_cost": warm_up_cost_ind,
                            "dependency_sorter_cost": dependency_sorter_cost_ind,
                            "r_cost": r_cost_ind,
                            "issue": error_message,
                        }
                    },
                }
                return (None, last_contexts, round_count,
                summary_cost, warm_up_cost, dependency_aware_cost, dependency_sorter_cost, g_cost, r_cost)
            dependency_analysis_history.append({"sorted_dependencies": sorted_dependencies})
            logger.info(f"Sorted dependencies: {sorted_dependencies}\n\n")
        #===============================================
        #== Stage 2: agentic iterative retrieval ==
        idx = 0 # used to track the current dependency index

        while round_count < self.max_rounds and idx < len(sorted_dependencies):
            round_count += 1
            
            current_query = sorted_dependencies[idx]
            if self.filter_repeats:
                new_contexts_raw, r_cost, r_cost_ind = self._retrieve_with_filter(current_query, retrieved_chunks_set, r_cost)
                for chunk in new_contexts_raw:
                    retrieved_chunks_set.add(self._context_key(chunk))
            else:
                new_contexts_raw, r_cost, r_cost_ind = self.retriever.Retrieve(current_query, r_cost)
            new_contexts = self._contexts_to_text_list(new_contexts_raw)
            last_contexts = new_contexts  # Save current contexts
            
            
            # Generate or refine information summary with new contexts
            info_summary, summary_cost, summary_cost_ind = self.summary_generator.refine_summary_with_context(
                question, 
                new_contexts, 
                info_summary,
                summary_cost,
                250
            )
            
            logger.info(f"Agentic retrieval at round {round_count}")
            logger.info(f"current query: {current_query}")
            
            analysis, dependency_aware_cost, dependency_aware_cost_ind = self.dependency_analyzer.dependency_analysis(
                question, info_summary, sorted_dependencies, idx,
                dependency_aware_cost,
                250
            )

            retrieval_history.append({
                "round": round_count,
                "query": current_query,
                "contexts": new_contexts_raw,
                "summary_cost": summary_cost_ind,
                "dependency_aware_cost": dependency_aware_cost_ind,
                "r_cost": r_cost_ind,
            }) 

            dependency_analysis_history.append({
                "round": round_count,
                "query": current_query,
                "analysis": analysis
            })

            if not isinstance(analysis, dict) or "can_answer" not in analysis:
                error_message = f"dependency_analysis_error: invalid analysis response for query: {current_query}"
                logger.error(f"{Fore.RED}{error_message}{Style.RESET_ALL}")
                self.last_dependency_analysis = dependency_analysis_history
                self.last_execution_trace = self._build_execution_trace(
                    question=question,
                    answer=None,
                    sorted_dependencies=sorted_dependencies,
                    retrieval_history=retrieval_history,
                    dependency_analysis_history=dependency_analysis_history,
                    info_summary=info_summary,
                    warm_up_summary_cost_ind=warm_up_summary_cost_ind,
                    warm_up_cost_ind=warm_up_cost_ind,
                )
                self.last_execution_trace["issue"] = error_message
                self.last_execution_trace["generation"] = None
                self.last_execution_trace["provenance"]["0"]["generation"] = None
                self.last_execution_trace["provenance"]["0"]["issue"] = error_message
                return (None, last_contexts, round_count,
                summary_cost, warm_up_cost, dependency_aware_cost, dependency_sorter_cost, g_cost, r_cost)

            if analysis["can_answer"]:
                # Generate and return final answer
                answer, g_cost, g_cost_ind = self.answer_generator.generate_answer(question, info_summary, g_cost, 250)
                # Store dependency analysis history for evaluation access
                self.last_dependency_analysis = dependency_analysis_history
                self.last_execution_trace = self._build_execution_trace(
                    question=question,
                    answer=answer,
                    sorted_dependencies=sorted_dependencies,
                    retrieval_history=retrieval_history,
                    dependency_analysis_history=dependency_analysis_history,
                    info_summary=info_summary,
                    warm_up_summary_cost_ind=warm_up_summary_cost_ind,
                    warm_up_cost_ind=warm_up_cost_ind,
                )
                # We return the last retrieved contexts for evaluation purposes
                return (answer, last_contexts, round_count,
                summary_cost, warm_up_cost, dependency_aware_cost, dependency_sorter_cost, g_cost, r_cost)
            else:
                idx += 1
        
        # If max rounds reached, generate best possible answer
        logger.info(f"Reached maximum rounds ({self.max_rounds}). Generating final answer...")
        answer, g_cost, g_cost_ind = self.answer_generator.generate_answer(question, info_summary, g_cost, 250)
        # Store dependency analysis history for evaluation access
        self.last_dependency_analysis = dependency_analysis_history
        self.last_execution_trace = self._build_execution_trace(
            question=question,
            answer=answer,
            sorted_dependencies=sorted_dependencies,
            retrieval_history=retrieval_history,
            dependency_analysis_history=dependency_analysis_history,
            info_summary=info_summary,
            warm_up_summary_cost_ind=warm_up_summary_cost_ind,
            warm_up_cost_ind=warm_up_cost_ind,
        )
        return (answer, last_contexts, round_count,
        summary_cost, warm_up_cost, dependency_aware_cost, dependency_sorter_cost, g_cost, r_cost)
