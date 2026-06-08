import logging
from typing import List, Dict, Tuple
from colorama import Fore, Style, init
from module.utils import fix_json_response
from prompt.prompt import *
from utils.module import *
from utils.utils import run_llm_with_cost



# Initialize colorama
init()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SummaryGenerator:
    def __init__(self, llm_type, tokenizer, model):
        self.llm_type = llm_type
        self.tokenizer = tokenizer
        self.model = model

    def refine_summary_with_context( # Generate a new summary or refine an existing one based on newly retrieved contexts.
        self, 
        question: str,               # The original question
        new_contexts: List[str],     # Newly retrieved context chunks
        current_summary: str = "",   # Current information summary (if any)
        summary_cost: dict = None,
        max_token: int = 250
    ) -> str:
        try:
            context_text = "\n".join(new_contexts)
            
            if not current_summary:
                # Generate initial summary
                prompt = INIT_SUMMARY_GENERATOR.format(question=question, context_text=context_text)

            else:
                # Refine existing summary with new information
                prompt = REFINE_SUMMARY_GENERATOR.format(question=question, current_summary=current_summary, context_text=context_text)
            
            # summary = get_response_with_retry(prompt)
            summary, summary_cost, summary_cost_ind = run_llm_with_cost(
                llm_type=self.llm_type, 
                tokenizer=self.tokenizer,
                model=self.model,
                sys_prompt="",
                input_prompt=prompt,
                g_cost=summary_cost,
                max_token=max_token,
            )
            return summary, summary_cost, summary_cost_ind
            
        except Exception as e:
            logger.error(f"{Fore.RED}Error generating/refining summary: {e}{Style.RESET_ALL}")
            summary_cost_ind = {"latency": 0, "input": 0, "output": 0, "call": 0}
            # If error occurs, concatenate current summary with new contexts as fallback
            if current_summary:
                return f"{current_summary}\n\nNew information:\n{context_text}", summary_cost, summary_cost_ind
            return context_text, summary_cost, summary_cost_ind


class WarmUpAnalyzer:
    def __init__(self, llm_type, tokenizer, model):
        self.llm_type = llm_type
        self.tokenizer = tokenizer
        self.model = model

    def warm_up_analysis(            # This is a warm-up analysis, which is used to analyze if the question can be answered with simple fact retrieval, without any dependency analysis.
        self, 
        question: str,               # The original question
        info_summary: str,           # Current information summary
        warm_up_cost: dict = None,
        max_token: int = 250
    ) -> Dict:
        try:
            prompt = WARM_UP_ANALYSIS_GENERATOR.format(question=question, info_summary=info_summary)
            response, warm_up_cost, warm_up_cost_ind = run_llm_with_cost(
                llm_type=self.llm_type, 
                tokenizer=self.tokenizer,
                model=self.model,
                sys_prompt="",
                input_prompt=prompt,
                g_cost=warm_up_cost,
                max_token=max_token,
            )
            
            # Clean up response to ensure it's valid JSON
            response = response.strip()
            
            # Remove any markdown code block markers
            response = response.replace('```json', '').replace('```', '')
            
            # Parse the cleaned response using fix_json_response
            result = fix_json_response(response)
            if result is None:
                warm_up_cost_ind = {"latency": 0, "input": 0, "output": 0, "call": 0}
                return {
                    "can_answer": True,
                    "missing_info": "",
                    "subquery": question,
                    "current_understanding": "Failed to parse reflection response.",
                    "dependencies": ["Information relevant to the question"],
                    "missing_reason": "Parse error occurred"
                }, warm_up_cost, warm_up_cost_ind
            
            # Validate required fields
            required_fields = ["can_answer", "missing_info", "subquery", "current_understanding"]
            if not all(field in result for field in required_fields):
                logger.error(f"{Fore.RED}Missing required fields in response: {response}{Style.RESET_ALL}")
                raise ValueError("Missing required fields")
            
            # Add default values for new interpretability fields if missing
            if "dependencies" not in result:
                result["dependencies"] = ["Information relevant to the question"]
            if "missing_reason" not in result:
                result["missing_reason"] = "Additional context needed" if not result["can_answer"] else "No missing information"
            
            # Ensure boolean type for can_answer
            result["can_answer"] = bool(result["can_answer"])
            
            # Ensure non-empty subquery
            if not result["subquery"]:
                result["subquery"] = question
            
            return result, warm_up_cost, warm_up_cost_ind
                
        except Exception as e:
            logger.error(f"{Fore.RED}Error in analyze_dependency_graph: {e}{Style.RESET_ALL}")
            warm_up_cost_ind = {"latency": 0, "input": 0, "output": 0, "call": 0}
            return {
                "can_answer": True,
                "missing_info": "",
                "subquery": question,
                "current_understanding": f"Error during analysis: {str(e)}",
                "dependencies": ["Information relevant to the question"],
                "missing_reason": "Analysis error occurred"
            }, warm_up_cost, warm_up_cost_ind


class DependencyAnalyzer:
    def __init__(self, llm_type, tokenizer, model):
        self.llm_type = llm_type
        self.tokenizer = tokenizer
        self.model = model

    def dependency_analysis(       # this function analyzes whether the current information summary is sufficient to answer the question with the decomposed dependencies as references.
        self,                       # And the function will answer whether the question can be answered, and if not, it will update the current query with dependencies as references.
        question: str, 
        info_summary: str, 
        dependencies: List[str], 
        idx: int,
        dependency_aware_cost: dict = None,
        max_token: int = 250
    ) -> str:
        try:
            prompt = DEPENDENCY_ANALYSIS_GENERATOR.format(question=question, info_summary=info_summary, dependencies=dependencies, dependency=dependencies[idx])
            response, dependency_aware_cost, dependency_aware_cost_ind = run_llm_with_cost(
                llm_type=self.llm_type, 
                tokenizer=self.tokenizer,
                model=self.model,
                sys_prompt="",
                input_prompt=prompt,
                g_cost=dependency_aware_cost,
                max_token=max_token,
            )
            result = fix_json_response(response)
            return result, dependency_aware_cost, dependency_aware_cost_ind
        except Exception as e:
            logger.error(f"{Fore.RED}Error in dependency_aware_rag: {e}{Style.RESET_ALL}")
            dependency_aware_cost_ind = {"latency": 0, "input": 0, "output": 0, "call": 0}
            return {
                "can_answer": True,
                "current_understanding": f"Error during analysis: {str(e)}",
            }, dependency_aware_cost, dependency_aware_cost_ind


class AnswerGenerator:
    def __init__(self, llm_type, tokenizer, model):
        self.llm_type = llm_type
        self.tokenizer = tokenizer
        self.model = model

    def generate_answer(self, question: str, info_summary: str, g_cost: dict = None, max_token: int = 250) -> str:
        try:
            prompt = ANSWER_GENERATOR.format(question=question, info_summary=info_summary)   
            return run_llm_with_cost(
                llm_type=self.llm_type, 
                tokenizer=self.tokenizer,
                model=self.model,
                sys_prompt="",
                input_prompt=prompt,
                g_cost=g_cost,
                max_token=max_token,
            )
        except Exception as e:
            logger.error(f"{Fore.RED}Error generating answer: {e}{Style.RESET_ALL}")
            g_cost_ind = {"latency": 0, "input": 0, "output": 0, "call": 0}
            return "", g_cost, g_cost_ind


class DependencySorter:
    def __init__(self, llm_type, tokenizer, model):
        self.llm_type = llm_type
        self.tokenizer = tokenizer
        self.model = model

    @staticmethod
    def _topological_sort(dependencies: List[str], dependencies_pairs: List[Tuple[int, int]]) -> List[str]:
        """
        Use graph-based algorithm to sort the dependencies in a topological order.
        Args:
            dependencies: List[str]
            dependencies_pairs: List[Tuple[int, int]]
        Returns:
            List[str]
        """
        graph = {dep: [] for dep in dependencies}
        
        for dependent_idx, dependency_idx in dependencies_pairs:
            if dependent_idx < len(dependencies) and dependency_idx < len(dependencies):
                dependent = dependencies[dependent_idx]
                dependency = dependencies[dependency_idx]
                graph[dependency].append(dependent)  # dependency -> dependent
        
        visited = set()
        stack = []
        
        def dfs(node):
            if node in visited:
                return
            visited.add(node)
            for neighbor in graph[node]:
                dfs(neighbor)
            stack.append(node)

        for node in graph:
            if node not in visited:
                dfs(node)
        
        return stack[::-1]

    def sort_dependencies(
        self, 
        dependencies: List[str], 
        query: str,
        dependency_sorter_cost: dict = None,
        max_token: int = 250
    ) -> List[Tuple]:
        """
        given a list of dependencies and the original query,
        sort the dependencies in a topological order, that is solving a dependency A relies on the solution of the dependent dependency B,
        then B should be before A in the sorted string.

        Args:
            dependencies: List[str]
            query: str

            
        For example, if the question is "What is the mayor of the capital of France?",
        the input dependencies for this question are:
        - The capital of France
        - The mayor of this capital

        Then the output should be:
        - The capital of France
        - The mayor of this capital

        there are two steps to solve this problem:
        1. generate the dependency pairs that dependency A relies on dependency B
        2. use graph-based algorithm to sort the dependencies in a topological order

        For example, answering the question "What is the mayor of the capital of France?"
        the input dependencies are:
        - The capital of France
        - The mayor of this capital

        Then the dependency pairs are:
        - [(1, 0)]
        because the mayor of the capital of France relies on the capital of France

        Then the topological order is computed by the self._topological_sort function, which is a graph-based algorithm. The output is a list of indices of the dependencies in the topological order.
        In this case, the output is:
        [0, 1]

        The sorted dependencies are thus:
        - The capital of France
        - The mayor of this capital
        """

        # Step 1: generate the dependency pairs by prompting LLMs
        prompt = DEPENDENCY_PAIRS_GENERATOR.format(query=query, dependencies=dependencies)
        response, dependency_sorter_cost, dependency_sorter_cost_ind = run_llm_with_cost(
            llm_type=self.llm_type, 
            tokenizer=self.tokenizer,
            model=self.model,
            sys_prompt="",
            input_prompt=prompt,
            g_cost=dependency_sorter_cost,
            max_token=max_token,
        )
        result = fix_json_response(response)
        dependency_pairs = result["dependency_pairs"]

        # Step 2: use graph-based algorithm to sort the dependencies in a topological order
        try:
            sorted_dependencies = self._topological_sort(dependencies, dependency_pairs)
        except Exception as e:
            setattr(e, "dependency_sorter_cost", dependency_sorter_cost)
            setattr(e, "dependency_sorter_cost_ind", dependency_sorter_cost_ind)
            raise
        return sorted_dependencies, dependency_sorter_cost, dependency_sorter_cost_ind
