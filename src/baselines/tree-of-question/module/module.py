import re
import json
import time
from typing import List, Dict
from prompt.prompt import *
from utils.model_utils import *
from module.utils import SubQuestion, AnswerResult, EvalMetrics, Document
from utils.utils import run_llm_with_cost


# ============================================================
# Section 3.1, Prompt B.2
# ============================================================

class QuestionDecomposer:
    """Decompose a complex question into sub-questions."""
    
    def __init__(self, llm_type, tokenizer, model):
        self.llm_type = llm_type
        self.tokenizer = tokenizer
        self.model = model

    def _parse_output(
        self, 
        output: str, 
        original_question: str
    ) -> List[SubQuestion]:
        decomp_match = re.search(r'Answer:\s*', output)
        if decomp_match:
            decomp_text = output[decomp_match.end():]
        else:
            decomp_text = output
        
        pattern = r'(\d+)\.\s*(.+?)(?=\n\d+\.|$)'
        matches = re.findall(pattern, decomp_text, re.DOTALL)
        
        if not matches:
            return [SubQuestion(text=original_question)]
        
        sub_questions = []
        for num, text in matches:
            text = text.strip()
            if not text:
                continue
            
            ans_matches = re.findall(r'\[ANS_(\d+)\]', text)
            depends_on = [int(m) - 1 for m in ans_matches] if ans_matches else None
            
            sub_questions.append(SubQuestion(text=text, depends_on=depends_on))
        
        if not sub_questions:
            return [SubQuestion(text=original_question)]
        
        return sub_questions

    def Decompose(self, question, cost_dict, max_token) -> List[SubQuestion]:
        question_plan_cost = cost_dict["question_plan_cost"]

        system_prompt = "" 
        input_prompt = Q_DECOMPOSER.format(question = question)
        
        generation, question_plan_cost, question_plan_cost_individual = run_llm_with_cost(
            llm_type=self.llm_type, 
            tokenizer=self.tokenizer,
            model=self.model,
            sys_prompt=system_prompt,
            input_prompt=input_prompt,
            g_cost=question_plan_cost,
            max_token=max_token,
        )
        sub_questions = self._parse_output(generation, original_question = question)
        cost_dict["question_plan_cost"] = question_plan_cost
        return sub_questions, cost_dict, question_plan_cost_individual


# ============================================================
# Section 3.2, Prompt B.1
# ============================================================

class AnswerIntegrator:
    """Extract an answer span from a document."""

    def __init__(self, llm_type, tokenizer, model):
        self.llm_type = llm_type
        self.tokenizer = tokenizer
        self.model = model

    def _parse_output(self, output: str) -> AnswerResult:
        try:
            json_match = re.search(r'\{.*\}', output, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group())
                return AnswerResult(
                    relevance=data.get("relevance", "irrelevant"),
                    answer_span=data.get("answer_span")
                )
        except json.JSONDecodeError:
            pass
        return AnswerResult(relevance="irrelevant", answer_span=None)

    def Integrate(self, question, document, cost_dict, max_token) -> AnswerResult:
        """Extract an answer span from one document."""
        ans_span_cost = cost_dict["ans_span_cost"]

        system_prompt = ""

        doc_json = json.dumps({
            "title": document.title,
            "summary": document.content
        }, ensure_ascii=False)

        input_prompt = ANSWER_INTEGRATOR.format(question = question, document=doc_json)

        generation, ans_span_cost, ans_span_cost_individual = run_llm_with_cost(
            llm_type=self.llm_type, 
            tokenizer=self.tokenizer,
            model=self.model,
            sys_prompt=system_prompt,
            input_prompt=input_prompt,
            g_cost=ans_span_cost,
            max_token=max_token,
        )
        cost_dict["ans_span_cost"] = ans_span_cost
        return self._parse_output(generation), cost_dict, ans_span_cost_individual


    def fill_answer_placeholders(
        self, 
        text: str, 
        answers: Dict[int, str]
    ) -> str:
        """Replace [ANS_N] placeholders with dependency answers."""
        result = text
        for idx, answer in answers.items():
            placeholder = f"[ANS_{idx + 1}]"  # 1-indexed marker
            result = result.replace(placeholder, answer)
        return result



# ============================================================
# Section 3.3, Prompt B.3
# ============================================================

class QueryGenerator:
    """Rewrite a user question as a retrieval query."""
    
    def __init__(self, llm_type, tokenizer, model):
        self.llm_type = llm_type
        self.tokenizer = tokenizer
        self.model = model
    
    def Generate(self, question, cost_dict, max_token) -> str:
        """Generate an optimized retrieval query."""
        query_gen_cost = cost_dict["query_gen_cost"]

        system_prompt = "" 
        input_prompt = Q_GEN.format(question = question)

        generation, query_gen_cost, query_gen_cost_individual = run_llm_with_cost(
            llm_type=self.llm_type, 
            tokenizer=self.tokenizer,
            model=self.model,
            sys_prompt=system_prompt,
            input_prompt=input_prompt,
            g_cost=query_gen_cost,
            max_token=max_token,
        )
        cost_dict["query_gen_cost"] = query_gen_cost
        new_query = generation.replace("Query: ", "").strip()
        return new_query, cost_dict, query_gen_cost_individual


# ============================================================
# Query Evaluator (Section 3.4, Prompt B.4)
# ============================================================

class QueryEvaluator:
    """Evaluate query-response quality."""
    
    def __init__(self, llm_type, tokenizer, model):
        self.llm_type = llm_type
        self.tokenizer = tokenizer
        self.model = model
    
    def _parse_output(self, output: str) -> EvalMetrics:
        """Parse an LLM response following Prompt B.4."""
        try:
            json_match = re.search(r'\{.*\}', output, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group())
                
                coherence = float(data.get("semantic_coherence", 5))
                answerability = float(data.get("answerability", 50))

                overall = data.get("overall_assessment")
                validity = data.get("response_validity")                

                if overall is not None:
                    overall = float(overall)
                if validity is not None:
                    if isinstance(validity, str):
                        validity = validity.lower() == "true"
                    else:
                        validity = bool(validity)

                return EvalMetrics.from_llm_output(
                    coherence=coherence,
                    answerability=answerability,
                    overall=overall,
                    validity=validity
                )
        except (json.JSONDecodeError, ValueError, TypeError):
            pass
        return EvalMetrics.default()
    
    def is_positive(self, metrics: EvalMetrics) -> bool:
        """
        Section 3.4:
        "Response Validity indicator, a binary (true/false) metric 
        that determines the adequacy of responses"
        """
        return metrics.response_validity

    def Evaluate(self, original_question, response, cost_dict, max_token) -> EvalMetrics:
        """Evaluate response quality."""
        eval_cost = cost_dict["eval_cost"]

        system_prompt = ""
        input_prompt = Q_EVAL.format(question = original_question, response = response)

        generation, eval_cost, eval_cost_individual = run_llm_with_cost(
            llm_type=self.llm_type, 
            tokenizer=self.tokenizer,
            model=self.model,
            sys_prompt=system_prompt,
            input_prompt=input_prompt,
            g_cost=eval_cost,
            max_token=max_token,
        )
        cost_dict["eval_cost"] = eval_cost

        metrics = self._parse_output(generation)
        return metrics, cost_dict, eval_cost_individual


# ============================================================
# Response Generator
# ============================================================

class ResponseGenerator:
    def __init__(self, llm_type, tokenizer, model, dataset_name):
        self.llm_type = llm_type
        self.tokenizer = tokenizer
        self.model = model
        self.dataset_name = dataset_name

    def _format_documents(self, documents: List[Document]) -> str:
        formatted = []
        for i, doc in enumerate(documents, 1):
            formatted.append(f"[doc {i}] {doc.title}\n{doc.content}")
        return "\n\n".join(formatted)

    def Generate(self, question, query, documents, is_root, cost_dict, max_token) -> str:
        """Generate a document-grounded response."""
        g_cost = cost_dict["g_cost"]

        docs_text = self._format_documents(documents)
        input_prompt = ANSWER_GEN.format(question=question, query=query, documents=docs_text)
        system_prompt = ROOT_SYS_PROMPT[self.dataset_name] if is_root else LEAF_SYS_PROMPT

        generation, g_cost, g_cost_individual = run_llm_with_cost(
            llm_type=self.llm_type, 
            tokenizer=self.tokenizer,
            model=self.model,
            sys_prompt=system_prompt,
            input_prompt=input_prompt,
            g_cost=g_cost,
            max_token=max_token,
        )
        cost_dict["g_cost"] = g_cost
        return generation.strip(), cost_dict, g_cost_individual

    def Final_Generate(self, original_question, context, cost_dict, max_token):
        input_prompt = f"Current Question {original_question}\n\nsub-QA pairs for the current question:\n"
        input_prompt += context
        input_prompt += ROOT_PROMPT[self.dataset_name]
        input_prompt += "\nAnswer:"

        system_prompt = ROOT_SYS_PROMPT[self.dataset_name]
        
        g_cost = cost_dict["g_cost"]

        generation, g_cost, g_cost_individual = run_llm_with_cost(
            llm_type=self.llm_type, 
            tokenizer=self.tokenizer,
            model=self.model,
            sys_prompt=system_prompt,
            input_prompt=input_prompt,
            g_cost=g_cost,
            max_token=max_token,
        )
        cost_dict["g_cost"] = g_cost
        return generation.strip(), cost_dict, g_cost_individual
