from prompt.prompt import *
from utils.model_utils import *
from utils.utils import *


class Generator:
    def __init__(self, llm_type, tokenizer, model, dataset_name):
        self.llm_type = llm_type
        self.tokenizer= tokenizer
        self.model = model
        self.dataset_name = dataset_name

    def generate_reasoning_plan(self, query, q_plan_cost, max_token):
        input_prompt = REASONING_PLAN_PROMPT.format(query=query)

        return run_llm_with_cost(
            llm_type=self.llm_type, 
            tokenizer=self.tokenizer,
            model=self.model,
            sys_prompt="",
            input_prompt=input_prompt,
            g_cost=q_plan_cost,
            max_token=max_token,
        )

    def generate_subquery(self, query, parent_qa_pairs, rewrite_cost, max_token):
        input_prompt = DYNAMIC_SUBQUERY_GENERATION_PROMPT.format(query=query, parent_qa_pairs=parent_qa_pairs)

        return run_llm_with_cost(
            llm_type=self.llm_type, 
            tokenizer=self.tokenizer,
            model=self.model,
            sys_prompt="",
            input_prompt=input_prompt,
            g_cost=rewrite_cost,
            max_token=max_token,
        )

    def generate_subanswer(self, question, query, known_answers, retrievals, g_cost, max_token):
        ret_input_prompt = ""
        for docs in retrievals:
            ret_input_prompt += f"title: {docs['title']}\ncontent:{docs['content']}\n\n"

        input_prompt = ANSWER_GEN.format(question=question, query=query, known_answers=known_answers, retrievals=ret_input_prompt)

        return run_llm_with_cost(
            llm_type=self.llm_type, 
            tokenizer=self.tokenizer,
            model=self.model,
            sys_prompt=LEAF_SYS_PROMPT,
            input_prompt=input_prompt,
            g_cost=g_cost,
            max_token=max_token,
        )

    def generate_final_answer(self, query, parent_qa_pairs, g_cost, max_token):
        input_prompt = f"Current Question {query}\n\nsub-QA pairs for the current question:\n"
        input_prompt += parent_qa_pairs
        input_prompt += ROOT_PROMPT[self.dataset_name]
        input_prompt += "\nAnswer:"

        return run_llm_with_cost(
            llm_type=self.llm_type, 
            tokenizer=self.tokenizer,
            model=self.model,
            sys_prompt=ROOT_SYS_PROMPT[self.dataset_name],
            input_prompt=input_prompt,
            g_cost=g_cost,
            max_token=max_token,
        )

