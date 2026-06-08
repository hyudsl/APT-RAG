from typing import Dict
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from rt_utils import *
from module.prompt import *
from config import *


#---------------------------------------------------- Question Structure Analysis -----------------------------------------------
class QuestionStructureAnalyzer:  
    def __init__(self, llm_type, tokenizer, model):
        self.llm_type = llm_type
        self.tokenizer = tokenizer
        self.model = model
    
    def analyze_question_structure(self, question, cost, max_token):
        """
        Analyze question and return its structure with limiting conditions
        """
        examples_string = QUESTION_STRUCTURE_EXAMPLES
        system_prompt = QUESTION_STRUCTURE_SYSTEM_PROMPT
        user_message = examples_string + "\nQuestion: \"" + question + "\"\n"
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message}
        ]

        tmp_cost = cost["analyze_q_str_cost"]
        api_response, tmp_cost, cost_individual = generate_response(messages, self.tokenizer, self.model, tmp_cost, max_tokens=max_token, temperature=0)
        cost["analyze_q_str_cost"] = tmp_cost


        if api_response:
            # Replace multiple consecutive newlines with a single newline for easier processing
            api_response = api_response.replace("\n\n", "\n")
            
            structure_start = api_response.find("So the structure is:")
            if structure_start != -1:
                structure_text = api_response[structure_start + len("So the structure is:"):].strip()
                return structure_text, cost, cost_individual
        
        return "[Core Query: Unknown Known Entities: Unknown Unknown Entities: Unknown]", cost, cost_individual


#---------------------------------------------------- Question Variant Generation -----------------------------------------------
class QuestionVariantGenerator:
    """
    Generate question variants that maintain the exact semantics of the original query
    to ensure all variants would receive the same answer as the original question

    Parameters:
    original_query (str): The original query
    history_queries (list, optional): List of historical queries to avoid duplicating
    variant_type (str, optional): Type of variant to generate (e.g., 'wh_transform', 'syntax_reform', 'random')
    num_variants (int, optional): Number of unique variants to generate

    Returns:
    list: List of generated question variants that preserve exact semantics
    """
    def __init__(self, llm_type, tokenizer, model):
        self.llm_type = llm_type
        self.tokenizer = tokenizer
        self.model = model

    def generate_question_variants(self, original_query, cost, num_variants=2, logger=None):
        system_prompt = QUESTION_VARIANT_SYSTEM_PROMPT
        user_prompt = QUESTION_VARIANT_USER_PROMPT.format(original_query=original_query, num_variants=num_variants)
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        tmp_cost = cost["gen_q_var_cost"]
        response_text, tmp_cost, cost_individual = generate_response(messages, self.tokenizer, self.model, tmp_cost, temperature=0.6)
        cost["gen_q_var_cost"] = tmp_cost

        # Parse the numbered list of variants
        variants = []
        for line in response_text.strip().split('\n'):
            line = line.strip()
            if line and (line[0].isdigit() or line.startswith('- ')):
                # Remove the numbering or bullet points
                variant = line.split('.', 1)[-1].strip() if '.' in line else line[2:].strip()
                variants.append(variant)
        
        # If no proper format is detected, treat the whole response as one variant
        if not variants and response_text.strip():
            variants = [response_text.strip()]
        
        # Return only the requested number of variants
        variants = variants[:num_variants]
        
        # Add the original query to the list of variants
        all_variants = [original_query] + variants
        
        logger.debug(f"Generated {len(variants)} question variants:")
        for i, variant in enumerate(all_variants):
            logger.debug(f"{i}. {variant}")
        
        return all_variants, cost, cost_individual


#---------------------------------------------------- Decomposition modules -----------------------------------------------
class SimilarExamplesFinder:
    def __init__(self, vectorizer=None):
        self.vectorizer = TfidfVectorizer()

    def find_similar_examples(self, question, examples, num_examples=3): # Vector Similarity
        example_questions = [ex["question"] for ex in examples]
        all_questions = example_questions + [question]
        vectors = self.vectorizer.fit_transform(all_questions)
        question_vector = vectors[-1]
        example_vectors = vectors[:-1]
        similarities = cosine_similarity(question_vector, example_vectors)[0]
        most_similar_indices = similarities.argsort()[-num_examples:][::-1]
        return [examples[i] for i in most_similar_indices]



class Decomposer:  
    def __init__(self, llm_type, tokenizer, model):
        self.llm_type = llm_type
        self.tokenizer= tokenizer
        self.model = model

    def decompose(self, modified_question, similar_examples, structure, cost, max_tokens=800, temperature=0, top_p=1.0, logger=None): 
        prompt = construct_prompt(modified_question, similar_examples, structure)

        system_message = DECOMP_SYSTEM_PROMPT

        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": prompt}
        ]
        
        tmp_cost = cost["decomp_cost"]
        response, tmp_cost, cost_individual = generate_response(
            messages, self.tokenizer, self.model, tmp_cost, max_tokens, temperature, top_p
        )
        cost["decomp_cost"] = tmp_cost
        decomposition = parse_decomposition_response(response, logger)

        return decomposition, cost, cost_individual



#---------------------------------------------------- Answer generation modules -----------------------------------------------

import spacy
try:
    nlp = spacy.load("en_core_web_sm")
except:
    import subprocess
    subprocess.call(["python", "-m", "spacy", "download", "en_core_web_sm"])
    nlp = spacy.load("en_core_web_sm")

class IterAnswerGenerator:  
    def __init__(self, llm_type, tokenizer, model, retriever):
        self.llm_type = llm_type
        self.tokenizer= tokenizer
        self.model = model
        self.retriever = retriever
    
    # Extract keywords function
    def extract_keywords(self, question: str) -> str:
        doc = nlp(question)
        keywords_with_positions = []
        matched_spans = set()

        name_patterns = [
            re.compile(r'\b[A-Z][a-z]+\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)?\b'),
            re.compile(r'\b[A-Z][a-z]+(?:-[A-Z]?[a-z]+)?\b'),
        ]
        
        for ent in doc.ents:
            ent_text = ent.text
            ent_start = question.find(ent_text)
            if ent_start != -1:
                matched_spans.add((ent_start, ent_start + len(ent_text)))
                keywords_with_positions.append((ent_text, ent_start))

        for pattern in name_patterns:
            for match in pattern.finditer(question):
                start, end = match.span()
                if not any(s <= start < e or s < end <= e for s, e in matched_spans):
                    matched_spans.add((start, end))
                    keywords_with_positions.append((match.group(), start))

        important_pos = {"NOUN", "PROPN", "ADJ", "VERB", "NUM"}
        for token in doc:
            if token.pos_ in important_pos and not token.is_stop:
                token_start = token.idx
                if not any(s <= token_start < e for s, e in matched_spans):
                    matched_spans.add((token_start, token_start + len(token.text)))
                    keywords_with_positions.append((token.text, token_start))

        keywords_with_positions.sort(key=lambda x: x[1])
        final_keywords = []
        seen_keywords = set()
        for kw, pos in keywords_with_positions:
            if not any(kw in other_kw and kw != other_kw for other_kw in seen_keywords):
                final_keywords.append(kw)
                seen_keywords.add(kw)
        return " ".join(final_keywords)

    # Parse generated text
    def parse_generated_text(self, generated_text, logger) -> Dict[str, str]:
        """
        Parse generated text, robustly handling various formats, including multi-line text and different marker forms.
        
        Args:
            generated_text: Raw text generated by API
            
        Returns:
            Dictionary containing cot and answer
        """
        # Normalize text - handle various line break formats
        text = generated_text.replace('\r\n', '\n').replace('\r', '\n')
        
        # Try multiple possible answer markers
        answer_markers = [
            "so the answer is:", 
            "So the answer is:", 
            "the answer is:", 
            "The answer is:",
            "FINAL ANSWER:"
        ]
        
        # Find CoT start position
        cot_markers = ["cot:", "COT:", "REASONING:", "Reasoning:"]
        cot_start = -1
        used_cot_marker = ""
        
        for marker in cot_markers:
            pos = text.find(marker)
            if pos != -1:
                cot_start = pos
                used_cot_marker = marker
                break
        
        # Find answer marker position
        answer_start = -1
        used_marker = ""
        
        for marker in answer_markers:
            # Use case-insensitive search
            pos = text.lower().find(marker.lower())
            if pos != -1:
                answer_start = pos
                used_marker = text[pos:pos+len(marker)]  # Preserve original case
                break
        
        # Process text based on found markers
        if cot_start != -1 and answer_start != -1 and cot_start < answer_start:
            # Found both CoT and answer markers
            cot = text[cot_start + len(used_cot_marker):answer_start].strip()
            answer = text[answer_start + len(used_marker):].strip()
            
            # Handle possible extra text in answer (like code parts)
            code_start = answer.find("import ")
            if code_start != -1:
                answer = answer[:code_start].strip()
        elif answer_start != -1:
            # Only found answer marker
            cot = text[:answer_start].strip()
            answer = text[answer_start + len(used_marker):].strip()
            
            # Handle possible extra text in answer
            code_start = answer.find("import ")
            if code_start != -1:
                answer = answer[:code_start].strip()
        elif cot_start != -1:
            # Only found CoT marker
            cot = text[cot_start + len(used_cot_marker):].strip()
            answer = "[none]"
        else:
            # No markers found, try to extract possible answer directly from text
            lines = text.split('\n')
            non_empty_lines = [line.strip() for line in lines if line.strip()]
            
            if non_empty_lines:
                # Check if last line looks like an answer
                last_line = non_empty_lines[-1]
                if len(last_line) < 100 and not last_line.startswith("import"):
                    answer = last_line
                    cot = '\n'.join(non_empty_lines[:-1])
                else:
                    cot = text
                    answer = "[none]"
            else:
                cot = text
                answer = "[none]"
        
        # Clean code segments
        if "import " in answer:
            answer = answer.split("import ")[0].strip()
        
        # Clean quotes
        answer = answer.strip('"\'')
        answer = re.sub(r'[*_#]', '', answer)

        logger.debug(f"[DEBUG] Parse result:\nCoT start position: {cot_start}, Answer start position: {answer_start}")
        logger.debug(f"[DEBUG] Extracted answer: '{answer}'")
        
        return {"cot": cot, "answer": answer}

    # Format complete response
    def format_full_response(self, question: str, document: str, generated_text: str, logger:None) -> str:
        result = self.parse_generated_text(generated_text, logger)
        cot = result["cot"]
        answer = result["answer"]
        return f"question: {question} documents: {document} \ncot: {cot} so the answer is: {answer}"

    def call_api_for_answer(self, question, documents, cost, max_tokens=2000, temperature=0, top_p=0.9, top_k=1):

        prompt = ITER_ANSWER_USER_PROMPT.format(document=documents, question=question)
        
        # Construct message list
        messages = [
            {"role": "system", "content": ITER_ANSWER_SYSTEM_PROMPT},
            {"role": "user", "content": prompt}
        ]
        
        tmp_cost = cost["answer_cost"]
        response, tmp_cost, cost_individual = generate_response(
            messages,
            tokenizer=self.tokenizer,
            model=self.model,
            cost=tmp_cost,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
        )
        cost["answer_cost"] = tmp_cost

        if response:
            return response, cost, cost_individual
        return "Error", cost, cost_individual

    # Generate optimized query
    def generate_refined_query(self, question, history_queries, cost):
        """
        Generate an optimized query based on the original question and historical queries.
        """
        # Organize historical queries as a string
        previous_queries = "\n".join(
            f"Query {i+1}: {query}"
            for i, query in enumerate(history_queries)
        )
        
        # Generate optimization query prompt
        new_query_prompt = REFINE_QUERY_USER_PROMPT.format(question=question, previous_queries=previous_queries)

        # Construct message list
        messages = [
            {"role": "system", "content": REFINE_QUERY_SYSTEM_PROMPT},
            {"role": "user", "content": new_query_prompt}
        ]
        
        tmp_cost = cost["refine_cost"]
        response, tmp_cost, cost_individual = generate_response(
            messages, self.tokenizer, self.model, tmp_cost, max_tokens=50, temperature=0.5
        )
        cost["refine_cost"] = tmp_cost

        if response:
            return response.strip(), cost, cost_individual
        else:
            raise Exception("Error generating new query")

    def answer_question(self, question, cost, logger:None, max_iterations: int = 4):
        """
        Answer questions using iterative method
        Returns:
            Formatted Q&A result, updated cost, and structured iterative log
        """
        # Initialize query history and iterative log
        history_queries = []
        iterative_log = {}
        iteration_idx = 1

        # First round: Use keyword query
        base_query = self.extract_keywords(question)
        current_query = f"{base_query} "
        history_queries.append(current_query)

        documents_list, cost["r_cost"], r_cost_individual = self.retriever.Retrieve(
            current_query, cost["r_cost"]
        )
        docuemnts_str = ""

        for i, item in enumerate(documents_list, start=1):
            str_ = f"-doc{i}:\ntitle:{item['title']}\ncontent:{item['content']}\n\n"
            docuemnts_str += str_

        documents = docuemnts_str

        # Generate multiple answers for the same documents
        all_responses = []
        generation_costs = []
        for i in range(SAMPLING_ITERATIONS):
            temp = RETRIEVE_TEMPERATURE  # Keep consistent temperature parameter
            generated_text, cost, gen_cost = self.call_api_for_answer(question, documents, cost, temperature=temp)
            all_responses.append(generated_text)
            generation_costs.append(gen_cost)
            logger.debug(f"\nIteration 1 - Generated response #{i+1}:\n{generated_text}")
        
        # Parse all responses and count answers
        answer_counter = {}
        all_results = []

        for response in all_responses:
            result = self.parse_generated_text(response, logger)
            all_results.append(result)
            answer = result["answer"].strip()
            
            if answer in answer_counter:
                answer_counter[answer] += 1
            else:
                answer_counter[answer] = 1

        # Select the most frequent answer
        most_common_answer = max(answer_counter.items(), key=lambda x: x[1]) if answer_counter else ("[none]", 0)
        logger.debug(f"\nAnswer statistics: {answer_counter}")
        logger.debug(f"Selected most frequent answer: {most_common_answer[0]} (appeared {most_common_answer[1]} times)")

        # Log this outer iteration
        iterative_log[str(iteration_idx)] = {
            "query": current_query,
            "retrieved_documents": [
                {
                    "rank": i,
                    "title": item.get("title"),
                    "content": item.get("content"),
                }
                for i, item in enumerate(documents_list, start=1)
            ],
            "responses": [
                {
                    "raw": resp,
                    "cot": res["cot"],
                    "answer": res["answer"],
                    "generation_cost": gen_cost,
                }
                for resp, res, gen_cost in zip(all_responses, all_results, generation_costs)
            ],
            "answer_stats": answer_counter,
            "selected_answer": most_common_answer[0],
            "r_cost": r_cost_individual,
        }

        # If the most common answer is not [none], return the corresponding complete response
        if "[none]" not in most_common_answer[0].lower():
            # Find the first response containing this answer
            for i, result in enumerate(all_results):
                if result["answer"].strip() == most_common_answer[0]:
                    full = self.format_full_response(question, documents, all_responses[i], logger)
                    return full, cost, iterative_log
        
        # If the most common answer is [none], continue iteration
        history_docs = documents  # Save previous documents for final result
        best_answer = most_common_answer[0]  # Save current best answer

        for iteration in range(2, max_iterations + 1):
            iteration_idx += 1
            # Generate new optimized query
            current_query, cost, refine_cost = self.generate_refined_query(question, history_queries, cost)
            history_queries.append(current_query)
            
            logger.debug(f"\nIteration {iteration} - Optimized query: {current_query}")

            documents_list, cost["r_cost"], r_cost_individual = self.retriever.Retrieve(
                current_query, cost["r_cost"]
            )
            docuemnts_str = ""

            for i, item in enumerate(documents_list, start=1):
                str_ = f"-doc{i}:\ntitle:{item['title']}\ncontent:{item['content']}\n\n"
                docuemnts_str += str_

            documents = docuemnts_str

            history_docs = documents

            # Generate multiple answers for new documents
            all_responses = []
            generation_costs = []
            for i in range(SAMPLING_ITERATIONS):
                temp = RETRIEVE_TEMPERATURE
                generated_text, cost, gen_cost = self.call_api_for_answer(question, documents, cost, temperature=temp)
                all_responses.append(generated_text)
                generation_costs.append(gen_cost)
                logger.debug(f"\nIteration {iteration} - Generated response #{i+1}:\n{generated_text}")

            # Parse all responses and count answers
            answer_counter = {}
            all_results = []
            
            for response in all_responses:
                result = self.parse_generated_text(response, logger)
                all_results.append(result)
                answer = result["answer"].strip()
                
                if answer in answer_counter:
                    answer_counter[answer] += 1
                else:
                    answer_counter[answer] = 1

            # Select the most frequent answer
            most_common_answer = max(answer_counter.items(), key=lambda x: x[1]) if answer_counter else ("[none]", 0)
            logger.debug(f"\n7 answer statistics: {answer_counter}")
            logger.debug(f"Selected most frequent answer: {most_common_answer[0]} (appeared {most_common_answer[1]} times)")

            # Log this outer iteration
            iterative_log[str(iteration_idx)] = {
                "query": current_query,
                "retrieved_documents": [
                    {
                        "rank": i,
                        "title": item.get("title"),
                        "content": item.get("content"),
                    }
                    for i, item in enumerate(documents_list, start=1)
                ],
                "responses": [
                {
                        "raw": resp,
                        "cot": res["cot"],
                        "answer": res["answer"],
                    "generation_cost": gen_cost,
                    }
                    for resp, res, gen_cost in zip(all_responses, all_results, generation_costs)
                ],
                "answer_stats": answer_counter,
                "selected_answer": most_common_answer[0],
                "refine_query_cost": refine_cost,
                "r_cost": r_cost_individual,
            }

            # If the most common answer is not [none], return the corresponding complete response
            if "[none]" not in most_common_answer[0].lower():
                # Find the first response containing this answer
                for i, result in enumerate(all_results):
                    if result["answer"].strip() == most_common_answer[0]:
                        full = self.format_full_response(question, documents, all_responses[i], logger)
                        return full, cost, iterative_log
            else:
                # Update best answer
                best_answer = most_common_answer[0]

        # Reached maximum iterations, use best answer
        modified_cot = "After multiple attempts, no conclusive answer was found in the documents. Providing the most frequent answer from multiple attempts."
        modified_generated_text = f"cot: {modified_cot} so the answer is: {best_answer}"
        
        full = self.format_full_response(question, history_docs, modified_generated_text, logger)
        return full, cost, iterative_log

    def generate_right_question_with_llm(self, parent_question, left_question, left_answer, original_right_question, cost, max_tokens=800, temperature=0.2, top_p=1.0):
        messages = [
            {
                "role": "system",
                "content": "You are an intelligent AI assistant tasked with generating appropriate follow-up questions based on context. Generate clear, relevant questions that flow naturally from previous information."
            },
            {
                "role": "user",
                "content": RIGHT_QUESTION_USER_PROMPT.format(parent_question=parent_question, left_question=left_question, left_answer=left_answer, original_right_question=original_right_question)
            }
        ]
    
        tmp_cost = cost["gen_right_q_cost"]
        response, tmp_cost, cost_individual = generate_response(
            messages=messages,
            tokenizer=self.tokenizer,
            model=self.model,
            cost=tmp_cost,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
        )
        cost["gen_right_q_cost"] = tmp_cost

        new_question = response.strip() if response else "Unable to generate a follow-up question."
    
        if "[answer_subquestion1]" in new_question or "[answer from" in new_question:
            new_question = new_question.replace("[answer_subquestion1]", "the aforementioned")
            new_question = new_question.replace("[answer from", "the aforementioned")
        
        return new_question, cost, cost_individual
    
    def answer_with_reasoning(self, question, documents, cost, max_tokens=10000, temperature=0):
        """
        Function that answers a question based on provided documents, allowing the LLM to use 
        reasoning and its internal knowledge in addition to document content.
        """
        prompt = f"""You are a knowledgeable AI assistant tasked with answering questions.

    You will be provided with a question and some documents that might contain relevant information.

    INSTRUCTIONS:
    1. Read the question and documents carefully
    2. You MUST use explicit step-by-step reasoning to arrive at your answer
    3. Your reasoning MUST rely on either:
    - Information from the provided documents, OR
    - Your internal knowledge when documents are insufficient
    4. Analyze the question from multiple angles and consider different interpretations
    5. When the documents contain relevant information, ensure you incorporate it in your reasoning
    6. When the documents are incomplete, use your knowledge to fill gaps through explicit reasoning
    7. You MUST ALWAYS provide a concrete answer - "I don't know", "None", or similar responses are NOT acceptable
    8. If uncertain, provide your best reasoned guess based on available information

    YOUR RESPONSE MUST STRICTLY FOLLOW THIS FORMAT:

    cot: [Your detailed step-by-step reasoning process using document information or internal knowledge]

    so the answer is: [Your final answer with NO additional decorations, explanations, or qualifiers - just the direct, concise answer]

    For example:
    - If the answer is "Paris", just write "Paris"
    - If the answer is a date, just write the date
    - If the answer is a person's name, just write the name
    - Do NOT add phrases like "I believe", "According to the documents", "The answer would be", etc.

    QUESTION: {question}

    DOCUMENTS:
    {documents}
    """

        # Construct message list
        messages = [
            {"role": "system", "content": 
                "You are a knowledgeable AI assistant that MUST provide answers through explicit reasoning. You MUST perform step-by-step reasoning using either document information or your internal knowledge to reach conclusions. You MUST ALWAYS provide a concrete answer - 'I don't know', 'None', or empty responses are NOT acceptable. If uncertain, provide your best reasoned guess based on available information. You MUST follow the exact output format: 'cot: [detailed reasoning] so the answer is: [direct answer with NO additional text]'"
            },
            {"role": "user", "content": prompt}
        ]
        
        tmp_cost = cost["answer_reason_cost"]
        response, tmp_cost, cost_individual = generate_response(
            messages,
            self.tokenizer,
            self.model,
            tmp_cost,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        cost["answer_reason_cost"] = tmp_cost

        # Ensure a valid response is returned
        if not response or response.strip() == "" or response.lower() == "none":
            # Fallback response if the model returns None or empty
            return "cot: The question requires a response even with limited information. Based on the available documents and general knowledge, I must provide my best assessment. so the answer is: [Best possible answer based on limited information]", cost, cost_individual
        
        return response, cost, cost_individual

    def direct_answer(self, question, cost, logger):
        query = self.extract_keywords(question) 
        
        documents_list, cost["r_cost"], r_cost_individual = self.retriever.Retrieve(
            query, cost["r_cost"]
        )
        docuemnts_str = ""

        for i, item in enumerate(documents_list, start=1):
            str_ = f"-doc{i}:\ntitle:{item['title']}\ncontent:{item['content']}\n\n"
            docuemnts_str += str_

        documents = docuemnts_str
        
        answers = []
        for _ in range(1): # for _ in range(SAMPLING_ITERATIONS):
            response, cost, reason_cost = self.answer_with_reasoning(question, documents, cost)
            logger.debug(response)
            parsed_answer = self.parse_generated_text(response, logger)['answer']
            answers.append(parsed_answer)
        
        # Count the frequency of each answer
        answer_counts = {}
        for answer in answers:
            if answer in answer_counts:
                answer_counts[answer] += 1
            else:
                answer_counts[answer] = 1
        
        # Sort answers by frequency (highest to lowest)
        sorted_answers = sorted(answer_counts.items(), key=lambda x: x[1], reverse=True)
        
        # Return the first non-"none" answer with highest frequency
        for answer, count in sorted_answers:
            if "none" not in answer.lower():
                return answer, cost, reason_cost, r_cost_individual
        
        # If all answers contain "none", return the most frequent one
        return sorted_answers[0][0], cost, reason_cost, r_cost_individual



class Generator:
    def __init__(self, llm_type, tokenizer, model):
        self.llm_type = llm_type
        self.tokenizer = tokenizer
        self.model = model

    def final_generate(self, question, sub_questions, cost):
        # get_final_answer
        prompt = construct_final_prompt(question, sub_questions)
        
        messages = [
            {"role": "system", "content": "You are an expert at answering questions based on provided subquestions and their answers."},
            {"role": "user", "content": prompt}
        ]

        tmp_cost = cost["final_ans_gen_cost"]
        response, tmp_cost, cost_individual = generate_response(
            messages, self.tokenizer, self.model, tmp_cost, max_tokens=800, temperature=0
        )
        cost["final_ans_gen_cost"] = tmp_cost
        return response, cost, cost_individual






