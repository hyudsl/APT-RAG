import os
import logging
import re
import json
import time
import backoff
from openai import OpenAI
from ratelimit import limits, sleep_and_retry
from collections import Counter
from typing import List, Dict, Any
from colorama import Fore, Style, init
from config.config import (
    OPENAI_API_KEY,
    DEFAULT_MODEL,
    DEFAULT_MAX_TOKENS,
    CALLS_PER_MINUTE,
    PERIOD,
    MAX_RETRIES,
    RETRY_DELAY
)

# Initialize colorama
init()

# Configure logging
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

# Initialize token cost tracking
TOKEN_COST = {"prompt": 0, "completion": 0}

# Configure OpenAI
client = OpenAI(api_key=OPENAI_API_KEY)
os.environ["TOKENIZERS_PARALLELISM"] = "false"

REFLECTION_PROMPT = """Based on the question and the retrieved context, analyze:
1. Can you confidently answer the question with the given context and your knowledge?
2. If not, what specific information is missing?
3. Generate a focused search query to find the missing information.

Format your response as:
{
    "can_answer": true/false,
    "missing_info": "description of what information is missing",
    "subquery": "specific search query for missing information",
    "current_understanding": "brief summary of current understanding"
}
"""

@sleep_and_retry
@limits(calls=CALLS_PER_MINUTE, period=PERIOD)
@backoff.on_exception(
    backoff.expo,
    (Exception),
    max_tries=MAX_RETRIES,
    max_time=300
)
def get_response_with_retry(prompt: str, temperature: float = 0.0, print_cost: bool = False) -> str:
    """Get response from OpenAI API with retry logic."""
    global TOKEN_COST
    try:
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ]
        response = client.chat.completions.create(
            model=DEFAULT_MODEL,
            messages=messages,
            temperature=temperature,
            max_tokens=DEFAULT_MAX_TOKENS
        )
        # Update token costs
        if response.usage:
            TOKEN_COST["prompt"] += response.usage.prompt_tokens
            TOKEN_COST["completion"] += response.usage.completion_tokens
        if print_cost:
            logger.info(f"Prompt tokens: {response.usage.prompt_tokens}")
            logger.info(f"Completion tokens: {response.usage.completion_tokens}")
            logger.info(f"Total tokens: {response.usage.total_tokens}")
        return response.choices[0].message.content.strip()
    except Exception as e:
        logger.error(f"Error in get_response_with_retry: {str(e)}")
        return ""

def fix_json_response(response: str) -> str:
    """Fix JSON response from OpenAI API.
    Handles two common cases:
    1. Missing closing brace at the end
    2. Cut-off current_understanding field
    """
    # Remove any markdown code block markers and whitespace
    response = response.strip()
    response = response.replace('```json', '').replace('```', '')
    original_response = response  # Store original for comparison
    
    try:
        # First try to parse as is
        result = json.loads(response)
        return result
    except json.JSONDecodeError:
        # Case 1: Check if just missing closing brace
        if response.count('{') > response.count('}'):
            try:
                fixed_response = response + '}'
                result = json.loads(fixed_response)
                logger.info(f"{Fore.RED}Fixed JSON by adding closing brace:{Style.RESET_ALL}")
                logger.info(f"{Fore.RED}Original: {original_response}{Style.RESET_ALL}")
                logger.info(f"{Fore.GREEN}Fixed: {fixed_response}{Style.RESET_ALL}")
                return result
            except json.JSONDecodeError:
                pass
                
        # Case 2: Check for incomplete current_understanding
        try:
            # Find the last complete field before current_understanding
            pattern = r'(.*"current_understanding"\s*:\s*"[^"]*)("|$)'
            match = re.search(pattern, response, re.DOTALL)
            if match:
                fixed_response = match.group(1)  # Get everything up to the cut-off point
                if not fixed_response.endswith('"'):
                    fixed_response += '..."'  # Add ellipsis and close the quote
                if response.count('{') > response.count('}'):
                    fixed_response += '}'  # Add closing brace if needed
                try:
                    result = json.loads(fixed_response)
                    logger.info(f"{Fore.RED}Fixed truncated current_understanding:{Style.RESET_ALL}")
                    logger.info(f"{Fore.RED}Original: {original_response}{Style.RESET_ALL}")
                    logger.info(f"{Fore.GREEN}Fixed: {fixed_response}{Style.RESET_ALL}")
                    return result
                except json.JSONDecodeError:
                    pass
        except:
            pass
            
        logger.error(f"{Fore.RED}Failed to fix JSON response: {original_response}{Style.RESET_ALL}")
        return None

def normalize_answer(text: str) -> str:
    """Normalize answer text for comparison."""
    if not isinstance(text, str):
        return ""
    
    # Convert to lowercase
    text = text.lower()
    
    # Replace hyphen with space
    text = text.replace('-', ' ')
    
    # Remove punctuation
    text = re.sub(r'[^\w\s]', '', text)
    
    # Remove extra whitespace
    text = ' '.join(text.split())
    
    return text

def save_results(results: Dict, output_file: str, results_dir: str = 'result'):
    """Save evaluation results to file.
    
    Args:
        results: Dictionary containing results to save
        output_file: Filename for the results
        results_dir: Directory to save results in
    """
    os.makedirs(results_dir, exist_ok=True)
    output_path = os.path.join(results_dir, output_file)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    logger.info(f"Results saved to {output_path}")

def evaluate_with_llm(generated: str, gold: str) -> bool:
    """Use LLM to evaluate if the generated answer correctly answers the question."""
    if not isinstance(generated, str) or not isinstance(gold, str):
        return False
        
    prompt = f"""You are an expert evaluator. Please evaluate if the generated answer is correct by comparing it with the gold answer.

Generated answer: {generated}
Gold answer: {gold}

The generated answer should be considered correct if it:
1. Contains the key information from the gold answer
2. Is factually accurate and consistent with the gold answer
3. Does not contain any contradicting information

Respond with ONLY 'correct' or 'incorrect'.
Response:"""

    try:
        response = get_response_with_retry(prompt, temperature=0.0, print_cost=True)
        return response.strip().lower() == "correct"
    except Exception as e:
        logger.error(f"Error in LLM evaluation: {e}")
        return False

def string_based_evaluation(generated: str, gold: str) -> dict:
    """Evaluate string similarity between generated and gold answers.
    
    Args:
        generated: Generated answer string
        gold: Gold/ground truth answer string
        
    Returns:
        Dictionary containing accuracy, precision, recall metrics
    """
    # Normalize answers
    normalized_prediction = normalize_answer(generated)
    normalized_ground_truth = normalize_answer(gold)
    
    # Calculate accuracy
    accuracy = 1 if normalized_ground_truth in normalized_prediction else 0
    
    # Calculate precision and recall
    prediction_tokens = normalized_prediction.split()
    ground_truth_tokens = normalized_ground_truth.split()
    
    # Handle yes/no/noanswer cases
    if (normalized_prediction in ["yes", "no", "noanswer"] and 
        normalized_prediction != normalized_ground_truth) or \
       (normalized_ground_truth in ["yes", "no", "noanswer"] and 
        normalized_prediction != normalized_ground_truth):
        return {
            "accuracy": accuracy,
            "precision": 0,
            "recall": 0
        }
    
    # Calculate token overlap
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    
    # Calculate precision and recall
    precision = 1.0 * num_same / len(prediction_tokens) if prediction_tokens else 0
    recall = 1.0 * num_same / len(ground_truth_tokens) if ground_truth_tokens else 0
    
    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall
    } 