import logging
import re
import json
from colorama import Fore, Style, init


init() # Initialize colorama
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)


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
