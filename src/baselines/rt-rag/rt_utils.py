import os
import re
import time
import torch
import logging
import datetime
import random
import numpy as np
from typing import List, Tuple

# CUDA_VISIBLE_DEVICES: set in main.py / main2.py before any imports


def _count_tokens(tokenizer, text: str) -> int:
    return len(tokenizer.encode(text, add_special_tokens=False))


def set_seed(seed_value=42):
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)  # if you are using multi-GPU.
    random.seed(seed_value)
    os.environ['PYTHONHASHSEED'] = str(seed_value)
    
    # The below two lines are for deterministic algorithm behavior in CUDA
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def print_all_nodes(node, indent="", logger=None):
    logger.debug(f"{indent}Node ID: {node.id}")
    logger.debug(f"{indent}Type: {node.type}")
    logger.debug(f"{indent}Question: {node.display_question}")
    if node.depends_on:
        logger.debug(f"{indent}Depends on: {node.depends_on}")
    if node.answer:
        logger.debug(f"{indent}Answer: {node.answer}")
    if node.left:
        logger.debug(f"{indent}Left child:")
        print_all_nodes(node.left, indent + "  ", logger)
    if node.right:
        logger.debug(f"{indent}Right child:")
        print_all_nodes(node.right, indent + "  ", logger)


def save_tree_stats(question, answer, original_height, final_height, file_path, success=True, logger=None):
    """
    Save tree statistics to a file
    
    Parameters:
    - question: The original question
    - answer: The final answer obtained
    - original_height: Initial height of the tree before expansion
    - final_height: Final height after expansion and solving
    - file_path: Path to save the stats
    - success: Whether the question was successfully answered
    """
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    
    # Format current timestamp
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Clean up the question and answer for single-line storage
    clean_question = question.replace('\n', ' ').strip()
    clean_answer = answer.replace('\n', ' ').strip() if answer else "[none]"
    success_str = "SUCCESS" if success else "FAILURE"
    
    # Prepare the statistics line
    stats_line = f"{timestamp}|{success_str}|{original_height}|{final_height}|{clean_question}|{clean_answer}\n"
    
    # Append to file
    with open(file_path, 'a', encoding='utf-8') as f:
        f.write(stats_line)
    
    logger.debug(f"Tree statistics saved to {file_path}")


#---------------------------------------------------- Parsing Response -----------------------------------------------
def parse_decomposition_response(response, logger):
    # Function to parse decomposition response remains unchanged
    type_match = re.search(r'So the Type is:\s*(\w+)', response, re.IGNORECASE)
    question_type = type_match.group(1) if type_match else "None"
    
    # Extract Subquestion 1, ensuring only a single line of question content
    subq1_match = re.search(r'So the Subquestion 1 is:\s*(.+)', response, re.DOTALL)
    subq1 = subq1_match.group(1).strip() if subq1_match else ""
    # Clean up possible next line content
    subq1 = subq1.split('\n')[0].strip()  # Take only the first line
    
    subq2 = ""
    if question_type.lower() != "none":
        # Extract Subquestion 2, ensuring only a single line of question content
        subq2_match = re.search(r'So the Subquestion 2 is:\s*(.+)', response, re.DOTALL)
        subq2 = subq2_match.group(1).strip() if subq2_match else ""
        subq2 = subq2.split('\n')[0].strip()  # Take only the first line
        if not subq2 and question_type.lower() != "none":
            question_type = "None"
            
        # Check if subq2 contains [answer_subquestion1], if so ensure type is Sequential
        if "[answer_subquestion1]" in subq2:
            if question_type.lower() != "sequential":
                logger.debug(f"Detected subquestion2 containing reference to subquestion1 answer, correcting type from {question_type} to Sequential.")
                question_type = "Sequential"
        # If type is Sequential but subq2 doesn't reference subq1's answer, issue a warning but don't modify the type
        elif question_type.lower() == "sequential" and "[answer_subquestion1]" not in subq2:
            logger.debug(f"Warning: Type is Sequential but subquestion2 doesn't reference subquestion1's answer.")
    
    return {"type": question_type, "subq1": subq1, "subq2": subq2}


#---------------------------------------------------- Tree Statistics Collection -----------------------------------------------
def get_all_nodes_postorder(node):
    """Get all nodes in post-order traversal"""
    if node is None:
        return []
    left_nodes = get_all_nodes_postorder(node.left)
    right_nodes = get_all_nodes_postorder(node.right)
    return left_nodes + right_nodes + [node]


def get_tree_statistics(root):
    """
    Calculate the statistics of a tree: height and node count
    Returns a tuple (height, node_count)
    """
    if root is None:
        return (0, 0)
    
    # Get all nodes
    all_nodes = get_all_nodes_postorder(root)
    node_count = len(all_nodes)
    
    # Calculate tree height
    def tree_height(node):
        if node is None:
            return 0
        return max(tree_height(node.left), tree_height(node.right)) + 1
    
    height = tree_height(root)
    
    return (height, node_count)


#---------------------------------------------------- Examples Database -----------------------------------------------
def get_examples_database():
    examples = [
        {
            "question": "What is the capital of France or Italy?",
            "structure": "[Core Query: What is the capital Known Entities: {Subject: France, Limitation: country}, {Subject: Italy, Limitation: country} Unknown Entities: {Subject: Capital of France, Limitation: city serving as French capital}, {Subject: Capital of Italy, Limitation: city serving as Italian capital}]",
            "type": "Parallel",
            "cot": "Let's think step by step. First, I need to determine if decomposition is necessary. The core query is 'What is the capital', but the question contains the logical operator 'OR' connecting two countries: France and Italy. The known entities are the countries France and Italy. Looking at the unknown entities, I need to identify the capital city of each country. The logical 'OR' suggests the user wants information about either one or both capitals. I can deliberately IGNORE other cities in these countries or government structures - while these would provide context about the countries, they don't help identify their capitals. This requires parallel decomposition to independently determine each capital before presenting the information about both possibilities.",
            "subq1": "What is the capital of France?",
            "subq2": "What is the capital of Italy?"
        },
        {
            "question": "What language is spoken in both Switzerland and Luxembourg?",
            "structure": "[Core Query: What language is spoken in both Known Entities: {Subject: Switzerland, Limitation: country}, {Subject: Luxembourg, Limitation: country} Unknown Entities: {Subject: Languages of Switzerland, Limitation: officially spoken in Switzerland}, {Subject: Languages of Luxembourg, Limitation: officially spoken in Luxembourg}, {Subject: Common languages, Limitation: spoken in both countries}]",
            "type": "Parallel",
            "cot": "Let's think step by step. First, I need to evaluate if decomposition is necessary. The core query asks 'What language is spoken in both', seeking languages common to two countries. The known entities are Switzerland and Luxembourg. The logical 'BOTH' indicates I need to find the intersection of two sets. Looking at the unknown entities, I need to identify languages spoken in Switzerland and languages spoken in Luxembourg, then determine which appear in both sets. I can deliberately IGNORE dialect variations or historical language development - while linguistically interesting, they don't help identify which official languages are shared between the countries. This requires parallel decomposition to independently determine each set of languages before finding their intersection.",
            "subq1": "What languages are spoken in Switzerland?",
            "subq2": "What languages are spoken in Luxembourg?"
        },
        {
            "question": "Which actor starred in The Godfather and Apocalypse Now?",
            "structure": "[Core Query: Which actor starred Known Entities: {Subject: The Godfather, Limitation: film}, {Subject: Apocalypse Now, Limitation: film} Unknown Entities: {Subject: Godfather cast, Limitation: actors who starred in The Godfather}, {Subject: Apocalypse Now cast, Limitation: actors who starred in Apocalypse Now}, {Subject: Common actors, Limitation: appeared in both films}]",
            "type": "Parallel",
            "cot": "Let's think step by step. First, I need to determine if decomposition is necessary. The core query asks 'Which actor starred', seeking performers common to two films. The known entities are The Godfather and Apocalypse Now. The logical 'AND' indicates I need to find the intersection of two sets. Looking at the unknown entities, I need to identify actors who starred in The Godfather and actors who starred in Apocalypse Now, then determine who appears in both casts. I can deliberately IGNORE directors, release dates, or plot details - while these provide context about the films, they don't help identify which actors appeared in both. This requires parallel decomposition to independently determine each cast list before finding their intersection.",
            "subq1": "Which actors starred in The Godfather?",
            "subq2": "Which actors starred in Apocalypse Now?"
        },
        {
            "question": "When did the director of the film \"The Seventh Seal\" die?",
            "structure": "[Core Query: When did person die Known Entities: {Subject: Film, Limitation: \"The Seventh Seal\"} Unknown Entities: {Subject: Director identity, Limitation: of \"The Seventh Seal\"}, {Subject: Death date, Limitation: of identified director}, {Subject: Director career, Limitation: other works}, {Subject: Film details, Limitation: release date, critical reception}]",
            "type": "Sequential",
            "cot": "Let's think step by step. First, I'll determine if decomposition is necessary. The core query is 'When did person die', seeking a death date. The known entity is the film 'The Seventh Seal'. Looking at the unknown entities, I need director identity and death date to answer the question. I must first identify the director before finding their death date, as that's a clear dependency. However, I can deliberately IGNORE the director's career and film details - while they provide context about the director and film, they're not essential for answering when the director died. This requires sequential decomposition to first identify the director, then find their death date.",
            "subq1": "Who directed the film \"The Seventh Seal\"?",
            "subq2": "When did [answer_subquestion1] die?"
        },
        {
            "question": "Who is Boraqchin (Wife Of Ögedei)'s father-in-law?",
            "structure": "[Core Query: Who is person's father-in-law Known Entities: {Subject: Boraqchin, Limitation: Wife of Ögedei}, {Subject: Ögedei, Limitation: Boraqchin's husband} Unknown Entities: {Subject: Father-in-law identity, Limitation: father of Ögedei, spouse's father to Boraqchin}, {Subject: Family relationship chain, Limitation: spouse relationship connects Boraqchin to Ögedei's father}]",
            "type": "Sequential",
            "cot": "Let's think step by step. First, I need to determine if decomposition is necessary. The core query is 'Who is person's father-in-law'. The known entities are 'Boraqchin' and 'Ögedei'. Looking at the unknown entities, I need to identify Boraqchin's father-in-law. Since Boraqchin is explicitly identified as Ögedei's wife, her father-in-law would be Ögedei's father. I can deliberately IGNORE historical context about these figures - while this might provide interesting background, it doesn't help identify the specific person who is Ögedei's father. This requires sequential decomposition because finding Boraqchin's father-in-law depends on first identifying who Boraqchin's spouse is, then determining who that person's father was.",
            "subq1": "Who is Boraqchin's spouse?",
            "subq2": "Who is [answer_subquestion1]'s father?"
        },
        {
            "question": "Did the composer of \"Symphony No. 9\" (Choral Symphony) die after the painter of \"The Starry Night\"?",
            "structure": "[Core Query: Did person A die after person B Known Entities: {Subject: Symphony, Limitation: \"Symphony No. 9\" (Choral Symphony)}, {Subject: Painting, Limitation: \"The Starry Night\"} Unknown Entities: {Subject: Composer identity, Limitation: of Symphony No. 9}, {Subject: Painter identity, Limitation: of The Starry Night}, {Subject: Death date, Limitation: of identified composer}, {Subject: Death date, Limitation: of identified painter}, {Subject: Birth dates, Limitation: of both artists}, {Subject: Artistic styles, Limitation: of both artists}]",
            "type": "Parallel",
            "cot": "Let's think step by step. First, I need to determine if decomposition is necessary. The core query compares two death dates: 'Did person A die after person B'. Known entities are the works 'Symphony No. 9' and 'The Starry Night'. Examining the unknown entities, I need composer identity, painter identity, and their death dates to make the comparison. I can deliberately IGNORE birth dates and artistic styles - while these might be interesting for context, they don't contribute to determining who died after whom. This requires parallel decomposition since I need to find each death date independently before comparing them.",
            "subq1": "When did the composer of \"Symphony No. 9\" (Choral Symphony) die?",
            "subq2": "When did the painter of \"The Starry Night\" die?"
        },
        {
            "question": "Which instrument did the composer of \"Symphony No. 9\" (Choral Symphony) play as a child?",
            "structure": "[Core Query: Which instrument did person play as a child Known Entities: {Subject: Symphony, Limitation: \"Symphony No. 9\" (Choral Symphony)} Unknown Entities: {Subject: Composer identity, Limitation: of Symphony No. 9}, {Subject: Instrument, Limitation: played by identified composer during childhood}, {Subject: Musical education, Limitation: training of composer}, {Subject: Composition style, Limitation: characteristics of composer's work}]",
            "type": "Sequential",
            "cot": "Let's think step by step. First, I need to evaluate if decomposition is necessary. The core query asks 'Which instrument did person play as a child'. The known entity is 'Symphony No. 9'. Looking at the unknown entities, I need to identify the composer first, then determine what instrument they played as a child. I can deliberately IGNORE musical education and composition style - while relevant to the composer's development, they don't directly answer what specific instrument was played. This requires sequential decomposition because identifying the childhood instrument depends on first knowing who the composer was.",
            "subq1": "Who composed \"Symphony No. 9\" (Choral Symphony)?",
            "subq2": "Which instrument did [answer_subquestion1] play as a child?"
        },
        {
            "question": "In which calendar year did the literary figure responsible for penning the dystopian novel '1984' ultimately pass away?",
            "structure": "[Core Query: In which year did person pass away Known Entities: {Subject: Novel, Limitation: dystopian, titled '1984'} Unknown Entities: {Subject: Author identity, Limitation: of the novel '1984'}, {Subject: Death year, Limitation: of identified author}, {Subject: Other works, Limitation: by same author}, {Subject: Political views, Limitation: of author that influenced novel}]",
            "type": "Sequential",
            "cot": "Let's think step by step. First, I'll assess if decomposition is necessary. The core query seeks a death year with 'In which year did person pass away'. The known entity is the novel '1984'. Looking at the unknown entities, I need to identify the author first, then determine their death year. I can deliberately IGNORE other works and political views - while these provide context about the author's career and influences, they don't help identify when the author died. This requires sequential decomposition since finding the death year depends on first identifying the author.",
            "subq1": "Who authored the dystopian novel '1984'?",
            "subq2": "In which year did [answer_subquestion1] pass away?"
        },
        {
            "question": "When did the director of film Hypocrite (Film) die?",
            "structure": "[Core Query: When did person die Known Entities: {Subject: Film, Limitation: titled \"Hypocrite\"} Unknown Entities: {Subject: Director identity, Limitation: of film Hypocrite}, {Subject: Death date, Limitation: of identified director}, {Subject: Film release date, Limitation: when Hypocrite was released}, {Subject: Director's filmography, Limitation: other films by same director}]",
            "type": "Sequential",
            "cot": "Let's think step by step. First, I need to determine if decomposition is necessary. The core query is 'When did person die', asking for a death date. The known entity is the film 'Hypocrite'. Looking at the unknown entities, I need to identify the director before I can find their death date. I can deliberately IGNORE the film release date and director's filmography - while these provide context about the film and director's career, they don't help determine when the director died. This requires sequential decomposition because finding the death date depends on first identifying who directed the film.",
            "subq1": "Who directed the film Hypocrite (Film)?",
            "subq2": "When did [answer_subquestion1] die?"
        },
        {
            "question": "Are both Kurram Garhi and Trojkrsti located in the same country?",
            "structure": "[Core Query: Are both located in the same country Known Entities: {Subject: Kurram Garhi, Limitation: location name}, {Subject: Trojkrsti, Limitation: location name} Unknown Entities: {Subject: Country, Limitation: containing Kurram Garhi}, {Subject: Country, Limitation: containing Trojkrsti}, {Subject: Geographic features, Limitation: of both locations}, {Subject: Population data, Limitation: of both locations}]",
            "type": "Parallel",
            "cot": "Let's think step by step. First, I need to assess if decomposition is necessary. The core query asks 'Are both located in the same country', a comparison question. Known entities are locations 'Kurram Garhi' and 'Trojkrsti'. Looking at the unknown entities, I need to determine the country for each location to make the comparison. I can deliberately IGNORE geographic features and population data - while these provide context about the locations, they don't help determine which countries they're in. This requires parallel decomposition to independently determine each location's country before comparing them.",
            "subq1": "In which country is Kurram Garhi located?",
            "subq2": "In which country is Trojkrsti located?"
        },
        {
            "question": "Do director of film Coolie No. 1 (1995 Film) and director of film The Sensational Trial have the same nationality?",
            "structure": "[Core Query: Do person A and person B have the same nationality Known Entities: {Subject: Film A, Limitation: Coolie No. 1 (1995)}, {Subject: Film B, Limitation: The Sensational Trial} Unknown Entities: {Subject: Director A identity, Limitation: of Coolie No. 1}, {Subject: Director B identity, Limitation: of The Sensational Trial}, {Subject: Nationality, Limitation: of Director A}, {Subject: Nationality, Limitation: of Director B}, {Subject: Film genres, Limitation: of both films}, {Subject: Box office performance, Limitation: of both films}]",
            "type": "Parallel",
            "cot": "Let's think step by step. First, I need to determine if decomposition is necessary. The core query asks if two people share the same nationality. Known entities are the films 'Coolie No. 1' and 'The Sensational Trial'. Looking at the unknown entities, I need to identify each director and their nationalities to make the comparison. I can deliberately IGNORE film genres and box office performance - while these provide context about the films, they're irrelevant to the directors' nationalities. This requires parallel decomposition since I can find each director's nationality independently before comparing them.",
            "subq1": "What is the nationality of the director of Coolie No. 1 (1995 Film)?",
            "subq2": "What is the nationality of the director of The Sensational Trial?"
        },
        {
            "question": "Who was born first out of Martin Hodge and Ivania Martinich?",
            "structure": "[Core Query: Who was born first Known Entities: {Subject: Martin Hodge, Limitation: person for comparison}, {Subject: Ivania Martinich, Limitation: person for comparison} Unknown Entities: {Subject: Birth date, Limitation: of Martin Hodge}, {Subject: Birth date, Limitation: of Ivania Martinich}, {Subject: Professional accomplishments, Limitation: of both individuals}, {Subject: Nationality, Limitation: of both individuals}]",
            "type": "Parallel",
            "cot": "Let's think step by step. First, I need to determine if decomposition is necessary. The core query is 'Who was born first', a comparison of birth dates. Known entities are 'Martin Hodge' and 'Ivania Martinich'. Looking at the unknown entities, I need the birth dates of both individuals to determine who was born first. I can deliberately IGNORE professional accomplishments and nationalities - while these might provide context about who these people are, they're irrelevant to determining birth order. This requires parallel decomposition to independently find each birth date before comparing them.",
            "subq1": "When was Martin Hodge born?",
            "subq2": "When was Ivania Martinich born?"
        },
        {
            "question": "When did the director of film Laughter In Hell die?",
            "structure": "[Core Query: When did person die Known Entities: {Subject: Film, Limitation: titled \"Laughter In Hell\"} Unknown Entities: {Subject: Director identity, Limitation: of film Laughter In Hell}, {Subject: Death date, Limitation: of identified director}, {Subject: Film production details, Limitation: studio, budget}, {Subject: Director's other films, Limitation: filmography}]",
            "type": "Sequential",
            "cot": "Let's think step by step. First, I need to assess if decomposition is necessary. The core query is 'When did person die', asking for a death date. The known entity is the film 'Laughter In Hell'. Looking at the unknown entities, I need to identify the director before I can find their death date. I can deliberately IGNORE film production details and the director's other films - while these provide context about the film and director's career, they don't help determine when the director died. This requires sequential decomposition because finding the death date depends on first identifying who directed the film.",
            "subq1": "Who directed the film Laughter In Hell?",
            "subq2": "When did [answer_subquestion1] die?"
        },
        {
            "question": "Which film has the director died later, The Gal Who Took the West or Twenty Plus Two?",
            "structure": "[Core Query: Which film's director died later Known Entities: {Subject: Film A, Limitation: The Gal Who Took the West}, {Subject: Film B, Limitation: Twenty Plus Two} Unknown Entities: {Subject: Director A identity, Limitation: of The Gal Who Took the West}, {Subject: Director B identity, Limitation: of Twenty Plus Two}, {Subject: Death date, Limitation: of Director A}, {Subject: Death date, Limitation: of Director B}, {Subject: Film release dates, Limitation: of both films}, {Subject: Critical reception, Limitation: of both films}]",
            "type": "Parallel",
            "cot": "Let's think step by step. First, I need to evaluate if decomposition is necessary. The core query is 'Which film's director died later', comparing death dates. Known entities are the films 'The Gal Who Took the West' and 'Twenty Plus Two'. Looking at the unknown entities, I need to identify each director and their death dates to determine who died later. I can deliberately IGNORE film release dates and critical reception - while these provide context about the films, they don't help determine when the directors died. This requires parallel decomposition to independently find when each director died before comparing the dates.",
            "subq1": "When did the director of The Gal Who Took the West die?",
            "subq2": "When did the director of Twenty Plus Two die?"
        },
        {
            "question": "Who is the grandchild of Krishna Shah (Nepalese Royal)?",
            "structure": "[Core Query: Who is person's grandchild Known Entities: {Subject: Krishna Shah, Limitation: Nepalese Royal} Unknown Entities: {Subject: Child identity, Limitation: of Krishna Shah}, {Subject: Grandchild identity, Limitation: child of Krishna Shah's child}, {Subject: Royal lineage, Limitation: historical significance}, {Subject: Reign dates, Limitation: period of authority}]",
            "type": "Sequential",
            "cot": "Let's think step by step. First, I need to determine if decomposition is necessary. The core query is 'Who is person's grandchild'. The known entity is 'Krishna Shah'. Looking at the unknown entities, I need to identify Shah's children first, then their children (Shah's grandchildren). I can deliberately IGNORE royal lineage and reign dates - while these provide historical context about Shah's position, they don't help identify specific family relationships. This requires sequential decomposition because identifying grandchildren depends on first identifying children.",
            "subq1": "Who is the child of Krishna Shah (Nepalese Royal)?",
            "subq2": "Who is the child of [answer_subquestion1]?"
        },
        {
            "question": "What is the official currency of Brazil?",
            "structure": "[Core Query: What is the official currency Known Entities: {Subject: Brazil, Limitation: country} Unknown Entities: {Subject: Currency, Limitation: official for Brazil}, {Subject: Currency history, Limitation: previous currencies of Brazil}, {Subject: Exchange rate, Limitation: value relative to USD}, {Subject: Economic indicators, Limitation: inflation, GDP}]",
            "type": "None",
            "cot": "Let's think step by step. First, I need to evaluate if decomposition is truly necessary. The core query is 'What is the official currency'. The known entity is 'Brazil'. Looking at the unknown entities, I only need to identify Brazil's current official currency. I can deliberately IGNORE currency history, exchange rates, and economic indicators - while they provide context about Brazil's economy, they don't help identify what the current official currency is. This is a straightforward factual question that can be answered in one step without decomposition.",
            "subq1": "What is the official currency of Brazil?",
            "subq2": ""
        },
        {
            "question": "What city, where the creator of 'The Scream' spent most of his childhood, is now considered a major cultural center?",
            "structure": "[Core Query: What city is a cultural center Known Entities: {Subject: Artwork, Limitation: 'The Scream'} Unknown Entities: {Subject: Creator identity, Limitation: of 'The Scream'}, {Subject: Childhood city, Limitation: where identified creator spent most childhood}, {Subject: Cultural status, Limitation: of identified city in present day}, {Subject: Artist's technique, Limitation: painting style}, {Subject: Museum location, Limitation: where artwork is displayed}]",
            "type": "Sequential",
            "cot": "Let's think step by step. First, I need to determine if decomposition is necessary. The core query is 'What city is a cultural center'. The known entity is 'The Scream'. Looking at the unknown entities, I need to identify the creator of 'The Scream', then determine where they spent most of their childhood. I can deliberately IGNORE the artist's technique and museum location - while these provide context about the artwork, they don't help identify the childhood city. This requires sequential decomposition because identifying the city depends on first identifying the artist.",
            "subq1": "Who created 'The Scream'?",
            "subq2": "In which city did [answer_subquestion1] spend most of his childhood?"
        },
        {
            "question": "In which century did the composer, whose opera premiered in the same year as the French Revolution began, die?",
            "structure": "[Core Query: In which century did person die Known Entities: {Subject: French Revolution, Limitation: historical event with specific beginning year} Unknown Entities: {Subject: Composer identity, Limitation: whose opera premiered same year as French Revolution began}, {Subject: Death century, Limitation: of identified composer}, {Subject: Opera details, Limitation: title and musical style}, {Subject: Composer's nationality, Limitation: country of origin}]",
            "type": "Sequential",
            "cot": "Let's think step by step. First, I need to evaluate if decomposition is necessary. The core query is 'In which century did person die'. The known entity is the 'French Revolution'. Looking at the unknown entities, I need to identify which composer had an opera premiere coinciding with the start of the French Revolution, then determine in which century they died. I can deliberately IGNORE opera details and composer's nationality - while these provide context about the composer, they don't help determine when they died. This requires sequential decomposition because finding the death century depends on first identifying the specific composer.",
            "subq1": "Which composer's opera premiered in the same year as the French Revolution began?",
            "subq2": "In which century did [answer_subquestion1] die?"
        },
        {
            "question": "Which language, spoken by the indigenous people who first inhabited the region where Silicon Valley is now located, has fewer native speakers today?",
            "structure": "[Core Query: Which language has fewer speakers Known Entities: {Subject: Silicon Valley, Limitation: geographic region with specific indigenous history} Unknown Entities: {Subject: Indigenous languages, Limitation: spoken by first inhabitants of Silicon Valley region}, {Subject: Speaker counts, Limitation: current number of native speakers for each identified language}, {Subject: Cultural traditions, Limitation: of indigenous groups}, {Subject: Historical territories, Limitation: exact boundaries of indigenous lands}]",
            "type": "Sequential",
            "cot": "Let's think step by step. First, I need to determine if decomposition is necessary. The core query is 'Which language has fewer speakers'. The known entity is 'Silicon Valley'. Looking at the unknown entities, I first need to identify which indigenous languages were spoken in the Silicon Valley region, then compare their current speaker counts to determine which has fewer. I can deliberately IGNORE cultural traditions and historical territories - while these provide important context about the indigenous peoples, they don't help determine current speaker counts. This requires sequential decomposition because comparing speaker counts depends on first identifying the relevant languages.",
            "subq1": "What indigenous languages were spoken by the people who first inhabited the region where Silicon Valley is now located?",
            "subq2": "Which of [answer_subquestion1] has the fewest native speakers today?"
        },
        {
            "question": "Did the mathematician whose theorem is fundamental to modern calculus die before or after the astronomer who first proposed the heliocentric model?",
            "structure": "[Core Query: Did person A die before or after person B Known Entities: {Subject: Calculus theorem, Limitation: fundamental to modern calculus}, {Subject: Heliocentric model, Limitation: astronomical theory} Unknown Entities: {Subject: Mathematician identity, Limitation: created fundamental calculus theorem}, {Subject: Astronomer identity, Limitation: first proposed heliocentric model}, {Subject: Death date, Limitation: of identified mathematician}, {Subject: Death date, Limitation: of identified astronomer}, {Subject: Publications, Limitation: major works of both scientists}, {Subject: Academic positions, Limitation: institutions where they worked}]",
            "type": "Parallel",
            "cot": "Let's think step by step. First, I need to evaluate if decomposition is necessary. The core query is 'Did person A die before or after person B', comparing death dates. Known entities are 'calculus theorem' and 'heliocentric model'. Looking at the unknown entities, I need to identify both the mathematician and astronomer, then determine their death dates to make the comparison. I can deliberately IGNORE publications and academic positions - while these provide context about their careers, they don't help determine when they died. This requires parallel decomposition because I can find each death date independently before comparing them.",
            "subq1": "When did the mathematician whose theorem is fundamental to modern calculus die?",
            "subq2": "When did the astronomer who first proposed the heliocentric model die?"
        },
        {
            "question": "Which painting, created by an artist who studied under the same teacher as Leonardo da Vinci, is currently housed in the Louvre Museum?",
            "structure": "[Core Query: Which painting is in the Louvre Known Entities: {Subject: Leonardo da Vinci, Limitation: famous artist}, {Subject: Louvre Museum, Limitation: art institution} Unknown Entities: {Subject: Teacher, Limitation: of da Vinci}, {Subject: Other students, Limitation: studied under same teacher as da Vinci}, {Subject: Paintings, Limitation: created by identified students and housed in Louvre}, {Subject: Artistic techniques, Limitation: used by the artists}, {Subject: Historical period, Limitation: when artists were active}]",
            "type": "Sequential",
            "cot": "Let's think step by step. First, I need to determine if decomposition is necessary. The core query is 'Which painting is in the Louvre'. Known entities are 'Leonardo da Vinci' and 'Louvre Museum'. Looking at the unknown entities, I need to identify da Vinci's teacher, then other students of this teacher, then paintings by these artists in the Louvre. I can deliberately IGNORE artistic techniques and historical periods - while these provide context about the artists, they don't help identify which paintings are in the Louvre. This requires sequential decomposition because finding the paintings depends on first identifying the relevant artists.",
            "subq1": "Which artists studied under the same teacher as Leonardo da Vinci?",
            "subq2": "Which paintings created by [answer_subquestion1] are housed in the Louvre Museum?"
        },
        {
            "question": "What is the capital of the country where the inventor of the telephone spent his final years?",
            "structure": "[Core Query: What is the capital of country Known Entities: {Subject: Telephone, Limitation: invention} Unknown Entities: {Subject: Inventor identity, Limitation: of telephone}, {Subject: Country, Limitation: where identified inventor spent final years}, {Subject: Capital, Limitation: of identified country}, {Subject: Invention date, Limitation: when telephone was created}, {Subject: Other inventions, Limitation: by same inventor}]",
            "type": "Sequential",
            "cot": "Let's think step by step. First, I need to assess if decomposition is necessary. The core query is 'What is the capital of country'. The known entity is 'telephone'. Looking at the unknown entities, I need to identify the telephone inventor, then determine where they spent their final years, and finally identify that country's capital. I can deliberately IGNORE the invention date and other inventions - while these provide context about the inventor's achievements, they don't help identify where they spent their final years or that country's capital. This requires sequential decomposition because each step depends on the previous one.",
            "subq1": "In which country did the inventor of the telephone spend his final years?",
            "subq2": "What is the capital of [answer_subquestion1]?"
        },
        {
            "question": "Which musical instrument, played by the composer who wrote the most famous requiem while on his deathbed, was his primary instrument?",
            "structure": "[Core Query: Which instrument was primary Known Entities: {Subject: Requiem, Limitation: most famous, written on deathbed} Unknown Entities: {Subject: Composer identity, Limitation: wrote famous requiem on deathbed}, {Subject: Primary instrument, Limitation: of identified composer}, {Subject: Other compositions, Limitation: by same composer}, {Subject: Musical era, Limitation: period when composer was active}]",
            "type": "Sequential",
            "cot": "Let's think step by step. First, I need to evaluate if decomposition is necessary. The core query is 'Which instrument was primary'. The known entity is the 'famous requiem written on deathbed'. Looking at the unknown entities, I need to identify the composer who wrote this requiem, then determine their primary instrument. I can deliberately IGNORE other compositions and musical era - while these provide context about the composer's work and time period, they don't help identify their primary instrument. This requires sequential decomposition because identifying the instrument depends on first identifying the composer.",
            "subq1": "Which composer wrote the most famous requiem while on his deathbed?",
            "subq2": "What was [answer_subquestion1]'s primary musical instrument?"
        },
        {
            "question": "From which university did the physicist, whose equation unifies electricity and magnetism into a single theory, graduate?",
            "structure": "[Core Query: From which university did person graduate Known Entities: {Subject: Equation, Limitation: unifies electricity and magnetism} Unknown Entities: {Subject: Physicist identity, Limitation: created unifying equation}, {Subject: University, Limitation: where identified physicist graduated}, {Subject: Year of graduation, Limitation: when physicist completed studies}, {Subject: Other scientific contributions, Limitation: additional work by physicist}]",
            "type": "Sequential",
            "cot": "Let's think step by step. First, I need to determine if decomposition is necessary. The core query is 'From which university did person graduate'. The known entity is 'equation about electricity and magnetism'. Looking at the unknown entities, I need to first identify which physicist created this unifying equation, then determine their alma mater. I can deliberately IGNORE graduation year and other scientific contributions - while these provide context about the physicist's education and career, they don't help identify which university they attended. This requires sequential decomposition because finding the university depends on first identifying the physicist.",
            "subq1": "Which physicist's equation unifies electricity and magnetism into a single theory?",
            "subq2": "From which university did [answer_subquestion1] graduate?"
        },
        {
            "question": "Which disease, that caused the death of the author whose novel depicted a young orphan in Victorian England, was most prevalent in European cities of that era?",
            "structure": "[Core Query: Which disease was most prevalent Known Entities: {Subject: Victorian England, Limitation: historical period}, {Subject: Novel theme, Limitation: young orphan} Unknown Entities: {Subject: Author identity, Limitation: wrote novel about orphan in Victorian England}, {Subject: Disease, Limitation: caused identified author's death and prevalent in European cities of that era}, {Subject: Author's other works, Limitation: bibliography}, {Subject: Medical treatments, Limitation: available in Victorian era}]",
            "type": "Sequential",
            "cot": "Let's think step by step. First, I need to assess if decomposition is necessary. The core query is 'Which disease was most prevalent'. Known entities are 'Victorian England' and 'novel about orphan'. Looking at the unknown entities, I need to identify the author of novels about orphans in Victorian England, then determine what disease caused their death and was also prevalent in that era. I can deliberately IGNORE the author's other works and medical treatments - while these provide historical context, they don't help identify the specific disease that caused the author's death. This requires sequential decomposition because identifying the disease depends on first identifying the author.",
            "subq1": "Which author wrote a novel depicting a young orphan in Victorian England?",
            "subq2": "What disease, that caused the death of [answer_subquestion1], was most prevalent in European cities of the Victorian era?"
        },
        {
            "question": "In what year was the landmark building, designed by the architect who also created the famous glass pyramid, completed?",
            "structure": "[Core Query: In what year was building completed Known Entities: {Subject: Glass pyramid, Limitation: famous architectural work} Unknown Entities: {Subject: Architect identity, Limitation: created the glass pyramid}, {Subject: Landmark building, Limitation: designed by identified architect}, {Subject: Completion year, Limitation: of identified landmark building}, {Subject: Architectural style, Limitation: of both structures}, {Subject: Construction materials, Limitation: used in both structures}]",
            "type": "Sequential",
            "cot": "Let's think step by step. First, I need to determine if decomposition is necessary. The core query is 'In what year was building completed'. The known entity is 'glass pyramid'. Looking at the unknown entities, I need to identify the architect who created the glass pyramid, then their landmark building, and finally its completion year. I can deliberately IGNORE architectural style and construction materials - while these provide interesting context about the structures, they don't help identify the completion year. This requires sequential decomposition due to the dependencies between identifying the architect, the building, and then its completion year.",
            "subq1": "Which architect created the famous glass pyramid?",
            "subq2": "In what year was the landmark building designed by [answer_subquestion1] completed?"
        },
        {
            "question": "Are both Sagrada Familia and Notre-Dame Cathedral designated as UNESCO World Heritage sites?",
            "structure": "[Core Query: Are both designated as UNESCO World Heritage sites Known Entities: {Subject: Sagrada Familia, Limitation: architectural landmark}, {Subject: Notre-Dame Cathedral, Limitation: architectural landmark} Unknown Entities: {Subject: UNESCO status, Limitation: of Sagrada Familia}, {Subject: UNESCO status, Limitation: of Notre-Dame Cathedral}, {Subject: Construction history, Limitation: of both buildings}, {Subject: Architectural styles, Limitation: of both buildings}]",
            "type": "Parallel",
            "cot": "Let's think step by step. First, I need to evaluate if decomposition is necessary. The core query is 'Are both designated as UNESCO World Heritage sites', a status comparison. Known entities are 'Sagrada Familia' and 'Notre-Dame Cathedral'. Looking at the unknown entities, I need to determine the UNESCO World Heritage status of each site. I can deliberately IGNORE construction history and architectural styles - while these provide context about the buildings, they don't help determine their UNESCO status. This requires parallel decomposition to check each status independently before comparing them.",
            "subq1": "Is Sagrada Familia designated as a UNESCO World Heritage site?",
            "subq2": "Is Notre-Dame Cathedral designated as a UNESCO World Heritage site?"
        },
        {
            "question": "Who was the first female astronaut to travel to space?",
            "structure": "[Core Query: Who was the first female astronaut Known Entities: {Subject: Space travel, Limitation: accomplished by females} Unknown Entities: {Subject: Astronaut identity, Limitation: female, first to travel to space}, {Subject: Launch date, Limitation: of first female space mission}, {Subject: Spacecraft, Limitation: used for first female space mission}, {Subject: Mission duration, Limitation: length of first female space flight}]",
            "type": "None",
            "cot": "Let's think step by step. First, I need to evaluate if decomposition is truly necessary. The core query is 'Who was the first female astronaut'. The known entity is 'space travel'. Looking at the unknown entities, I only need to identify which female was the first to travel to space. I can deliberately IGNORE launch date, spacecraft, and mission duration - while these provide interesting context about the historic mission, they don't help identify who the first female astronaut was. This is a straightforward factual question answerable in one step without decomposition.",
            "subq1": "Who was the first female astronaut to travel to space?",
            "subq2": ""
        },
        {
            "question": "When did the CEO of Tesla also become the owner of the social media platform previously known as Twitter?",
            "structure": "[Core Query: When became owner Known Entities: {Subject: CEO of Tesla, Limitation: person with specific role}, {Subject: Social media platform, Limitation: previously known as Twitter} Unknown Entities: {Subject: Acquisition date, Limitation: when CEO became owner of platform}, {Subject: Purchase price, Limitation: amount paid for acquisition}, {Subject: Platform changes, Limitation: modifications after acquisition}]",
            "type": "None",
            "cot": "Let's think step by step. First, I need to evaluate if decomposition is truly necessary. The core query is 'When became owner', seeking a date. Known entities are 'CEO of Tesla' and 'social media platform previously known as Twitter'. Looking at the unknown entities, I only need the acquisition date to answer when the ownership changed. I can deliberately IGNORE purchase price and platform changes - while these provide context about the acquisition and its aftermath, they don't help determine when the ownership transfer occurred. This is a straightforward factual question answerable in one step without decomposition.",
            "subq1": "When did the CEO of Tesla also become the owner of the social media platform previously known as Twitter?",
            "subq2": ""
        },
    ]
    return examples


def generate_response(messages, tokenizer, model, cost, max_tokens=800, temperature=0.2, top_p=1.0):
    """Generic API call function to replace original requests.post call"""

    model_input = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
        padding = 'longest').to(model.device)

    _ = model.eval()
    with torch.no_grad():
        if temperature == 0 or temperature==0.0:
            gen_kwargs = {
            "max_new_tokens": max_tokens,
            "temperature": None, 
            "do_sample": False,
            "top_p": None,
            "top_k": None,
            "eos_token_id": tokenizer.eos_token_id,
            "pad_token_id": tokenizer.eos_token_id,
            "output_scores": True,
            "output_logits": True,
            "return_dict_in_generate": True
        } 
        else:
            gen_kwargs = {
                "max_new_tokens": max_tokens,
                "temperature": temperature, 
                "top_p": top_p,
                "do_sample": True,
                "eos_token_id": tokenizer.eos_token_id,
                "pad_token_id": tokenizer.eos_token_id,
                "output_scores": True,
                "output_logits": True,
                "return_dict_in_generate": True
            } 
        t0 = time.perf_counter()
        if isinstance(model_input, torch.Tensor):
            output = model.generate(model_input, **gen_kwargs)
            prompt_len = model_input.shape[-1]
        else:
            output = model.generate(**model_input, **gen_kwargs)
            prompt_len = model_input["input_ids"].shape[-1]
        elapsed_sec = time.perf_counter() - t0

        sequences = output.sequences if hasattr(output, "sequences") else output[0]
        result = sequences[0][prompt_len:]
        response = tokenizer.decode(result, skip_special_tokens=True)

        input_tokens = _count_tokens(tokenizer, messages[0]['content']) + _count_tokens(tokenizer, messages[1]['content'])
        output_tokens = _count_tokens(tokenizer, response)

        call_ = cost.get("call", 0) + 1
        input_tokens_ = input_tokens + cost.get("input", 0)
        output_tokens_ = output_tokens + cost.get("output", 0)
        latency_ = elapsed_sec + cost.get("latency", 0)

        cost = {"call": call_, "input": input_tokens_, "output": output_tokens_, "latency": latency_ }
    
        cost_individual = {"call": 1, "input": input_tokens, "output": output_tokens, "latency": elapsed_sec}
    
    return response, cost, cost_individual










def get_logger(log_path=None):
    logger = logging.getLogger(log_path or "rt-rag")
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()

    formatter = logging.Formatter("%(message)s")
    if log_path:
        file_handler = logging.FileHandler(log_path, encoding="utf-8")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    return logger


def extract_answer(full_response):
    match = re.search(r'so the answer is:\s*(.*)', full_response, re.IGNORECASE)
    if match:
        raw_answer = match.group(1).strip()
        # Clean up the answer, remove extra symbols
        cleaned_answer = re.sub(r'["*]+$', '', raw_answer)
        # Handle [none] special case
        if cleaned_answer.lower() == '[none]':
            return '[none]'
        return cleaned_answer
    return "Answer not found"