import re
from typing import Counter
from rt_utils import *
from module.prompt import *
from module.tree import *
from config import *


import spacy
try:
    nlp = spacy.load("en_core_web_sm")
except:
    import subprocess
    subprocess.call(["python", "-m", "spacy", "download", "en_core_web_sm"])
    nlp = spacy.load("en_core_web_sm")



class RTRAGFramework:
    def __init__(self, q_struc_analyzer, q_variant_generator, retriever, similar_ex_finder, decomposer, iter_ans_generator, generator):
        self.q_struc_analyzer = q_struc_analyzer
        self.q_variant_generator = q_variant_generator
        self.retriever = retriever
        self.similar_ex_finder = similar_ex_finder
        self.decomposer = decomposer
        self.iter_ans_generator = iter_ans_generator
        self.generator = generator
        # Run-level cost trace (kept even when nodes/trees are regenerated).
        self.cost_trace = []
        self.latest_cost_trace = []
        self.latest_tree_bundle = {}

    def _log_cost_event(self, cost_type, cost_individual, **context):
        if cost_individual is None:
            return
        event = {"cost_type": cost_type, "cost": cost_individual}
        event.update(context)
        self.cost_trace.append(event)

    #---------------------------------------------------- Tree Building and Traversal -----------------------------------------------
    def build_question_tree(self, question, cost, max_tokens=800, temperature=0.2, top_p=1.0, frequency_penalty=0.0, 
                        presence_penalty=0.0, examples_db=None, num_examples=3, depth=0, 
                        parent=None, is_left_child=True, max_height=3, placeholder_answers=None, logger=None):
        """
        Build a question tree through recursive decomposition
        
        Parameters:
        - question: The question to decompose
        - api_url, max_tokens, temperature, top_p, frequency_penalty, presence_penalty: API parameters
        - examples_db: Database of examples for finding similar examples
        - num_examples: Number of similar examples to use
        - depth: Current depth in the tree (0 for root)
        - parent: Parent node
        - is_left_child: Whether the current node is a left child
        - max_height: Maximum allowed height of the tree (1 = root only)
        - placeholder_answers: Dictionary of previously computed answers
        
        Returns:
        - node: The root node of the built tree
        """
        global global_node_counter
        
        # Initialize placeholder_answers dictionary for storing replacement values
        if placeholder_answers is None:
            placeholder_answers = {}

        # Check if we've exceeded the maximum height
        # depth starts at 0 for root, so if max_height is 3, we allow depths 0, 1, and 2
        if depth >= max_height:
            
            # Create a leaf node with a unique ID using the global counter
            leaf_node = QuestionNode(
                question=question, 
                q_type="None",
                subq1=question, 
                subq2="",
                parent=parent, 
                is_left_child=is_left_child
            )
            leaf_node.meta["leaf_reason"] = "max_height"
            leaf_node.meta["depth"] = depth

            # Handle replacement markers
            if parent and "[answer_subquestion1]" in leaf_node.question:
                if parent.type == "Sequential" and not is_left_child:
                    leaf_node.depends_on = parent.left.id if parent.left else None
                    leaf_node.display_question = leaf_node.question.replace("[answer_subquestion1]", f"[answer from {leaf_node.depends_on}]")

            elif parent and "[answer from" in leaf_node.question:
                match = re.search(r'\[answer from (.*?)\]', leaf_node.question)
                if match:
                    leaf_node.depends_on = match.group(1)
    
            return leaf_node, cost
        
        # Handle replacement markers, replace if replacement values are available
        modified_question = question
        depends_on = None

        if "[answer_subquestion1]" in question:
        
            if parent and parent.type == "Sequential" and not is_left_child and parent.left:
                depends_on = parent.left.id
            
                # Check if there's a replacement value available
                if depends_on in placeholder_answers:
                    modified_question = question.replace("[answer_subquestion1]", placeholder_answers[depends_on])
                    
        elif "[answer from" in question:
            
            match = re.search(r'\[answer from (.*?)\]', question)
            if match:
                depends_on = match.group(1)
            
                # Check if there's a replacement value available
                if depends_on in placeholder_answers:
                    modified_question = question.replace(f"[answer from {depends_on}]", placeholder_answers[depends_on])
                
        
        # If the question still contains replacement markers and there's no replacement value available, create a node to be replaced
        if ("[answer_subquestion1]" in modified_question or "[answer from" in modified_question) and not "[answer from" in modified_question:
        
            node = QuestionNode(
                question=question, 
                q_type="None",
                subq1=question, 
                subq2="",
                parent=parent, 
                is_left_child=is_left_child
            )
            
            if depends_on:
                node.depends_on = depends_on
            
                if "[answer_subquestion1]" in node.question:
                    node.display_question = node.question.replace("[answer_subquestion1]", f"[answer from {node.depends_on}]")
            node.meta["leaf_reason"] = "placeholder"

            return node, cost
        
        # If examples_db is None, get the examples database
        if examples_db is None:
            examples_db = get_examples_database()


        # ------------------------- 1: Question decomposition -------------------------#

        # Analyze question structure and build decomposition prompt 
    
        structure, cost, cost_individual = self.q_struc_analyzer.analyze_question_structure(modified_question, cost, max_token=800)
        
        logger.debug("++++++++++++++++++++++++")
        logger.debug(f"Question structure analysis (depth={depth}):, {structure}")
        

        similar_examples = self.similar_ex_finder.find_similar_examples(modified_question, examples_db, num_examples)

        decomposition, cost, decomp_cost_individual = self.decomposer.decompose(
            modified_question, similar_examples, structure, 
            cost, max_tokens, temperature, top_p, logger
        )
        logger.debug(f"Decomposition response (depth={depth}):, {structure}")

        # -----------------------------------------------------------------------------#


        # Create a new node with a unique ID
        node = QuestionNode(
            question=question, 
            q_type=decomposition['type'],
            subq1=decomposition['subq1'], 
            subq2=decomposition['subq2'],
            parent=parent, 
            is_left_child=is_left_child
        )
        node.meta["structure"] = structure
        node.meta["decomposition"] = {
            "type": decomposition["type"],
            "subq1": node.subq1_text,
            "subq2": node.subq2_text,
        }
        node.meta["depth"] = depth
        node.meta["analyze_q_str_cost"] = cost_individual
        node.meta["decomp_cost"] = decomp_cost_individual
        self._log_cost_event(
            "analyze_q_str_cost",
            cost_individual,
            node_id=node.id,
            depth=depth,
            question=modified_question,
        )
        self._log_cost_event(
            "decomp_cost",
            decomp_cost_individual,
            node_id=node.id,
            depth=depth,
            question=modified_question,
        )
        
        if parent:
            logger.debug(f"Parent node: {parent.id}")
        
        # Use the modified question as the display question
        if modified_question != question:
            node.display_question = modified_question
            
        
        # Handle replacement markers
        if depends_on:
            node.depends_on = depends_on


        # Check if we need to stop expansion at this level to respect max_height
        # If we're at depth max_height-1, the next level would be at max_height
        # So we force this node to be a leaf by setting its type to "None"
        if depth >= max_height - 1:
            # Force node type to be "None" to prevent further decomposition
            node.type = "None"
            node.meta["leaf_reason"] = "max_height"
            
            return node, cost

        if node.type != "None":
        
            # First handle the left subtree
        
            left_node , cost = self.build_question_tree(
                node.subq1_text, cost, max_tokens, temperature, top_p,
                frequency_penalty, presence_penalty, examples_db, num_examples,
                depth + 1, node, True, max_height, placeholder_answers, logger
            )
            node.left = left_node

            if node.left:
                logger.debug(f"Left child node id: {node.left.id}, type: {node.left.type}")
            
            # Handle the right subtree
            if node.type == "Sequential":
                
                # For sequential type, check if the right subquestion contains replacement markers
                right_question = node.subq2_text
                
                right_node, cost = self.build_question_tree(
                    right_question, cost, max_tokens, temperature, top_p,
                    frequency_penalty, presence_penalty, examples_db, num_examples,
                    depth + 1, node, False, max_height, placeholder_answers, logger
                )
                node.right = right_node
                
                if node.right:
                    logger.debug(f"Right child node id: {node.right.id}, type: {node.right.type}")
                
                # Set up dependency relationships for sequential type
                if node.right and node.left:
                    
                    if node.right.depends_on is None and "[answer_subquestion1]" in node.right.question:
                        node.right.depends_on = node.left.id
                        node.right.display_question = node.right.question.replace("[answer_subquestion1]", f"[answer from {node.right.depends_on}]")
                        
            else:  # Parallel type
                
                right_node, cost = self.build_question_tree(
                    node.subq2_text, cost, max_tokens, temperature, top_p,
                    frequency_penalty, presence_penalty, examples_db, num_examples,
                    depth + 1, node, False, max_height, placeholder_answers, logger
                )
                node.right = right_node
                
                if node.right:
                    logger.debug(f"Right child node id: {node.right.id}, type: {node.right.type}")

        # Simple tree height and node count estimation
        has_children = node.left is not None or node.right is not None
        
        if has_children:
            logger.debug(f"Left child: {'exists' if node.left else 'does not exist'}, right child: {'exists' if node.right else 'does not exist'}")
        
        
        return node, cost

    #---------------------------------------------------- Tree Solving -----------------------------------------------
    def build_enhanced_right_subtree(self, original_question, left_answer, cost, max_tokens=800, 
                                temperature=0.2, top_p=1.0, frequency_penalty=0.0, presence_penalty=0.0, 
                                examples_db=None, num_examples=3, max_height=3,
                                num_variants=2, trees_per_variant=3, logger=None):
        """
    Build multiple trees for the right subtree of Sequential type, following the same logic as the main function

    Parameters:

    -original_question: Original right subtree problem

    -left_answer: The answer of the left subtree, used to replace [answer_subquestion1]

    Other parameters are the same as build_question_tree


    Returns:

    -best_node: The best right subtree node
        """
        global global_node_counter
        

        if "[answer_subquestion1]" in original_question:
            question = original_question.replace("[answer_subquestion1]", left_answer)
        else:
            question = original_question
        
        logger.debug(f"\n{'='*80}")
        logger.debug(f"Construct the right subtree based on the problem: '{question}'")
        logger.debug(f"{'='*80}")
        

        max_attempts = num_variants
        current_attempt = 0
        current_question = question
        attempted_questions = [question]  
        
        while current_attempt < max_attempts:
            logger.debug(f"\nTry #{current_attempt+1} use the question: '{current_question}'")
            
            all_trees = []
            trees_to_generate = trees_per_variant  
            
            logger.debug(f"\nGenerate a {trees_to_generate} tree (maximum height: {max_height}) for the current problem.")
            
            for j in range(trees_to_generate):
            
                tree_temp = temperature
                
                logger.debug(f"\n #{j+1} (temperature={tree_temp}, max_height={max_height}):")
            
                root, cost = self.build_question_tree(
                    current_question, cost, max_tokens, tree_temp, top_p, frequency_penalty,
                    presence_penalty, examples_db, num_examples, depth=0,
                    placeholder_answers={}, max_height=max_height, logger=logger
                )
                
                height, node_count = get_tree_statistics(root)
                
                all_trees.append({
                    'tree': root,
                    'tree_num': j+1,
                    'height': height,
                    'node_count': node_count,
                    'question_text': current_question
                })
                
            tree_shape_counter = Counter()
            for tree_info in all_trees:
                shape = (tree_info['height'], tree_info['node_count'])
                tree_shape_counter[shape] += 1
            
            logger.debug("\nTree shape frequency (height, number of nodes)")
            for shape, count in tree_shape_counter.most_common():
                logger.debug(f"height: {shape [0]}, the number of nodes: {shape [1]} - frequency: {count}")
            
            if tree_shape_counter:
                most_common_shape, _ = tree_shape_counter.most_common(1)[0]
                logger.debug(f"\nMost common shapes: Height {most_common_shape[0]}, Number of nodes {most_common_shape[1]}")
                
                most_common_trees = [tree_info for tree_info in all_trees 
                                if (tree_info['height'], tree_info['node_count']) == most_common_shape]
                if most_common_trees:
                    
                    best_tree_info = most_common_trees[0]
                    
                    return best_tree_info['tree']
            
            current_attempt += 1
            if current_attempt < max_attempts:
                
                new_variants, cost, variant_cost = self.q_variant_generator.generate_question_variants(
                    question, cost, num_variants=1, logger=logger
                )
                self._log_cost_event(
                    "gen_q_var_cost",
                    variant_cost,
                    scope="enhanced_right_subtree",
                    attempt=current_attempt,
                    question=question,
                )
                
                if len(new_variants) > 1:
                    new_question = new_variants[1]
                    
                    current_question = new_question
                    attempted_questions.append(current_question)
                else:
                    logger.debug(f"Warning: Variant generation failed. Use the original question.")
                    break 
        
        leaf_node = QuestionNode(
            question=question, 
            q_type="None",
            subq1=question, 
            subq2=""
        )
        return leaf_node

    def build_enhanced_right_subtree(self, original_question, left_answer, cost, max_tokens=800, 
                                temperature=0.2, top_p=1.0, frequency_penalty=0.0, presence_penalty=0.0, 
                                examples_db=None, num_examples=3, max_height=3,
                                num_variants=2, trees_per_variant=3, logger=None):
        """
    Build multiple trees for the right subtree of Sequential type, following the same logic as the main function

    Parameters:

    -original_question: Original right subtree problem

    -left_answer: The answer of the left subtree, used to replace [answer_subquestion1]

    Other parameters are the same as build_question_tree


    Returns:

    -best_node: The best right subtree node
        """
        global global_node_counter
        

        if "[answer_subquestion1]" in original_question:
            question = original_question.replace("[answer_subquestion1]", left_answer)
        else:
            question = original_question
        
        logger.debug(f"\n{'='*80}")
        logger.debug(f"Construct the right subtree based on the problem: '{question}'")
        logger.debug(f"{'='*80}")
        

        max_attempts = num_variants
        current_attempt = 0
        current_question = question
        attempted_questions = [question]  
        
        while current_attempt < max_attempts:
            logger.debug(f"\nTry #{current_attempt+1} use the question: '{current_question}'")
            
            all_trees = []
            trees_to_generate = trees_per_variant  
            
            logger.debug(f"\nGenerate a {trees_to_generate} tree (maximum height: {max_height}) for the current problem.")
            
            for j in range(trees_to_generate):
            
                tree_temp = temperature
                
                logger.debug(f"\n #{j+1} (temperature={tree_temp}, max_height={max_height}):")
            
                root, cost = self.build_question_tree(
                    current_question, cost, max_tokens, tree_temp, top_p, frequency_penalty,
                    presence_penalty, examples_db, num_examples, depth=0,
                    placeholder_answers={}, max_height=max_height, logger=logger
                )
                
                height, node_count = get_tree_statistics(root)
                
                all_trees.append({
                    'tree': root,
                    'tree_num': j+1,
                    'height': height,
                    'node_count': node_count,
                    'question_text': current_question
                })
                
            tree_shape_counter = Counter()
            for tree_info in all_trees:
                shape = (tree_info['height'], tree_info['node_count'])
                tree_shape_counter[shape] += 1
            
            logger.debug("\nTree shape frequency (height, number of nodes)")
            for shape, count in tree_shape_counter.most_common():
                logger.debug(f"height: {shape [0]}, the number of nodes: {shape [1]} - frequency: {count}")
            
            if tree_shape_counter:
                most_common_shape, _ = tree_shape_counter.most_common(1)[0]
                logger.debug(f"\nMost common shapes: Height {most_common_shape[0]}, Number of nodes {most_common_shape[1]}")
                
                most_common_trees = [tree_info for tree_info in all_trees 
                                if (tree_info['height'], tree_info['node_count']) == most_common_shape]
                if most_common_trees:
                    
                    best_tree_info = most_common_trees[0]
                    
                    return best_tree_info['tree']
            
            current_attempt += 1
            if current_attempt < max_attempts:
                
                new_variants, cost, variant_cost = self.q_variant_generator.generate_question_variants(
                    question, cost, num_variants=1, logger=logger
                )
                self._log_cost_event(
                    "gen_q_var_cost",
                    variant_cost,
                    scope="enhanced_right_subtree",
                    attempt=current_attempt,
                    question=question,
                )
                
                if len(new_variants) > 1:
                    new_question = new_variants[1]
                    
                    current_question = new_question
                    attempted_questions.append(current_question)
                else:
                    logger.debug(f"Warning: Variant generation failed. Use the original question.")
                    break 
        
        leaf_node = QuestionNode(
            question=question, 
            q_type="None",
            subq1=question, 
            subq2=""
        )
        return leaf_node
    
    def solve_tree(self, root, original_question, cost, max_tokens=4000, 
              temperature=0, top_p=1.0, frequency_penalty=0.0, 
              presence_penalty=0.0, examples_db=None, num_examples=20,
              enhanced_right_subtree=True, right_subtree_variants=2, 
              right_subtree_trees_per_variant=2, max_height=3,
              placeholder_answers=None, logger=None):
        """
        Solve a single problem tree and return the answer, supporting the construction of an enhanced right subtree
        """
        global global_node_counter
        
        if placeholder_answers is None:
            placeholder_answers = {}  
        
        
        processed_node_ids = set()

        def solve_node(node, cost, updated_tree=False, current_depth=0, logger=None):
            if node is None:
                return {}, cost

            
            if node.id in processed_node_ids:
                return {node.id: placeholder_answers.get(node.id, "[none]")}, cost
            
            
            processed_node_ids.add(node.id)
            
            node_answers = {}
            
            if node.depends_on and node.depends_on not in placeholder_answers:
                
                def find_node_by_id(search_node, target_id):
                    if search_node is None:
                        return None,
                    if search_node.id == target_id:
                        return search_node
                    left_result = find_node_by_id(search_node.left, target_id)
                    if left_result:
                        return left_result
                    return find_node_by_id(search_node.right, target_id)
                
                dependent_node = find_node_by_id(root, node.depends_on)
                if dependent_node:
                    dependent_answers, cost = solve_node(dependent_node, cost, updated_tree, current_depth,logger)
                    node_answers.update(dependent_answers)
            
        
            if node.id in placeholder_answers:
                node.answer = placeholder_answers[node.id]
                node_answers[node.id] = node.answer
                
                return node_answers, cost

            if (node.left is None and node.right is None) or node.type == "None":
                actual_question = node.question
                node.meta["is_leaf"] = True
                node.meta["actual_question_used"] = actual_question
            
                if node.depends_on and node.depends_on in placeholder_answers:
                    dependent_answer = placeholder_answers[node.depends_on]
                    
                    if dependent_answer.lower() == "[none]":
                        node.answer = "[none]"
                        placeholder_answers[node.id] = "[none]"
                        
                        node_answers[node.id] = "[none]"
                        return node_answers, cost
                    
                    if "[answer_subquestion1]" in actual_question:
                        actual_question = actual_question.replace("[answer_subquestion1]", dependent_answer)
                    elif f"[answer from {node.depends_on}]" in actual_question:
                        actual_question = actual_question.replace(f"[answer from {node.depends_on}]", dependent_answer)
                    
                    node.display_question = actual_question

                full_response, cost, iterative_log = self.iter_ans_generator.answer_question(
                    question=actual_question,
                    cost=cost,
                    logger=logger,
                    max_iterations=MAX_ITERATIONS
                )
                answer = extract_answer(full_response)
                node.answer = answer
                placeholder_answers[node.id] = answer
                node_answers[node.id] = answer

                # Attach iterative retrieval/generation log to node meta
                try:
                    node.meta["iterative"] = iterative_log
                    # Flatten all retrieved documents across iterations for convenience
                    all_retrieved = []
                    for it in iterative_log.values():
                        all_retrieved.extend(it.get("retrieved_documents", []))
                    node.meta["retrieved"] = all_retrieved
                except Exception:
                    pass
                for iteration_id, iteration_data in iterative_log.items():
                    if not str(iteration_id).isdigit():
                        continue
                    refine_cost = iteration_data.get("refine_query_cost")
                    if refine_cost:
                        self._log_cost_event(
                            "refine_cost",
                            refine_cost,
                            node_id=node.id,
                            iteration=iteration_id,
                            question=actual_question,
                        )
                    for response_info in iteration_data.get("responses", []):
                        gen_cost = response_info.get("generation_cost")
                        if gen_cost:
                            self._log_cost_event(
                                "answer_cost",
                                gen_cost,
                                node_id=node.id,
                                iteration=iteration_id,
                                question=actual_question,
                            )

                if (("[answer_subquestion1]" in node.question or "[answer from" in node.question) and 
                    not updated_tree and node.answer and node.answer.lower() != "[none]"):
                    
                    
                    if node.parent and not node.is_left_child:
                        logger.debug(f"node {node.id}: Contains the replacement tag and has obtained the answer. It needs to be rebuilt")
                        
                
                return node_answers, cost      

            left_answers, cost = solve_node(node.left, cost, updated_tree, current_depth + 1, logger)
            node_answers.update(left_answers)

            needs_reconstruction = False
            if node.right and node.type == "Sequential":
                
                if ("[answer_subquestion1]" in node.right.question or 
                    (node.right.depends_on and f"[answer from {node.right.depends_on}]" in node.right.question)):
                    
                
                    if node.left and node.left.id in placeholder_answers:
                        left_answer = placeholder_answers[node.left.id]
                    
                        if left_answer.lower() != "[none]":
                            needs_reconstruction = True
            
        
            if needs_reconstruction and not updated_tree and enhanced_right_subtree:     
                new_right_question, cost, right_q_cost = self.iter_ans_generator.generate_right_question_with_llm(
                    parent_question=node.question,
                    left_question=node.left.question if node.left else "",
                    left_answer=placeholder_answers[node.left.id],
                    original_right_question=node.right.question,
                    cost = cost,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    top_p=top_p,
                )
                self._log_cost_event(
                    "gen_right_q_cost",
                    right_q_cost,
                    node_id=node.id,
                    depth=current_depth,
                    question=node.question,
                )

                remaining_height = max(max_height - current_depth - 1, 1)  

                new_right_node = self.build_enhanced_right_subtree(
                    original_question=new_right_question,
                    left_answer=placeholder_answers[node.left.id],
                    cost = cost,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    frequency_penalty=frequency_penalty,
                    presence_penalty=presence_penalty,
                    examples_db=examples_db,
                    num_examples=num_examples,
                    max_height=remaining_height,  
                    num_variants=right_subtree_variants,
                    trees_per_variant=right_subtree_trees_per_variant,
                    logger=logger
                )  

                node.right = new_right_node
                node.meta["reconstructed_right_subtree"] = True

                right_answers, cost = solve_node(node.right, cost, True, current_depth + 1, logger)
                node_answers.update(right_answers)
            elif needs_reconstruction and not updated_tree:

                new_right_question, cost, right_q_cost = self.iter_ans_generator.generate_right_question_with_llm(
                    parent_question=node.question,
                    left_question=node.left.question if node.left else "",
                    left_answer=placeholder_answers[node.left.id],
                    original_right_question=node.right.question,
                    cost=cost,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    top_p=top_p,
                )       
                self._log_cost_event(
                    "gen_right_q_cost",
                    right_q_cost,
                    node_id=node.id,
                    depth=current_depth,
                    question=node.question,
                )

                remaining_height = max(max_height - current_depth - 1, 1) 

                new_right_node, cost = self.build_question_tree(
                    new_right_question, cost, temperature, examples_db, 
                    depth=current_depth + 1, parent=node, is_left_child=False, 
                    max_height=remaining_height, placeholder_answers=placeholder_answers, logger=logger
                )

                node.right = new_right_node
                node.meta["reconstructed_right_subtree"] = True

                right_answers, cost = solve_node(node.right, cost, True, current_depth + 1, logger)
                node_answers.update(right_answers)
            else:

                if (node.right and ("[answer_subquestion1]" in node.right.question or 
                    (node.right.depends_on and f"[answer from {node.right.depends_on}]" in node.right.question))):
                    
                
                    if node.left and node.left.id in placeholder_answers:
                        new_question, cost, right_q_cost = self.iter_ans_generator.generate_right_question_with_llm(
                            parent_question=node.question,
                            left_question=node.left.question,
                            left_answer=placeholder_answers[node.left.id],
                            original_right_question=node.right.question,
                            cost=cost
                        )
                        self._log_cost_event(
                            "gen_right_q_cost",
                            right_q_cost,
                            node_id=node.id,
                            depth=current_depth,
                            question=node.question,
                        )
                        node.right.question = new_question
                        node.right.display_question = new_question
                
            
                right_answers, cost = solve_node(node.right, cost, updated_tree, current_depth + 1, logger)
                node_answers.update(right_answers)

            child_questions = []
            valid_child_answers = False
        
            if node.left and node.left.id in node_answers:
                left_answer = node_answers[node.left.id]
                left_question = node.left.display_question
                child_questions.append((left_question, left_answer))
                
                if left_answer.lower() != "[none]":
                    valid_child_answers = True
        
            if node.right and node.right.id in node_answers:
                right_answer = node_answers[node.right.id]
                right_question = node.right.display_question
                child_questions.append((right_question, right_answer))
            
                if right_answer.lower() != "[none]":
                    valid_child_answers = True
            
            if node.type == "Sequential" and node.left and node.left.id in node_answers:
                if node_answers[node.left.id].lower() == "[none]":
                    valid_child_answers = False

            if valid_child_answers and child_questions:
                node.meta["aggregation"] = True
                node.meta["child_questions"] = [(q, a) for q, a in child_questions]
                final_answer, cost, final_gen_cost = self.generator.final_generate(node.display_question, child_questions, cost)
                node.meta["final_ans_gen_cost"] = final_gen_cost
                self._log_cost_event(
                    "final_ans_gen_cost",
                    final_gen_cost,
                    node_id=node.id,
                    depth=current_depth,
                    question=node.display_question,
                )
                extracted_answer = re.search(r'final answer is:\s*(.*)', final_answer, re.DOTALL)
                if not extracted_answer:
                    extracted_answer = re.search(r'answer is:\s*(.*)', final_answer, re.DOTALL)
                node.answer = extracted_answer.group(1).strip() if extracted_answer else final_answer
                placeholder_answers[node.id] = node.answer
                node_answers[node.id] = node.answer

                if node.answer.lower() == "[none]":
                    full_response, cost, iterative_log = self.iter_ans_generator.answer_question(
                        question=node.display_question,
                        cost=cost,
                        logger=logger,
                        max_iterations=MAX_ITERATIONS
                    )
                    answer = extract_answer(full_response)
                    node.answer = answer
                    placeholder_answers[node.id] = answer
                    node_answers[node.id] = answer
                    try:
                        node.meta["iterative"] = iterative_log
                        all_retrieved = []
                        for it in iterative_log.values():
                            all_retrieved.extend(it.get("retrieved_documents", []))
                        node.meta["retrieved"] = all_retrieved
                    except Exception:
                        pass
                    for iteration_id, iteration_data in iterative_log.items():
                        if not str(iteration_id).isdigit():
                            continue
                        refine_cost = iteration_data.get("refine_query_cost")
                        if refine_cost:
                            self._log_cost_event(
                                "refine_cost",
                                refine_cost,
                                node_id=node.id,
                                iteration=iteration_id,
                                question=node.display_question,
                            )
                        for response_info in iteration_data.get("responses", []):
                            gen_cost = response_info.get("generation_cost")
                            if gen_cost:
                                self._log_cost_event(
                                    "answer_cost",
                                    gen_cost,
                                    node_id=node.id,
                                    iteration=iteration_id,
                                    question=node.display_question,
                                )
                
            else:
                node.meta["fallback_direct_answer"] = True
                full_response, cost, iterative_log = self.iter_ans_generator.answer_question(
                    question=node.display_question,
                    cost=cost,
                    logger=logger,
                    max_iterations=MAX_ITERATIONS
                )
                answer = extract_answer(full_response)
                node.answer = answer
                placeholder_answers[node.id] = answer
                node_answers[node.id] = answer

                try:
                    node.meta["iterative"] = iterative_log
                    all_retrieved = []
                    for it in iterative_log.values():
                        all_retrieved.extend(it.get("retrieved_documents", []))
                    node.meta["retrieved"] = all_retrieved
                except Exception:
                    pass
                for iteration_id, iteration_data in iterative_log.items():
                    if not str(iteration_id).isdigit():
                        continue
                    refine_cost = iteration_data.get("refine_query_cost")
                    if refine_cost:
                        self._log_cost_event(
                            "refine_cost",
                            refine_cost,
                            node_id=node.id,
                            iteration=iteration_id,
                            question=node.display_question,
                        )
                    for response_info in iteration_data.get("responses", []):
                        gen_cost = response_info.get("generation_cost")
                        if gen_cost:
                            self._log_cost_event(
                                "answer_cost",
                                gen_cost,
                                node_id=node.id,
                                iteration=iteration_id,
                                question=node.display_question,
                            )

            return node_answers, cost
        
    
        all_answers, cost = solve_node(root, cost, False, 0, logger)
        
        final_result = root.answer if root.answer else "[none]"
        
        return final_result, cost

    #---------------------------------------------------- Multi-Tree Multi-Variant Solution -----------------------------------------------
    def decompose_and_answer_with_variants(self, question, cost, trees_per_question=TREES_PER_QUESTION, max_tokens=MAX_TOKENS, 
                                     temperature=DECOMPOSE_TEMPERATURE, top_p=TOP_P, frequency_penalty=FREQUENCY_PENALTY, presence_penalty=PRESENCE_PENALTY, 
                                     num_examples=NUM_EXAMPLES, max_height=MAX_HEIGHT, enhanced_right_subtree=ENHANCED_RIGHT_SUBTREE,
                                     right_subtree_variants=RIGHT_SUBTREE_VARIANTS, right_subtree_trees_per_variant=RIGHT_SUBTREE_TREES_PER_VARIANT,
                                     max_variants=MAX_VARIANTS, stats_file_path=STATS_FILE_PATH, logger=None):

        """
        Generate 6 trees per question, categorize by shape, and solve only one tree from the most common type.
        If unsuccessful, generate new question variants and try again.
        """

        global global_node_counter
        
        # Reset global counter to ensure each call starts from 0
        global_node_counter = 0
        self.cost_trace = []
        self.latest_tree_bundle = {
            "selected_tree": None,
            "candidate_trees": [],
        }

        examples_db = get_examples_database()

        # Set trees_per_question to 6 as requested
        trees_per_question = trees_per_question

        # Track attempt with original question and variants
        attempt_count = 1
        current_question = question
        attempted_questions = [question]  # Keep track of questions we've tried
        
        # Variables to track tree heights
        initial_height = 0
        final_height = 0
        success = False

        while attempt_count <= MAX_VARIANTS:

            logger.debug(f"\n{'='*80}")
            logger.debug(f"ATTEMPT {attempt_count} with question: '{current_question}'")
            logger.debug(f"{'='*80}")

            all_trees = []

            # Generate trees for the current question
            logger.debug(f"\nGenerating {trees_per_question} trees for current question (max height: {max_height})")

            for j in range(trees_per_question):
                # Use different temperature for tree diversity
                tree_temp = 0
                
                logger.debug(f"\nBuilding tree {j+1} for current question (temperature={tree_temp}, max_height={max_height}):")
                
                # Build the tree with height limitation
                root, cost = self.build_question_tree(
                    current_question, cost, max_tokens, tree_temp, top_p, frequency_penalty,
                    presence_penalty, examples_db, num_examples, depth=0,
                    placeholder_answers={}, max_height=max_height, logger=logger
                )
                # Get tree statistics
                height, node_count = get_tree_statistics(root)      

                # Save tree information
                all_trees.append({
                    'tree': root,
                    'tree_num': j+1,
                    'height': height,
                    'node_count': node_count,
                    'question_text': current_question
                })
                self.latest_tree_bundle["candidate_trees"].append({
                    "attempt": attempt_count,
                    "tree_num": j + 1,
                    "height": height,
                    "node_count": node_count,
                    "question_text": current_question,
                    "tree": root.to_dict(),
                })

                logger.debug(f"Tree {j+1} - Height: {height}, Node count: {node_count}")

            # Calculate tree shape frequencies
            tree_shape_counter = Counter()
            for tree_info in all_trees:
                shape = (tree_info['height'], tree_info['node_count'])
                tree_shape_counter[shape] += 1

            # Print shape frequencies
            logger.debug("\nTree shape frequencies (height, node count):")
            for shape, count in tree_shape_counter.most_common():
                logger.debug(f"Height: {shape[0]}, Node count: {shape[1]} - Frequency: {count}")
            
            # Get the most common shape
            if tree_shape_counter:
                most_common_shape, most_common_count = tree_shape_counter.most_common(1)[0]
                logger.debug(f"\nMost common shape: Height {most_common_shape[0]}, Node count {most_common_shape[1]} (Count: {most_common_count})")
                
                # Filter trees to only include the most common shape
                most_common_trees = [tree_info for tree_info in all_trees 
                                if (tree_info['height'], tree_info['node_count']) == most_common_shape]
                
                logger.debug(f"Found {len(most_common_trees)} trees with the most common shape")

                # Only solve the first tree from the most common shape
                if most_common_trees:
                    tree_info = most_common_trees[0]
                    tree_root = tree_info['tree']
                    question_text = tree_info['question_text']
                    self.latest_tree_bundle["selected_tree"] = {
                        "attempt": attempt_count,
                        "tree_num": tree_info["tree_num"],
                        "height": tree_info["height"],
                        "node_count": tree_info["node_count"],
                        "question_text": question_text,
                        "selection_rule": "first_tree_from_most_common_shape",
                        "tree": tree_root.to_dict(),
                    }

                    # Save initial height
                    initial_height = tree_info['height']

                    
                    logger.debug(f"\n{'-'*80}")
                    logger.debug(f"Attempting to solve one tree from most common shape: Tree {tree_info['tree_num']}")
                    logger.debug(f"Question: '{question_text}'")
                    logger.debug(f"Tree height: {tree_info['height']}, Node count: {tree_info['node_count']}")
                    logger.debug(f"{'-'*80}")
                    
                    logger.debug("\nTree structure:")
                    print_all_nodes(tree_root, logger=logger)

                    # Create a new placeholder_answers dictionary for each tree to avoid confusion
                    placeholder_answers = {}

                    answer, cost = self.solve_tree(
                        tree_root, question_text, cost, max_tokens, temperature, top_p,
                        frequency_penalty, presence_penalty, examples_db, num_examples,
                        enhanced_right_subtree=enhanced_right_subtree,
                        right_subtree_variants=right_subtree_variants,
                        right_subtree_trees_per_variant=right_subtree_trees_per_variant,
                        max_height=max_height,
                        placeholder_answers=placeholder_answers, logger=logger
                    )

                    # Calculate the final height after solving (which might have expanded the tree)
                    final_height, _ = get_tree_statistics(tree_root)
                    
                    logger.debug(f"\nSelected tree returned answer: '{answer}'")
                    logger.debug(f"Initial tree height: {initial_height}, Final tree height after solving: {final_height}")

                    
                    # If we found a valid (non-[none]) answer, use it and stop
                    if answer.lower() != "[none]":
                        logger.debug(f"Found valid answer, stopping tree traversal")
                        logger.debug("\n" + "="*80)
                        logger.debug(f"Final answer for question: '{question}'")
                        logger.debug(f"Answer: '{answer}'")
                        logger.debug("="*80)
                        
                        # Save tree statistics
                        success = True
                        save_tree_stats(question, answer, initial_height, final_height, stats_file_path, success, logger)
                        if self.latest_tree_bundle.get("selected_tree") is not None:
                            self.latest_tree_bundle["selected_tree"]["tree"] = tree_root.to_dict()
                        
                        self.latest_cost_trace = list(self.cost_trace)
                        return answer, cost, tree_root
                    else:
                        logger.debug(f"Tree returned [none], will try with a new question variant")

            # Generate question variants 
            attempt_count += 1
            if attempt_count <= MAX_VARIANTS:
                logger.debug(f"\n{'-'*80}")
                logger.debug(f"No valid answer found from selected tree. Generating new question variant {attempt_count}")
                logger.debug(f"{'-'*80}")

                # Use the existing generate_question_variants function
                new_variants, cost, variant_cost = self.q_variant_generator.generate_question_variants(
                    question, cost, num_variants=1, logger=logger
                )
                self._log_cost_event(
                    "gen_q_var_cost",
                    variant_cost,
                    scope="decompose_and_answer_with_variants",
                    attempt=attempt_count,
                    question=question,
                )

                # Get the new variant (skipping the first one which is the original question)
                if len(new_variants) > 1:
                    new_question = new_variants[1]
                else:
                    logger.debug(f"Warning: generate_question_variants failed to produce a variant, falling back to original question")
                    new_question = question           

            else:
                # If we've exhausted all variants, use direct lookup on the original question
                logger.debug(f"\n{'-'*80}")
                logger.debug("Exhausted all variants. Using direct lookup on original question.")
                logger.debug(f"{'-'*80}")


                # Directly call answer_question on the original question
                final_answer, cost, direct_answer_cost, r_cost_individual = self.iter_ans_generator.direct_answer(question, cost, logger)
                self._log_cost_event(
                    "answer_reason_cost",
                    direct_answer_cost,
                    scope="decompose_and_answer_with_variants",
                    question=question,
                )
                self._log_cost_event(
                    "r_cost",
                    r_cost_individual,
                    scope="decompose_and_answer_with_variants",
                    question=question,
                )

                # Save tree statistics with direct lookup
                success = False
                save_tree_stats(question, final_answer, initial_height, final_height, stats_file_path, success, logger)
                
                self.latest_cost_trace = list(self.cost_trace)
                return final_answer, cost, None
            
        # If we somehow get here (should not happen with the logic above)
        # Save tree statistics as a failure
        success = False
        save_tree_stats(question, "Could not determine an answer", initial_height, final_height, stats_file_path, success, logger)
        
        self.latest_cost_trace = list(self.cost_trace)
        return "Could not determine an answer after trying original question and variants.", cost, None
