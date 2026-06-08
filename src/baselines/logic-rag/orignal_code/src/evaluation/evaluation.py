import logging
import pdb
import torch
import time
from typing import Dict, List, Tuple, Any
import json
import os
from tqdm import tqdm

from src.utils.utils import (
    normalize_answer, 
    evaluate_with_llm, 
    string_based_evaluation,
    save_results,
    TOKEN_COST
)
from src.models.logic_rag import LogicRAG
from config.config import RESULT_DIR

# Configure logging
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

# Dictionary of available RAG models
RAG_MODELS = {
    "logic-rag": LogicRAG,
}

# Directory for checkpoints
CHECKPOINT_DIR = os.path.join(RESULT_DIR, "checkpoints")
DEFAULT_CHECKPOINT_INTERVAL = 5  # Default: save checkpoint every 5 questions

class RAGEvaluator:
    """Evaluator for RAG models."""
    
    def __init__(self, model_name: str, corpus_path: str, max_rounds: int = 3, top_k: int = 5, 
                eval_top_ks: List[int] = [5, 10], checkpoint_interval: int = DEFAULT_CHECKPOINT_INTERVAL):
        """Initialize the evaluator with corpus path and parameters.
        
        Args:
            model_name: Name of the RAG model to evaluate
            corpus_path: Path to the corpus file
            max_rounds: Maximum number of rounds for agentic RAG
            top_k: Number of contexts to retrieve
            eval_top_ks: List of k values for top-k accuracy evaluation
            checkpoint_interval: Number of questions to process before saving checkpoint
        """
        self.model_name = model_name
        self.corpus_path = corpus_path
        self.max_rounds = max_rounds
        self.top_k = top_k
        self.eval_top_ks = sorted(eval_top_ks)  # Sort to ensure consistent processing
        self.checkpoint_interval = checkpoint_interval
        
        # Create result directory if it doesn't exist
        os.makedirs(RESULT_DIR, exist_ok=True)
        
        # Create checkpoint directory if it doesn't exist
        os.makedirs(CHECKPOINT_DIR, exist_ok=True)
        
        # Initialize the RAG model
        self._initialize_model()
    
    def _initialize_model(self):
        """Initialize the specified RAG model."""
        if self.model_name not in RAG_MODELS:
            raise ValueError(f"Unknown RAG model: {self.model_name}")
        
        # Create model instance
        model_class = RAG_MODELS[self.model_name]
        self.model = model_class(self.corpus_path)
        
        # Configure model
        self.model.set_top_k(self.top_k)
        
        # Set max rounds for agentic models
        if hasattr(self.model, 'set_max_rounds'):
            self.model.set_max_rounds(self.max_rounds)
        
        logger.info(f"Initialized {self.model_name} model")
    
    def evaluate_question(self, question: str, gold_answer: str) -> Dict:
        """Evaluate the model on a single question."""
        # Run the model on the question
        start_time = time.time()
        
        answer, contexts, rounds = self.model.answer_question(question)
        elapsed_time = time.time() - start_time
        
        # Evaluate answer with LLM
        is_correct = evaluate_with_llm(answer, gold_answer)
        
        result = {
            "question": question,
            "gold_answer": gold_answer,
            "answer": answer,
            "contexts": contexts,
            "time": elapsed_time,
            "rounds": rounds,
            "is_correct": is_correct
        }
        
        # Add dependency analysis for interpretable models
        if hasattr(self.model, 'last_dependency_analysis'):
            result["dependency_analysis"] = self.model.last_dependency_analysis
        
        return result
        
    def calculate_retrieval_metrics(self, retrieved_contexts: List[List[str]], answers: List[str]) -> Dict[str, float]:
        """Calculate retrieval-based metrics."""
        total = len(answers)
        found_in_context = 0
        
        # Initialize answer_in_top_k counters for each k in eval_top_ks
        answer_in_top_k = {k: 0 for k in self.eval_top_ks}
        
        for contexts, answer in zip(retrieved_contexts, answers):
            normalized_answer = normalize_answer(answer)
            
            # Check if answer is in any context
            for i, context in enumerate(contexts):
                if normalized_answer in normalize_answer(context):
                    found_in_context += 1
                    # Update counters for each k value
                    for k in self.eval_top_ks:
                        if i < k:
                            answer_in_top_k[k] += 1
                    break
        
        # Prepare result dictionary
        result = {
            "answer_found_in_context": found_in_context / total,
            "total_questions": total
        }
        
        # Add top-k metrics to result
        for k in self.eval_top_ks:
            result[f"answer_in_top{k}"] = answer_in_top_k[k] / total
            
        return result
    
    def _get_checkpoint_path(self, output_file: str) -> str:
        """Generate a checkpoint path based on the output file."""
        base_name = os.path.basename(output_file)
        name, ext = os.path.splitext(base_name)
        return os.path.join(CHECKPOINT_DIR, f"{name}_checkpoint{ext}")
    
    def _save_checkpoint(self, results: List[Dict], metrics: Dict, processed_count: int, output_file: str):
        """Save a checkpoint of current evaluation progress."""

        # If the last "answer" is empty in the results, it indicates that we have lost connection to the LLM API
        # We should not save the checkpoint in this case, and we should terminate the whole pipeline
        if results[-1]["answer"] == "":
            print("\n\n\033[91mLost connection to the LLM API, skipping checkpoint save\033[0m\n\n")
            exit(1)  # Use exit code 1 to indicate error condition

        checkpoint = {
            "model": self.model_name,
            "metrics": metrics,
            "results": results,
            "processed_count": processed_count,
            "token_cost": {
                "prompt": TOKEN_COST["prompt"],
                "completion": TOKEN_COST["completion"]
            }
        }
        
        checkpoint_path = self._get_checkpoint_path(output_file)
        
        with open(checkpoint_path, 'w', encoding='utf-8') as f:
            json.dump(checkpoint, f, ensure_ascii=False, indent=2)
        logger.info(f"Checkpoint saved: {processed_count} questions processed")
    
    def _load_checkpoint(self, output_file: str) -> Tuple[List[Dict], Dict, int]:
        """Load evaluation checkpoint if it exists."""
        checkpoint_path = self._get_checkpoint_path(output_file)
        
        if not os.path.exists(checkpoint_path):
            return [], {}, 0
        
        try:
            with open(checkpoint_path, 'r', encoding='utf-8') as f:
                checkpoint = json.load(f)
            
            # Restore token costs
            if "token_cost" in checkpoint:
                TOKEN_COST["prompt"] = checkpoint["token_cost"]["prompt"]
                TOKEN_COST["completion"] = checkpoint["token_cost"]["completion"]
                logger.info(f"Restored token costs - Prompt: {TOKEN_COST['prompt']}, Completion: {TOKEN_COST['completion']}")
            
            logger.info(f"Loaded checkpoint: {checkpoint['processed_count']} questions already processed")
            return checkpoint["results"], checkpoint["metrics"], checkpoint["processed_count"]
        except Exception as e:
            logger.error(f"Error loading checkpoint: {e}")
            return [], {}, 0
    
    def run_single_model_evaluation(self, eval_data: List[Dict], output_file: str = "evaluation_results.json"):
        """Run evaluation of a single model on the given evaluation data."""
        # Try to load checkpoint
        results, metrics, processed_count = self._load_checkpoint(output_file)
        
        # Skip already processed questions
        if processed_count > 0:
            eval_data = eval_data[processed_count:]
            logger.info(f"Resuming from checkpoint: {processed_count} questions already processed, {len(eval_data)} remaining")
        
        # If all questions were already processed
        if not eval_data:
            logger.info("All questions were already processed in previous run.")
            # Load full results from final output
            output_path = os.path.join(RESULT_DIR, output_file)
            if os.path.exists(output_path):
                with open(output_path, 'r', encoding='utf-8') as f:
                    evaluation_summary = json.load(f)
                return evaluation_summary
        
        # If starting fresh, initialize metrics and reset token costs
        if not metrics:
            # Reset token costs for the current model evaluation
            TOKEN_COST["prompt"] = 0
            TOKEN_COST["completion"] = 0
            
            # Initialize metrics dictionary with dynamic top-k keys
            metrics = {
                "total_time": 0,
                "answer_coverage": 0,
                "answer_accuracy": 0,
                "string_accuracy": 0,
                "string_precision": 0,
                "string_recall": 0
            }
            
            # Add top-k hits for each k in eval_top_ks
            for k in self.eval_top_ks:
                metrics[f"top{k}_hits"] = 0
            
            # Add rounds tracking
            metrics["total_rounds"] = 0
        else:
            logger.info(f"Restored token costs - Prompt: {TOKEN_COST['prompt']}, Completion: {TOKEN_COST['completion']}")
        
        # Evaluation metrics
        total_questions = len(eval_data) + processed_count
        
        for i, item in enumerate(tqdm(eval_data, desc=f"Evaluating {self.model_name}")):
            question = item['question']
            gold_answer = item['answer']
            
            # Evaluate the model on this question
            result = self.evaluate_question(
                question=question,
                gold_answer=gold_answer
            )
            results.append(result)
            
            # Update metrics
            metrics["total_time"] += result["time"]
            normalized_gold = normalize_answer(gold_answer)
            
            # String-based evaluation
            string_metrics = string_based_evaluation(
                result["answer"], 
                gold_answer
            )
            metrics["string_accuracy"] += string_metrics["accuracy"]
            metrics["string_precision"] += string_metrics["precision"]
            metrics["string_recall"] += string_metrics["recall"]
            
            # Check retrieval coverage
            for j, ctx in enumerate(result["contexts"]):
                if normalized_gold in normalize_answer(ctx):
                    metrics["answer_coverage"] += 1
                    # Update counters for each k value
                    for k in self.eval_top_ks:
                        if j < k:
                            metrics[f"top{k}_hits"] += 1
                    break
            
            # Update rounds
            if "rounds" in result:
                metrics["total_rounds"] += result["rounds"]
            
            # Evaluate answer using LLM
            if result["is_correct"]:
                metrics["answer_accuracy"] += 1
            
            # Save checkpoint at regular intervals
            current_count = processed_count + i + 1
            if (current_count % self.checkpoint_interval == 0) or (i == len(eval_data) - 1):
                self._save_checkpoint(results, metrics, current_count, output_file)
        
        # Calculate average metrics
        avg_metrics = {
            "avg_time": metrics["total_time"] / total_questions,
            "answer_coverage": metrics["answer_coverage"] / total_questions * 100,
            "answer_accuracy": metrics["answer_accuracy"] / total_questions * 100,
            "string_accuracy": metrics["string_accuracy"] / total_questions * 100,
            "string_precision": metrics["string_precision"] / total_questions * 100,
            "string_recall": metrics["string_recall"] / total_questions * 100
        }
        
        # Add top-k coverage (renamed from accuracy) for each k in eval_top_ks
        for k in self.eval_top_ks:
            avg_metrics[f"top{k}_coverage"] = metrics[f"top{k}_hits"] / total_questions * 100
        
        # Add average rounds
        avg_metrics["avg_rounds"] = metrics["total_rounds"] / total_questions
        
        # Organize metrics by category
        organized_metrics = {
            "performance": {
                "avg_time": avg_metrics["avg_time"]
            },
            "string_based": {
                "accuracy": avg_metrics["string_accuracy"],
                "precision": avg_metrics["string_precision"],
                "recall": avg_metrics["string_recall"]
            },
            "llm_evaluated": {
                "answer_accuracy": avg_metrics["answer_accuracy"]
            },
            "retrieval": {
                "answer_coverage": avg_metrics["answer_coverage"]
            }
        }
        
        # Add rounds
        organized_metrics["performance"]["avg_rounds"] = avg_metrics["avg_rounds"]
        
        # Add token cost metrics
        if total_questions > 0:
            organized_metrics["performance"]["avg_prompt_tokens"] = TOKEN_COST["prompt"] / total_questions
            organized_metrics["performance"]["avg_completion_tokens"] = TOKEN_COST["completion"] / total_questions
            organized_metrics["performance"]["avg_total_tokens"] = (TOKEN_COST["prompt"] + TOKEN_COST["completion"]) / total_questions
        else:
            organized_metrics["performance"]["avg_prompt_tokens"] = 0
            organized_metrics["performance"]["avg_completion_tokens"] = 0
            organized_metrics["performance"]["avg_total_tokens"] = 0
        
        # Add top-k coverage metrics
        for k in self.eval_top_ks:
            organized_metrics["retrieval"][f"top{k}_coverage"] = avg_metrics[f"top{k}_coverage"]
        
        # Add raw metrics for backwards compatibility
        organized_metrics["raw"] = metrics
        
        # Prepare final evaluation summary
        evaluation_summary = {
            "model": self.model_name,
            "metrics": organized_metrics,
            "results": results
        }
        
        # Save results
        save_results(
            results=evaluation_summary,
            output_file=output_file,
            results_dir=RESULT_DIR
        )
        
        # Log results in three sections
        logger.info(f"\nEvaluation Summary for {self.model_name}:")
        
        # Performance metrics
        logger.info(f"Average time per question: {avg_metrics['avg_time']:.2f} seconds")
        logger.info(f"Average rounds per question: {avg_metrics['avg_rounds']:.2f}")
        
        # Log token costs
        logger.info(f"Average prompt tokens per question: {organized_metrics['performance']['avg_prompt_tokens']:.2f}")
        logger.info(f"Average completion tokens per question: {organized_metrics['performance']['avg_completion_tokens']:.2f}")
        logger.info(f"Average total tokens per question: {organized_metrics['performance']['avg_total_tokens']:.2f}")
        
        # 1. String-based metrics
        logger.info("\n1. String-based Metrics:")
        logger.info(f"  • Accuracy: {avg_metrics['string_accuracy']:.2f}%")
        logger.info(f"  • Precision: {avg_metrics['string_precision']:.2f}%")
        logger.info(f"  • Recall: {avg_metrics['string_recall']:.2f}%")
        
        # 2. LLM evaluated metrics
        logger.info("\n2. LLM Evaluated Metrics:")
        logger.info(f"  • Answer Accuracy: {avg_metrics['answer_accuracy']:.2f}%")
        
        # 3. Retrieval performance
        logger.info("\n3. Retrieval Performance:")
        logger.info(f"  • Answer Coverage: {avg_metrics['answer_coverage']:.2f}%")
        
        # Log top-k coverage metrics
        for k in self.eval_top_ks:
            logger.info(f"  • Top-{k} Coverage: {avg_metrics[f'top{k}_coverage']:.2f}%")
        
        return evaluation_summary