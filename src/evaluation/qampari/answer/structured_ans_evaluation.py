import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
from tqdm import tqdm

from basic_ans_evaluation import (
    _build_evaluated_query_set,
    _load_existing_results,
    create_inverse_mappings,
    create_mapping,
    exact_match_score,
    parse_generation_to_predictions,
)


def evaluate_traces(input_path: str, save_path: str = None) -> Dict[str, Any]:
    metrics = {'precision': list(), 'recall': list(), 'f1': list()}
    results: List[Dict[str, Any]] = []
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        results = _load_existing_results(save_path)
        for item in results:
            metrics['precision'].append(item["precision"])
            metrics['recall'].append(item["recall"])
            metrics['f1'].append(item["f1"])

    trace_dir = Path(input_path)
    trace_files = sorted(
        trace_dir.glob("ex_*.json"),
        key=lambda file_path: int(file_path.stem.split("_")[1]),
    )
    evaluated_queries = _build_evaluated_query_set(results)

    for file_path in tqdm(trace_files, desc="Evaluating traces"):
        with file_path.open("r", encoding="utf-8") as file:
            trace_data = json.load(file)

        if "generation" not in trace_data:
            raise KeyError(f"Prediction instance must contain 'generation': {file_path}")
        if "validated_answer" not in trace_data:
            raise KeyError(
                f"Prediction instance must contain 'validated_answer': {file_path}"
            )

        identity = trace_data["query"].strip()
        if identity in evaluated_queries:
            continue

        prediction = {
            "answer_list": trace_data["validated_answer"],
            "predictions": parse_generation_to_predictions(trace_data["generation"]),
            "ex_num": trace_data.get("ex_num"),
            "query": trace_data.get("query", ""),
            "generation": trace_data["generation"],
        }

        already_predicted = list()
        mapping = create_mapping(prediction)
        inversed_mapping = create_inverse_mappings(mapping)
        predicted_answers = set(prediction["predictions"])
        all_answers = len(prediction["answer_list"])

        for predicted in predicted_answers:
            for alias in inversed_mapping:
                if exact_match_score(predicted, alias):
                    if inversed_mapping[alias] not in already_predicted:
                        already_predicted.append(inversed_mapping[alias])
                    break

        curr_prec = (len(already_predicted) / len(predicted_answers)) if len(predicted_answers) > 0 else 0
        curr_rec = (len(already_predicted) / all_answers) if all_answers > 0 else 0
        if curr_rec == 0 or curr_prec == 0:
            curr_f1 = 0
        else:
            curr_f1 = ((2 * curr_rec * curr_prec) / (curr_rec + curr_prec))

        metrics['precision'].append(curr_prec)
        metrics['recall'].append(curr_rec)
        metrics['f1'].append(curr_f1)

        results.append(
            {
                "ex_num": prediction.get("ex_num"),
                "query": prediction.get("query", ""),
                "output": prediction.get("generation", ""),
                "parsed_predictions": list(predicted_answers),
                "validated_answer": [ans["answer_text"] for ans in prediction["answer_list"]],
                "precision": curr_prec,
                "recall": curr_rec,
                "f1": curr_f1,
                "gold_answers_length": all_answers,
                "predicted_answers_num": len(predicted_answers),
                "correct_predictions": already_predicted,
                "num_correct": len(already_predicted),
            }
        )

        if save_path:
            mean_prec = np.mean(np.array(metrics['precision']))
            mean_rec = np.mean(np.array(metrics['recall']))
            mean_f1 = np.mean(np.array(metrics['f1']))
            f1_above = np.sum(np.array(metrics['f1']) >= 0.5) / float(len(metrics['f1']))
            rec_above = np.sum(np.array(metrics['recall']) >= 0.8) / float(len(metrics['f1']))

            with open(save_path, "w", encoding="utf-8") as file:
                json.dump(
                    {
                        "avg_precision": mean_prec,
                        "avg_recall": mean_rec,
                        "avg_f1": mean_f1,
                        "avg_above_recall": rec_above,
                        "avg_above_f1": f1_above,
                        "results": results,
                    },
                    file,
                    indent=2,
                    ensure_ascii=False,
                )

    if metrics['f1']:
        output = {
            "avg_precision": np.mean(np.array(metrics['precision'])),
            "avg_recall": np.mean(np.array(metrics['recall'])),
            "avg_f1": np.mean(np.array(metrics['f1'])),
            "avg_above_recall": np.sum(np.array(metrics['recall']) >= 0.8) / float(len(metrics['f1'])),
            "avg_above_f1": np.sum(np.array(metrics['f1']) >= 0.5) / float(len(metrics['f1'])),
            "results": results,
        }
    else:
        output = {
            "avg_precision": 0.0,
            "avg_recall": 0.0,
            "avg_f1": 0.0,
            "avg_above_recall": 0.0,
            "avg_above_f1": 0.0,
            "results": results,
        }

    if save_path:
        print(f"Results saved to {save_path}")

    for key, val in output.items():
        if key == "results":
            continue
        print(f"{key}:\t {val}")

    return output


def parse_args():
    parser = argparse.ArgumentParser(description="Run QAMPARI trace evaluation.")
    parser.add_argument(
        "--input-path",
        "--input_path",
        dest="input_path",
        default=None,
    )
    parser.add_argument(
        "--save-path",
        "--save_path",
        dest="save_path",
        default=None,
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    evaluate_traces(args.input_path, args.save_path)
