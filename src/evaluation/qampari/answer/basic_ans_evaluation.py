# .../qampari/models/evaluation/reader_metrics.py


import argparse
import json
import os
import regex
import string
import numpy as np
from typing import Dict, List


def create_inverse_mappings(mapping):

    """
    Based on a mapping, inverses it and returns a dic of aliases as key and their basic name as value.
    :param mapping: dictionary of answer names as key and a list of their aliases as values
    """

    inverse_mapping = dict()
    for key in mapping:
        for alias in mapping[key]:
            inverse_mapping[alias] = key
    return inverse_mapping


def load_data(input_path: str):

    """
    Loads the data depending on the format of the file:  a jsonl or json
    :param input_path: path to  where the data is kept
    :return: the loaded data represented with a list
    """

    with open(input_path, 'r') as f:
        if 'jsonl' in input_path:
            samples = list()
            for line in f:
                samples.append(json.loads(line))
            return samples
        else:
            data = json.load(f)
            return data


def parse_generation_to_predictions(generation: str) -> List[str]:
    """
    Parse LLM generation string (e.g., "Answers: X, Y, Z") into list of answer strings.
    Handles formats: "Answers: X", "Answers: X, Y, Z", "Answer: X", etc.
    """
    if not generation or not isinstance(generation, str):
        return []
    text = generation.strip()
    # Strip common prefixes (case-insensitive)
    for prefix in ["Answers:", "Answer:", "answers:", "answer:"]:
        if text.lower().startswith(prefix.lower()):
            text = text[len(prefix):].strip()
            break
    if not text:
        return []
    # Split by comma, newline, or # (FiD-style)
    parts = []
    for sep in [",", "\n", "#"]:
        if sep in text:
            parts = [p.strip() for p in text.split(sep) if p.strip()]
            break
    if not parts:
        parts = [text] if text else []
    return parts


def transform_llm_results_format(data: List) -> List:
    """
    Transform LLM_only / Baseline results format to answer_evaluation expected format.
    Input: [{"id", "ex_num", "query", "generation", "info": {"answer_list", ...}}, ...]
    Output: [{"answer_list": [...], "predictions": list of str, "ex_num", "query", "generation"}, ...]
    """
    transformed = []
    for item in data:
        if "predictions" in item and "answer_list" in item:
            # Already in expected format - ensure metadata for logging
            item = dict(item)
            item.setdefault("ex_num", None)
            item.setdefault("query", "")
            item.setdefault("generation", str(item.get("predictions", [])))
            transformed.append(item)
            continue
        if "generation" in item and "info" in item:
            answer_list = item["info"].get("answer_list", [])
            predictions = parse_generation_to_predictions(item.get("generation", ""))
            transformed.append({
                "answer_list": answer_list,
                "predictions": predictions,
                "ex_num": item.get("ex_num"),
                "query": item.get("query", ""),
                "generation": item.get("generation", ""),
            })
        elif "generation" in item and "validated_answer" in item:
            answer_list = item.get("validated_answer", [])
            predictions = parse_generation_to_predictions(item.get("generation", ""))
            transformed.append({
                "answer_list": answer_list,
                "predictions": predictions,
                "ex_num": item.get("ex_num"),
                "query": item.get("query", ""),
                "generation": item.get("generation", ""),
            })
        else:
            raise ValueError(
                f"Unknown format. Expected 'generation' + 'info.answer_list', "
                f"'generation' + 'validated_answer', or "
                f"'predictions' + 'answer_list'. Got keys: {list(item.keys())}"
            )
    return transformed


def _load_existing_results(output_path: str) -> List[Dict]:
    if not output_path or not os.path.exists(output_path):
        return []

    with open(output_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    if isinstance(data, dict):
        results = data.get("results", [])
        if isinstance(results, list):
            return results

    return []


def _build_evaluated_query_set(results: List[Dict]) -> set[str]:
    evaluated = set()
    for item in results:
        evaluated.add(item["query"].strip())
    return evaluated


def normalize_answer(s):
    def remove_articles(text):
        return regex.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))

def exact_match_score(prediction, ground_truth):
    return normalize_answer(prediction) == normalize_answer(ground_truth)


def create_mapping(prediction: Dict):

    """
    Create a mapping from an answer name to its aliases
    """

    mapping = dict()
    for ans in prediction['answer_list']:
        mapping[ans['answer_text']] = ans['aliases']

    return mapping

def compute_metrics_qampari(predictions: List):

    """
    Computes the metrics for QAMPARI predictions.
    :param predictions: all the QAMPARI predictions
    :return: dictionary of the computed metrics
    """

    metrics = {'precision': list(), 'recall': list(), 'f1': list()}
    for pred in predictions:

        # checks the answers correctly predicted
        already_predicted = list()
        mapping = create_mapping(pred)
        inversed_mapping = create_inverse_mappings(mapping)
        predicted_answers = set(pred['predictions'])
        all_answers = len(pred['answer_list'])
        for predicted in predicted_answers:
            for alias in inversed_mapping:
                if exact_match_score(predicted, alias):
                    if inversed_mapping[alias] not in already_predicted:
                        already_predicted.append(inversed_mapping[alias])
                    break

        # computes the metrics
        curr_prec = (len(already_predicted) / len(predicted_answers)) if len(predicted_answers) > 0 else 0
        curr_rec = (len(already_predicted) / all_answers) if all_answers > 0 else 0
        if curr_rec == 0 or curr_prec == 0:
            metrics['f1'].append(0)
        else:
            metrics['f1'].append(((2 * curr_rec * curr_prec) / (curr_rec + curr_prec)))
        metrics['precision'].append(curr_prec)
        metrics['recall'].append(curr_rec)

    mean_prec = np.mean(np.array(metrics['precision']))
    mean_rec = np.mean(np.array(metrics['recall']))
    mean_f1 = np.mean(np.array(metrics['f1']))
    f1_above = np.sum(np.array(metrics['f1']) >= 0.5) / float(len(metrics['f1']))
    rec_above = np.sum(np.array(metrics['recall']) >= 0.8) / float(len(metrics['f1']))
    return {'f1': mean_f1, 'recall': mean_rec, 'precision': mean_prec, 'f1_above': f1_above, 'rec_above': rec_above}


def compute_metrics_qampari_with_details(predictions: List) -> tuple:
    """
    Computes QAMPARI metrics and returns per-example details for logging.
    :return: (metrics_dict, results_list)
    """
    metrics = {'precision': list(), 'recall': list(), 'f1': list()}
    results = []

    for pred in predictions:
        already_predicted = list()  # gold answer_text that were correctly predicted
        mapping = create_mapping(pred)
        inversed_mapping = create_inverse_mappings(mapping)
        predicted_answers = set(pred['predictions'])
        all_answers = len(pred['answer_list'])

        for predicted in predicted_answers:
            for alias in inversed_mapping:
                if exact_match_score(predicted, alias):
                    if inversed_mapping[alias] not in already_predicted:
                        already_predicted.append(inversed_mapping[alias])
                    break

        curr_prec = (len(already_predicted) / len(predicted_answers)) if len(predicted_answers) > 0 else 0
        curr_rec = (len(already_predicted) / all_answers) if all_answers > 0 else 0
        if curr_rec == 0 or curr_prec == 0:
            metrics['f1'].append(0)
        else:
            metrics['f1'].append(((2 * curr_rec * curr_prec) / (curr_rec + curr_prec)))
        metrics['precision'].append(curr_prec)
        metrics['recall'].append(curr_rec)

        # Build per-example result for logging
        parsed_predictions = list(predicted_answers)  # parsed/extracted answers from model output
        validated_answer = [ans["answer_text"] for ans in pred["answer_list"]]  # gold answers from answer_list
        result_entry = {
            "ex_num": pred.get("ex_num"),
            "query": pred.get("query", ""),
            "output": pred.get("generation", ""),
            "parsed_predictions": parsed_predictions,
            "validated_answer": validated_answer,
            "precision": metrics['precision'][-1],
            "recall": metrics['recall'][-1], 
            "f1": metrics['f1'][-1],
            "gold_answers_length": all_answers,
            "predicted_answers_num": len(predicted_answers),
            "correct_predictions": already_predicted,
            "num_correct": len(already_predicted),
        }
        results.append(result_entry)

    mean_prec = np.mean(np.array(metrics['precision']))
    mean_rec = np.mean(np.array(metrics['recall']))
    mean_f1 = np.mean(np.array(metrics['f1']))
    f1_above = np.sum(np.array(metrics['f1']) >= 0.5) / float(len(metrics['f1']))
    rec_above = np.sum(np.array(metrics['recall']) >= 0.8) / float(len(metrics['f1']))

    agg_metrics = {
        'avg_precision': mean_prec,
        'avg_recall': mean_rec,
        'avg_f1': mean_f1,
        'avg_above_f1': f1_above,
        'avg_above_recall': rec_above,
    }
    return agg_metrics, results

def compute_metrics_nq(predictions: List):

    """
    Computes the EM metric for preictions from NQ.
    """

    has_em = list()
    for pred in predictions:
        em_val = 0
        all_answers = pred['answers']
        for alias in all_answers:
            if exact_match_score(alias, pred['prediction']):
                em_val = 1
                break
        has_em.append(em_val)

    return {'em': np.mean(np.array(has_em))}



def main(input_path: str, output_path: str = None, is_nq: bool = False, llm_format: bool = False, detailed_log: bool = True):

    """
    Expect input to a prediction file. Expect a list of dicts of the following (minimum) format:
    {"answer_list": list of dict({"answer_text": str, "aliases": list of str}),
     "answers": list of str,
     "predictions": list of str}

    If llm_format=True, accepts LLM_only/Baseline format:
    {"id", "ex_num", "query", "generation": "Answers: X, Y, Z", "info": {"answer_list": [...]}}

    If detailed_log=True and output_path is set, saves per-example results in the output file.
    """

    data = load_data(input_path)
    if llm_format:
        data = transform_llm_results_format(data)

    if is_nq:
        metrics = compute_metrics_nq(data)
        if output_path is not None:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(metrics, f, indent=2, ensure_ascii=False)
    else:
        if detailed_log and output_path is not None:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            results = _load_existing_results(output_path)
            evaluated_queries = _build_evaluated_query_set(results)
            precision_list = [item["precision"] for item in results]
            recall_list = [item["recall"] for item in results]
            f1_list = [item["f1"] for item in results]

            for pred in data:
                query = pred["query"].strip()
                if query in evaluated_queries:
                    continue

                _, new_results = compute_metrics_qampari_with_details([pred])
                result_entry = new_results[0]
                results.append(result_entry)
                precision_list.append(result_entry["precision"])
                recall_list.append(result_entry["recall"])
                f1_list.append(result_entry["f1"])

                mean_prec = np.mean(np.array(precision_list))
                mean_rec = np.mean(np.array(recall_list))
                mean_f1 = np.mean(np.array(f1_list))
                f1_above = np.sum(np.array(f1_list) >= 0.5) / float(len(f1_list))
                rec_above = np.sum(np.array(recall_list) >= 0.8) / float(len(f1_list))

                with open(output_path, 'w', encoding='utf-8') as f:
                    json.dump(
                        {
                            "avg_precision": mean_prec,
                            "avg_recall": mean_rec,
                            "avg_f1": mean_f1,
                            "avg_above_recall": rec_above,
                            "avg_above_f1": f1_above,
                            "results": results,
                        },
                        f,
                        indent=2,
                        ensure_ascii=False,
                    )

            if f1_list:
                metrics = {
                    'avg_precision': np.mean(np.array(precision_list)),
                    'avg_recall': np.mean(np.array(recall_list)),
                    'avg_f1': np.mean(np.array(f1_list)),
                    'avg_above_f1': np.sum(np.array(f1_list) >= 0.5) / float(len(f1_list)),
                    'avg_above_recall': np.sum(np.array(recall_list) >= 0.8) / float(len(f1_list)),
                }
            else:
                metrics = {
                    'avg_precision': 0.0,
                    'avg_recall': 0.0,
                    'avg_f1': 0.0,
                    'avg_above_f1': 0.0,
                    'avg_above_recall': 0.0,
                }
        else:
            metrics = compute_metrics_qampari(data)
            if output_path is not None:
                with open(output_path, 'w', encoding='utf-8') as f:
                    json.dump(metrics, f, indent=2, ensure_ascii=False)

    for key, val in metrics.items():
        print(f'{key}:\t {val}')


def parse_args():
    parser = argparse.ArgumentParser(description="Run QAMPARI basic evaluation.")
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
    parser.add_argument("--is-nq", dest="is_nq", action="store_true")
    parser.add_argument("--llm-format", dest="llm_format", action="store_true")
    parser.add_argument("--no-llm-format", dest="llm_format", action="store_false")
    parser.add_argument("--detailed-log", dest="detailed_log", action="store_true")
    parser.add_argument("--no-detailed-log", dest="detailed_log", action="store_false")
    parser.set_defaults(is_nq=False, llm_format=True, detailed_log=True)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(
        args.input_path,
        args.save_path,
        is_nq=args.is_nq,
        llm_format=args.llm_format,
        detailed_log=args.detailed_log,
    )