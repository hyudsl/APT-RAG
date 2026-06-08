# original code: monaco/prompts/evaluate_final_answers.py


from typing import Dict, Optional, List, Union

class LLMJudgeScorerV2:
    def __init__(self, llm_judgement: str, gold_answers_length: int):
        self.llm_judgement = self._normalize_judgement(llm_judgement)
        self.gold_answers_length = gold_answers_length

    @staticmethod
    def _normalize_judgement(text: str) -> str:
        text = text.replace("final_answer_length", "final answer length")
        text = text.replace("overlapping_answers", "overlapping answers")
        text = text.replace("final_precision", "final precision")
        return text

    def _extract_single_answer_score(self) -> dict:
        try:
            prec_str = self.llm_judgement.split("final precision:")[1].split("\n")[0]
            prec = float(prec_str.replace("...", ""))
        except (ValueError, IndexError):
            prec = 0.0
        return {"judge_score": prec, "precision": prec, "recall": prec}

    def _extract_multi_answer_scores(self) -> Optional[Dict]:
        llm_judgement = self.llm_judgement
        gold_answers_length = self.gold_answers_length

        length_keyword = "\nfinal answer length:"
        if length_keyword not in llm_judgement:
            return None

        len_substr = llm_judgement.replace("final answer length: None ", "final answer length: 0 ")
        len_substr = len_substr.replace("The response lists over", "")
        len_substr = len_substr.split(length_keyword)[1].strip().split("\n")[0].strip()
        len_substr = len_substr.split(" ")[0].strip() if " " in len_substr else len_substr
        predicted_length = int(len_substr)

        correct_predicted_answers_keyword = "\noverlapping answers:"
        if correct_predicted_answers_keyword not in llm_judgement:
            return None

        answer_delimiter = "###"
        empty_answer_keyword = "NULL"

        answers_chunk = llm_judgement.split(correct_predicted_answers_keyword)[1].replace(
            answer_delimiter + empty_answer_keyword, answer_delimiter
        ).strip()

        if answers_chunk.endswith(answer_delimiter):
            answers_chunk = answers_chunk[:-3]

        answers = answers_chunk.split(answer_delimiter)
        num_correct = len(answers) if answers != [empty_answer_keyword] else 0

        if predicted_length == 0.0:
            recall = 0.0
            precision = 0.0
            f1 = 0.0
        else:
            if num_correct != 0:
                recall = float(min(num_correct, gold_answers_length)) / gold_answers_length
                predicted_length = max(predicted_length, num_correct)
                precision = float(num_correct) / predicted_length
                f1 = (2 * (precision * recall)) / (precision + recall)
            else:
                recall = 0.0
                precision = 0.0
                f1 = 0.0

        return {
            "judge_score": f1,
            "precision": precision,
            "recall": recall,
            "gold answers length": gold_answers_length,
            "predicted answers num": predicted_length,
            "correct predictions": answers,
            "num correct": num_correct,
        }

    def Evaluate(self) -> Optional[Dict]:
        if self.gold_answers_length == 1:
            return self._extract_single_answer_score()
        if self.gold_answers_length > 1:
            return self._extract_multi_answer_scores()
        raise Exception("gold_answers_length must be >= 1")
