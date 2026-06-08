# original code:  monaco/prompts/answer_judgement_prompt_V2.py



SINGLE_ANSWER_LLM_JUDGE = """Judge whether the following [response] to [question] is correct or not based on the precise and unambiguous [correct_answer] below.
[question]: {question}
[response]: '{response}'

Your judgment must be in the format and criteria specified below:

extracted_final_answer: The final exact answer extracted from the [response]. Put the extracted answer as ’None’ if there is no exact, final answer to extract from the response.

[correct_answer]: {correct_answer}

reasoning: Explain why the extracted_final_answer is correct or incorrect based on [correct_answer], focusing only on if there are meaningful differences between [correct_answer] and the extracted_final_answer. Do not comment on any background to the problem, do not attempt to solve the problem, do not argue for any answer different than [correct_answer], focus only on whether the answers match.

correct: Answer ’yes’ if extracted_final_answer matches the [correct_answer] given above, or is within a small margin of error for numerical problems, a margin of 1 to 3.5 percentage points is acceptable. Answer ’no’ otherwise, i.e. if there is any inconsistency, ambiguity, non-equivalency, or if the extracted answer is incorrect.

precision: Answer ’1’ if extracted_final_answer matches the [correct_answer] given above. Answer ’0’ otherwise, i.e. if there is any inconsistency, ambiguity, non-equivalency, or if the extracted answer is incorrect. In the case where [correct_answer] is a number or percentage, then answer with the following formula to compute the normalized similarity score: [1 - (abs([correct_answer] - extracted_final_answer) / max(abs([correct_answer]), abs(extracted_final_answer)))]

final precision: Extract the precision score from above, just the final score (number).
"""


MULTI_ANSWER_LLM_JUDGE = """Judge whether the following [response] to [question] is correct or not based on the precise and unambiguous [correct_answer] below.
[question]: {question}
[response]: '{response}'

Your judgment must be in the format and criteria specified below:

extracted_final_answer: The final exact answer extracted from the [response]. Put the extracted answer as ’None’ if there is no exact, final answer to extract from the response.

[correct_answer]: {correct_answer}

final answer length: Provide the overall number of unique answers that appear in [response], not just the correct ones. Be sure to provide a number, not an estimate! 

reasoning: Explain why the extracted_final_answer is correct or incorrect based on [correct_answer], focusing only on if there are meaningful differences between [correct_answer] and the extracted_final_answer. Do not comment on any background to the problem, do not attempt to solve the problem, do not argue for any answer different than [correct_answer], focus only on whether the answers match.

correct: Answer ’yes’ if extracted_final_answer matches the [correct_answer] given above, or is within a small margin of error for numerical problems, a margin of 1 to 5.5 percentage points is acceptable. Answer ’no’ otherwise, i.e. if there is any inconsistency, ambiguity, non-equivalency, or if the extracted answer is incorrect.

overlapping answers: List all of the answers in [response] that also appear in [correct_answer]. You can consider an answer from [response] to match with an answer in [correct_answer] if it is equivalent or is within a small margin of error for numerical problems, a margin of 1 to 5.5 percentage points is acceptable. List all of the [response] answer appearing in [correct_answer] with each answer delimited by '###'. If the number of overlapping answers is zero, output 'NULL'.
"""