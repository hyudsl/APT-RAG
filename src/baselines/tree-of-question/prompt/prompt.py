# Prompt B.1
ANSWER_INTEGRATOR = """
- Search the document for an answer span that exactly matches the intent of the user’s question.
- If the question and document are relevant, extract the answer span from the document that matches the user’s question intent.
- If the question and document are irrelevant, output None.
- Output in the following format:
{{"relevance": "relevant | irrelevant", "answer_span": "${{relevant span}}"}}
- The following is an example.
Symbol of Courage
{{"title": "Symbols that symbolize good luck", "summary": "Let’s take a look at the various symbols." ..."}}
{{"relevance": "irrelevant", "answer_span": "None"}}

Question: {question}
Document: {document}
Answer: """


ANSWER_INTEGRATOR_v2 = """
- Search the answer for an answer span that exactly matches the intent of the user’s question.
- If the question and answer are relevant, extract the answer span from the answer that matches the user’s question intent.
- If the question and answer are irrelevant, output None.
- Output in the following format:
{{"relevance": "relevant | irrelevant", "answer_span": "${{relevant span}}"}}
- The following is an example.
Symbol of Courage
{{"title": "Symbols that symbolize good luck", "summary": "Symbols of courage often mentioned include the lion, shield, and sword." ..."}}
{{"relevance": "relevant", "answer_span": "lion, shield, and sword"}}

Question: {question}
Answer: {answer}
Answer Span: """


# Prompt B.2
Q_DECOMPOSER = """
- Evaluates whether the user’s inquiry can be addressed through a single query in a search engine or whether it requires multiple searches to compile the necessary information.
- If multiple searches are required, decompose the question into multiple sentences.
- If a single search is required, return the user’s question without modification.
- If the answer to a previous question needs to be used again as a question, mark it as [ANS_N].
- The following is an example:
    Please recommend an electric car in a similar price range to the BMW i5.
    1. What is the price range of BMW i5?
    2. Please recommend an electric car in the price range of [ANS_1].

Question: {question}
Answer: """

# Prompt B.3
Q_GEN = """
- You are a model that generates queries to search users’ questions on search engines.
- Create one optimal search term to answer your question.
- Examples:
    Please recommend an electric car in a similar price range to the BMW i5.
    Query: recommendation of BMW i5 price range electric car.
    Please tell me the Samsung stock price.
    Query: Samsung stock price

Question: {question}
Query: """

# Prompt B.4
Q_EVAL = """
- Evaluates the semantic_coherence and answerability of each summary for the user question.
- Semantic Coherence: Evaluation of how the summary maintains a logical flow and relevance to the user’s question. Scores range from 1 (not at all) to 10 (exact match).
- Answerability: Estimation of the probability that the summary directly and completely answers the user question. Confidence is expressed as a percentage, with 0% indicating no confidence and 100% indicating complete confidence.
- Each summary’s overall assessment score is calculated by averaging the Semantic Coherence and Answerability results, converting Answerability from a 0%-100% score to a 1-10 scale.
- Examples:
    Why cosmetics review ratings are important
    [Cosmetics review rating meaning]: Cosmetics review rating is an indicator that evaluates product quality and user satisfaction. ...
    {{"semantic_coherence": 9, "answerability": 95, "overall_assessment": 9.5, "response_validity": true}}

Question: {question}
Response: {response}
Evaluation: """


ANSWER_GEN = """
Question: {question}
Search Query: {query}
Documents: {documents}

*** Above are provided with multiple excerpts of paragraphs and tables, each of document has a title, followed by the
actual content. Some of these documents might contain helpful information to answering the question. In case that the
information in the document is relevant, you may use it to solve the question. If a document is irrelevant feel free to ignore
it when answering.

Answer: """
# Search Query: {query}

# FINAL_ANSWER_GEN = """
# Please synthesize the following information to answer the original question.

# Original Question: {question}

# Collected Information:
# {context}

# Based on the information above, write a comprehensive answer to the original question.

# Answer: """


LEAF_SYS_PROMPT = """
You are a helpful question-answering assistant. Your task is to answer a complex question provided by the user. You may generate a brief explanation before presenting the final answer if necessary.
Even if your information is not up to date, you must answer the question based on the knowledge you have.
The response may consist of multiple sentences, but each sentence must be concise and clear. Since the answer will be passed to another agent for follow-up tasks, ensure that the response preserves sufficient context while remaining concise and clear.

Your response must strictly follow the format below:
Answers: {ANSWERS}

You must end your response immediately after the final answer. Do not include any additional text beyond the specified format. Even if your information is not up to date, you must still provide an answer based on your knowledge.
"""

ROOT_PROMPT = {
"monaco": """
Above are multiple sub question–answer (QA) pairs related to the current question.
Each sub-question represents a more specific formulation of the current question, and its corresponding answer contains information that may contribute to answering the current question.

Using the provided sub QA pairs, generate an answer to the current question that:
    - is **concise and direct**: state the answer without unnecessary explanation, padding, or repetition.
    - preserves relevant context so that each part of the answer is clearly tied to what it refers to (no decontextualized fragments or bare lists that omit what each item describes).
    - includes only what is needed to answer the question; omit filler and redundant phrasing.

Do not produce long explanations or prose when a short, clear answer suffices. Prefer brevity while remaining complete and unambiguous.
""",

"musique": """
Above are multiple sub question–answer (QA) pairs related to the current question.
Each sub-question represents a more specific formulation of the current question, and its corresponding answer may contain information that is directly relevant to answering the current question.

Using the provided sub QA pairs, generate an answer to the current question that:
    - is **concise and direct**: state the answer without unnecessary explanation, padding, or repetition.
    - includes only what is needed to answer the question; omit filler and redundant phrasing.

Do not produce long explanations or prose when a short, clear answer suffices. Prefer the shortest answer that still directly and unambiguously answers the question.
""",

"quest": """
Above are multiple sub question–answer (QA) pairs related to the current question.
Each sub-question represents a more specific formulation of the current question, and its corresponding answer may contain information that is directly relevant to answering the current question.

Using the provided sub QA pairs, generate an answer to the current question that:
    - is **concise and direct**: state the answer without unnecessary explanation, padding, or repetition.
    - includes only what is needed to answer the question; omit filler and redundant phrasing.

Do not produce long explanations or prose when a short, clear answer suffices. Prefer the shortest answer that still directly and unambiguously answers the question.
""",

"qampari": """
Above are multiple sub question–answer (QA) pairs related to the current question.
Each sub-question represents a more specific formulation of the current question, and its corresponding answer may contain information that is directly relevant to answering the current question.

Using the provided sub QA pairs, generate an answer to the current question that:
    - is **concise and direct**: state the answer without unnecessary explanation, padding, or repetition.
    - includes only what is needed to answer the question; omit filler and redundant phrasing.

Do not produce long explanations or prose when a short, clear answer suffices. Prefer the shortest answer that still directly and unambiguously answers the question.
"""
}



ROOT_SYS_PROMPT = {
"monaco": """
You are a helpful question answering assistant. Your task is to answer the question with high precision.
- List **only** answers that **directly and clearly** satisfy what the question asks. Do **not** include uncertain, tangential, or "possibly relevant" items.
- Prefer **fewer, correct** answers over a long list that mixes correct and incorrect items. When in doubt, omit the item.
- Your response must use the following format:
Answers: {ANSWERS}
""",

"musique": """
You are a helpful question answering assistant. Your task is to answer the question with high precision.
- List **only** answers that **directly and clearly** satisfy what the question asks. Do **not** include uncertain, tangential, or "possibly relevant" items.
- Prefer **fewer, correct** answers over a long list that mixes correct and incorrect items. When in doubt, omit the item.
- The question expects a **single correct answer**. Return only one answer.
- The answer must be a **short answer (entity or short phrase)**, not a full sentence.
- If multiple surface forms are possible, choose the single best one. Do **not** list aliases or alternative phrasings.
- Do **not** include any explanation or extra text before or after the answer line.

- Your response must use the following format:
Answers: answer
""",

"quest": """
You are a helpful question answering assistant. Your task is to answer the question with high precision.
- List **only** answers that **directly and clearly** satisfy what the question asks. Do **not** include uncertain, tangential, or "possibly relevant" items.
- Prefer **fewer, correct** answers over a long list that mixes correct and incorrect items. When in doubt, omit the item.
- Each answer must be a **short answer (entity or short phrase)**, not a full sentence.
- If multiple answers exist, separate them clearly using commas (,). If only one answer exists, return a single answer.
- List each distinct answer only once. For the same answer item, do **not** include aliases or alternative phrasings.
- Do **not** include any explanation or extra text before or after the answer line.

- Your response must use the following format:
Answers: answer1, answer2, ..., answerN
""",

"qampari": """
You are a helpful question answering assistant. Your task is to answer the question with high precision.
- List **only** answers that **directly and clearly** satisfy what the question asks. Do **not** include uncertain, tangential, or "possibly relevant" items.
- Prefer **fewer, correct** answers over a long list that mixes correct and incorrect items. When in doubt, omit the item.
- Each answer must be a **short answer (entity or short phrase)**, not a full sentence.
- If multiple answers exist, separate them clearly using commas (,). If only one answer exists, return a single answer.
- List each distinct answer only once. For the same answer item, do **not** include aliases or alternative phrasings.
- Do **not** include any explanation or extra text before or after the answer line.

- Your response must use the following format:
Answers: answer1, answer2, ..., answerN
"""
}


SYSTEM_PROMPT = """
You are a helpful question answering assistant. Your task is to answer a complex question provided by the user. You may generate an explanation before providing the answer. The answer must be generated as a concise list of one or more entities, numbers or dates. You must always answer the question, even if your information is not up-to-date, please answer based on it.
Your response must use the following format:
Answers: {ANSWERS}
Where ANSWERS is a list of potential answers, separated by commas. You must end your response after the final answer. You must always answer the question, even if your information is not up-to-date, please answer based on it.
"""
