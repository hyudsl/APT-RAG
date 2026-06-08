REASONING_PLAN_PROMPT = """
You are a reasoning DAG generator expert. The goal is to make a reasoning DAG with minimum nodes. Given a query, if it is complex and requires a reasoning plan, split it into smaller, independent, and individual subqueries. The query and subqueries are used to construct a rooted DAG so make sure there are NO cycles and all nodes are connected, there is only one leaf node with a single root and one sink. DAG incorporates Markov property i.e. you only need the answer of the parent to answer the subquery. The main query should be the parent node of the initial set of subatomic queries such that the DAG starts with it. Return a Python list of tuples of parent query and the subatomic query which can be directly given to eval().

Strictly follow the below template for output.

For the subquery generation, input a tag <AI.J> where the answer of the parent query should come to make the query complete.

NOTE: Make the DAG connected and for simple queries return the original query only without any reasoning DAG.

Example:

Query: Who is the current PM of India?
DAG: "Q: Who is the current PM of India?"

Query: What is the tallest mountain in the world and how tall is it?
DAG: [ 
	("Q: What is the tallest mountain in the world and how tall is it?", "Q1.1: What is the tallest mountain in the world?"), 
	("Q1.1: What is the tallest mountain in the world?", "Q2.1: How tall is <A1.1>?")
]

Query: What percentage of the worlds population lives in urban areas? 
DAG: [ 
	("Q: What percentage of the worlds population lives in urban areas?", "Q1.1: What is the total world population?"), 
	("Q: What percentage of the worlds population lives in urban areas?", "Q1.2: What is the total population living in urban areas worldwide?"), 
	("Q1.1: What is the total world population?", "Q2.1: Calculate the percentage living in urban areas worldwide when total population is <A1.1> and population living in urban areas is <A1.2>?"), 
	("Q1.2: What is the total population living in urban areas worldwide?", "Q2.1: Calculate the percentage living in urban areas worldwide when total population is <A1.1> and population living in urban areas is <A1.2>?") 
]

Query: {query}
DAG: """


DYNAMIC_SUBQUERY_GENERATION_PROMPT = """
You are provided a question with tags where the corresponding tag answers need to be replaced. Replace tags with answers of the previous question in such a way that the final question is coherent and logical.
Just replace all parts of the answers in the main question. Do not reason or answer the question. Your role is just to replace tags with all parts of the answer.

NOTE: Only output the question with no explanation or any other details.

Example:

Query: Q2.1: Who was the president of India when the captain of the Indian cricket team was <A1.1> and vice-captain was <A1.2> in 2018? 
Q1.1: Who was the captain of India cricket team in 2018? 
A1.1: The captain of Indian cricket team in 2018 was M.S.Dhoni. 
Q1.2: Who was the vice-captain of India cricket team in 2018? 
A1.2: The vice-captain of Indian cricket team in 2018 was Virat Kohli. 
Output: Q2.1: Who was the president of India when the captain of the Indian cricket team was M.S.Dhoni and vice-captain was Virat Kohli?

Query: {query}
{parent_qa_pairs}
Output: """


##### for fianl answer generation ####
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












##### for sub answer generation ####

### v1

SUBANS_GEN_PROMPT = """
You are a concise answering assistant. If relevant and provided, use the Retrievals while generating the answer or use your own knowledge. If Known answers are given, use them while generating the response. Generate a JSON with a single key "Response" and a value that is a short phrase or a few words. In JSON, put every value as a string always, not float. Strictly follow the format below, and provide only the "Generation" part.

Example:

Query: Who was the PM of India when India performed its first nuclear test? 
Retrievals: [["Indira Gandhi was the PM of India in 1974"], ["Indira Gandhi was the first woman PM of India."]] 
Known answers: Q=When did India perform the first nuclear test? A=India conducted its first nuclear test on May 18, 1974, at the Pokhran Test Range in Rajasthan, India. 
Generation: {{ 
	"Response": "Indira Gandhi was the PM of India when India performed its first nuclear test in 1974." 
}}

Query: What type of literature did John Keble write? 
Retrievals: [["John Keble (1792-1866) was an English clergyman, poet, and theologian, best known for his contributions to religious poetry"], ["Keble wrote essays and sermons emphasizing the importance of tradition, the authority of the Church, and the significance of the sacraments."]] 
Generation: {{ 
	"Response": "Religious and devotional poetry" 
}}

Query: {query}
Known answers: {known_answers}
Retrievals: {retrievals}
Generation: """


### v2


LEAF_SYS_PROMPT = """
You are a helpful question-answering assistant. Your task is to answer a complex question provided by the user. You may generate a brief explanation before presenting the final answer if necessary.
Even if your information is not up to date, you must answer the question based on the knowledge you have.
The response may consist of multiple sentences, but each sentence must be concise and clear. Since the answer will be passed to another agent for follow-up tasks, ensure that the response preserves sufficient context while remaining concise and clear.

Your response must strictly follow the format below:
Generation: {{ 
	"Response": "{ANSWERS}" 
}}

You must end your response immediately after the final answer. Do not include any additional text beyond the specified format. Even if your information is not up to date, you must still provide an answer based on your knowledge.
"""


ANSWER_GEN = """
If relevant and provided, use the Retrievals while generating the answer or use your own knowledge. If Known answers are given, use them while generating the response. Generate a JSON with a single key "Response". Strictly follow the format below, and provide only the "Generation" part.

Example:

Question: Who was the PM of India when India performed its first nuclear test? 
Search Query: PM of India first nuclear test year
Documents: title: ...
content: "Indira Gandhi was the PM of India in 1974"
title: ...
content: "Indira Gandhi was the first woman PM of India."
Known answers: Q=When did India perform the first nuclear test? A=India conducted its first nuclear test on May 18, 1974, at the Pokhran Test Range in Rajasthan, India. 
Generation: {{ 
	"Response": "Indira Gandhi was the PM of India when India performed its first nuclear test in 1974." 
}}

Question: What type of literature did John Keble write? 
Search Query: John Keble literature type
Documents: title: ...
content: "John Keble (1792-1866) was an English clergyman, poet, and theologian, best known for his contributions to religious poetry"
title: ...
content: "Keble wrote essays and sermons emphasizing the importance of tradition, the authority of the Church, and the significance of the sacraments."
Generation: {{ 
	"Response": "Religious and devotional poetry" 
}}

---

Question: {question}
Search Query: {query}
Known answers: {known_answers}
Documents: {retrievals}

*** Above are provided with multiple excerpts of paragraphs and tables, each of document has a title, followed by the
actual content. Some of these documents might contain helpful information to answering the question. In case that the
information in the document is relevant, you may use it to solve the question. If a document is irrelevant feel free to ignore
it when answering.

Generation: """
