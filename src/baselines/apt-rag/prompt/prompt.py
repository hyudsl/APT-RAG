########################## Planning prompt ###########################

DECOMPOSITION_PROMPT = """
You are a Decomposer who analyzes an input question and determines whether it should be handled as a single search (maintain) or decomposed into sub-queries (split).
---
[Input]
- input question: The user’s question that requires analysis and an execution strategy.
---
[Maintain / Split Decision Criteria]

1. Conditions for Splitting
A question should be divided into multiple sub-queries when **multiple pieces of data must be assembled**.

- Heterogeneous Data Sources (Multiple Domains):
	- When the required information belongs to different domains and is unlikely to be found in a single source (webpage, site, or document).
	- Example: "Who is older, the CEO of Apple or the CEO of Samsung Electronics?"
	
- Stepwise Dependency:
	- When the question contains filtering or conditions such that one result must be identified before the next search can proceed (Result of A → Input of B).
	- Example: "What is the average literacy rate of the five most populous countries in the world?"

- Complex Operations or Processing:
	- When the task requires more than simple lookup, such as sorting, ranking, or comparison.
	- Example: "List the G7 countries in order of defense spending."
	 
2. Conditions for Maintaining
A question should be sent directly to a search engine or database as-is when the data is already bundled together.

- Single Entity / Single Domain:
	- When the requested information belongs to one domain category and is likely to be found in a single source.
	- Example: "How many patents did Apple file in 2024?"

- Closed or Common Knowledge Set with Simple Enumeration:
	- When the question refers to a well-known fixed list or set, where the question itself represents a single knowledge unit.
	- Example: "List the planets of the Solar System in order from the Sun." (It would be inefficient to search for Mercury, Venus, etc. individually.)

- Explicit Relational Query:
	- When the relationship between two entities is itself the core search keyword.
	- Example: "When was the Free Trade Agreement (FTA) between South Korea and the United States signed?"
---
[Decomposition Rules]
- If the decision is to split, you must generate appropriate sub-queries.
-	Dependency Notation: If a sub-query depends on the result of a previous query, you must indicate this dependency using the notation <Qn>, where n refers to the number of the required prior query.
-	<Qn> signifies that the result of the nth query is required as input.
---
[Output Format]

The output must follow below JSON structure:
{{
  "decision": "maintain" or "split",
  "subqueries": [] or ["sub-query 1", "sub-query 2", ...]
}}

- If you decide to maintain, set "decision" to "maintain" and provide an empty list for "subqueries".
- If you decide to split, set "decision" to "split" and populate "subqueries" according to the decomposition rules above.
---
[Examples]
input question: For each European country, find its population and then identify the two countries with the smallest populations.
output:
{{
"decision": "split",
"subqueries": [
	"1. What is the list of countries in Europe?",
	"2. For each country identified in <Q1>, what is its total population?",
	"3. From the countries and populations identified in <Q2>, which two countries have the smallest populations?"
	]
}}

input question: Which universities did the CEOs of major global tech companies graduate from?
output:
{{
"decision": "split",
"subqueries": [
	"1. Who are the CEOs of major global tech companies?",
	"2. For each CEO in <Q1>, which university did they graduate from?"
	]
}}

input question: Tell me Elon Musk’s date of birth, educational background, and the companies he founded.
output:
{{
"decision": "maintain",
"subqueries": []
}}

input question: 1. Find the list of winners of the Emmy Awards’ “Outstanding Drama Series” category from 2010 to 2022.
output:
{{
"decision": "maintain",
"subqueries": []
}}

input question: 2. What is the percentage of the Amerindian (Indigenous) population in each of the following countries: Argentina, Bolivia, Brazil, Chile, Colombia, Costa Rica, Cuba, Dominican Republic, Ecuador, El Salvador, Guatemala, Honduras, Mexico, Nicaragua, Panama, Paraguay, Peru, Puerto Rico, Uruguay, Venezuela, and French Guiana?
output:
{{
"decision": "split",
"subqueries": [
	"1. What is the percentage of the Amerindian (Indigenous) population in Argentina?",
	"2. What is the percentage of the Amerindian (Indigenous) population in Bolivia?",
	"3. What is the percentage of the Amerindian (Indigenous) population in Brazil?",
	...
	"20. What is the percentage of the Amerindian (Indigenous) population in Venezuela?",
	"21. What is the percentage of the Amerindian (Indigenous) population in French Guiana?"
	]
}}

input question: 3. Among the following Emmy-winning shows, which ones depict stories about royal families or feature them as a central theme: Mad Men, Breaking Bad, Game of Thrones, Succession, and The Crown?
output:
{{
"decision": "split",
"subqueries": [
	"1. Does the TV series "Mad Men" depict stories about royal families or feature them as a central theme??",
	"2. Does the TV series "Breaking Bad" depict stories about royal families or feature them as a central theme?",
	"3. Does the TV series "Game of Thrones" depict stories about royal families or feature them as a central theme?",
	"4. Does the TV series "Succession" depict stories about royal families or feature them as a central theme?",
	"5. Does the TV series "The Crown" depict stories about royal families or feature them as a central theme?"
	]
}}
---
input question: {query}
output: 
"""

ANSWERABILITY_CHECK_PROMPT = """
You are an answerability checker who determines whether external search is required to answer the current question.

[Input]
- input question: The question for which you must determine whether search is necessary
- context: A list of previous question–answer (QA) pairs that provide context relevant to the current question.
---
[Answerability Decision Rules]
1. Sufficiency of Information
	-	Skip search: If the core entity and required attributes (e.g., person, numerical value, definition) are already explicitly present in the provided context
	-	Perform search: If the target entity or required detailed attributes are missing or uncertain in the provided context

2. Nature of the Question
	-	Skip search: If the question asks to compare, summarize, or reorganize information that is already fully available in the provided context
	-	Perform search: If the question requires new information or asks about an entity not present in the provided context
---
[Search Decision Process]
- Step 1. Entity Check: Determine whether the subject (entity) asked in the question already exists in the provided context.
- Step 2. Attribute Check: Determine whether the specific attributes (e.g., numerical values, characteristics, definitions) required by the question are present in the provided context.
- Step 3. Intent Check: Determine whether the question requires new information or merely reorganizes information already in the provided context.
- Step 4. Final Decision: Based on Steps 1–3, decide whether external search is required.

---
[Output Format]
- Output must strictly follow the JSON format below:
{{
  "step 1": "Content corresponding to Step 1 (maximum 2 sentences)",
  "step 2": "Content corresponding to Step 2 (maximum 2 sentences)",
  "step 3": "Content corresponding to Step 3 (maximum 2 sentences)",
  "step 4": True or False
}}
---
[Examples]
input question: "In what year was the Treaty of Lisbon signed?"
context:
	Q1: "What are the main provisions and key dates of the Treaty of Lisbon?"
	A1: "The Treaty of Lisbon was signed on 13 December 2007 and entered into force on 1 December 2009. It reformed EU institutions and decision-making procedures."
output:
{{
  "step 1": "The Treaty of Lisbon is the subject entity and appears in the provided context.",
  "step 2": "The signing year (2007) is explicitly stated in the provided context.",
  "step 3": "The question asks for a specific date already contained in the context rather than new external facts.",
  "step 4": false
}}

input question: "Compare the main revenue models of Apple and Netflix, and analyze which company has a more stable structure."
context:
	Q1: "What are the revenue models of Apple and Netflix?"
	A1: "Apple primarily generates revenue through hardware sales such as the iPhone and service commissions from the App Store. In contrast, Netflix’s revenue is almost entirely based on monthly subscription fees for its streaming service."
output:
{{
  "step 1": "The entities Apple and Netflix are already defined in the provided context.",
  "step 2": "The revenue model attributes (hardware sales, service commissions, subscription fees) are explicitly provided in the provided context.",
  "step 3": "The question asks for comparison and analysis of information already present in the context rather than new data.",
  "step 4": false
}}

input question: "What is the official population of Tokyo as of 2020?"
context:
	Q1: "What is the capital of France?"
	A1: "Paris is the capital and largest city of France, with a population of about 2.1 million within the city proper."
output:
{{
  "step 1": "The question concerns Tokyo, which does not appear in the provided context.",
  "step 2": "The required attribute—Tokyo’s official 2020 population—is absent from the provided context.",
  "step 3": "The question seeks factual information not contained in the context.",
  "step 4": true
}}

input question: "What is the recommended power consumption (TDP) of the Nvidia RTX 5090 graphics card?"
context:
	Q1: "What is the name of Nvidia’s newly announced next-generation graphics card?"
	A1: "Nvidia announced its next-generation flagship graphics card model named RTX 5090. It is expected to deliver significant performance improvements over the previous generation."
output:
{{
  "step 1": "The entity RTX 5090 is already identified in the provided context as Nvidia’s next-generation graphics card.",
  "step 2": "The specific technical specification—recommended power consumption (TDP)—is not present in the provided context.",
  "step 3": "The question requires confirming a precise technical value not available in the context.",
  "step 4": true
}}
---
input question: {query}
context:
"""

########################## Contextualization prompt ###########################

CONTEXTUALIZATION_PROMPT = """
You are a Contextualizer who extracts the key keywords needed to specify a vague query and rewrites the question in a more specific form.

[Input Information]
-	Target question: A question that requires specification. This question may contain reference expressions such as <Qn>.
	- <Qn> refers to the result (answer) of the nth question.
	- Example: "For each country identified in , what is the percentage of Amerindian (Indigenous) population?"
- Prior QA information: Previous questions and their corresponding answers that are necessary to specify the target question. This includes the question and answer corresponding to <Qn>.
---
[Keyword Extraction and Question Specification Process]
- Step 1. Analyze the target question: Identify the information that must ultimately be retrieved.
- Step 2. Identify required search conditions: Determine the search conditions that must be explicitly provided to a search engine or database in order to retrieve that information.
- Step 3. Extract keywords from prior answers: From the answers to <Qn>, extract the terms (identifiers, entities, constraints) that satisfy the search conditions identified in Step 2.
- Step 4. Rewrite the question: Using the extracted keywords, replace the vague pronouns or reference expressions in the target question (e.g., “it,” “identified in <Qn>,” “those,” etc.) with concrete information.

IMPORTANT RULES FOR STEP 3,4:
- Extract complete structured items from prior answers, including associated values when present, and preserve original formats (e.g., "entity (value)"). If values exist, NEVER omit numbers, units, percentages, or related data.
---
[Output Format]
- Output must follow the JSON format below
{{
  "step 1": "Content corresponding to Step 1 (maximum 2 sentences)",
  "step 2": "Content corresponding to Step 2 (maximum 2 sentences)",
  "step 3": ["keyword1", "keyword2", ...] or ["keyword1 (value1)", "keyword2 (value2)", ...],
  "step 4": "Rewritten question"
}}
---
[Examples]

target question: 2. For each country identified in <Q1>, what is the percentage of Amerindian (Indigenous) population?
Past QA information:
	Q: 1. What is the list of countries that belong to Latin America?
	A. The countries that belong to Latin America are Argentina, Bolivia, Brazil, Chile, Colombia, Costa Rica, Cuba, the Dominican Republic, Ecuador, El Salvador, Guatemala, Honduras, Mexico, Nicaragua, Panama, Paraguay, Peru, Puerto Rico (a territory of the United States), Uruguay, Venezuela, and the French overseas department of French Guiana (though not listed in the table, it is part of Latin America due to its French language and culture). The region includes nations in South America, Central America, and the Caribbean where Spanish, Portuguese, or French are the primary languages, reflecting a shared cultural and linguistic heritage rooted in Latin Europe. English- and Dutch-speaking countries such as Guyana, Suriname, Jamaica, Trinidad and Tobago, and Belize are excluded from Latin America despite their geographical location.
output:
{{
  "step 1": "To obtain the percentage of Amerindian (Indigenous) population in each Latin American country relative to its total population",
  "step 2": "To retrieve accurate figures from a search engine or database, the names of the individual countries identified in  must be included in the search query.",
  "step 3": ["Argentina", "Bolivia", "Brazil", "Chile", "Colombia", "Costa Rica", "Cuba", "Dominican Republic", "Ecuador", "El Salvador", "Guatemala", "Honduras", "Mexico", "Nicaragua", "Panama", "Paraguay", "Peru", "Puerto Rico", "Uruguay", "Venezuela", "French Guiana"],
  "step 4": "2. What is the percentage of the Amerindian (Indigenous) population in each of the following countries: Argentina, Bolivia, Brazil, Chile, Colombia, Costa Rica, Cuba, Dominican Republic, Ecuador, El Salvador, Guatemala, Honduras, Mexico, Nicaragua, Panama, Paraguay, Peru, Puerto Rico, Uruguay, Venezuela, and French Guiana?"
}}

target question: 3. Among the winners identified in <Q2>, which works depict stories about royal families or feature them as a central theme?
Past QA information:
	Q: 2. Find the list of winners of the Emmy Awards’ "Outstanding Drama Series" category from 2010 to 2022.
	A. The winners of the Primetime Emmy Award for Outstanding Drama Series from 2010 to 2022 are as follows:\n\n- 2010: Mad Men (AMC)\n- 2011: Mad Men (AMC)\n- 2012: Breaking Bad (AMC)\n- 2013: Breaking Bad (AMC)\n- 2014: Breaking Bad (AMC)\n- 2015: Game of Thrones (HBO)\n- 2016: Game of Thrones (HBO)\n- 2017: Game of Thrones (HBO)\n- 2018: Game of Thrones (HBO)\n- 2019: Game of Thrones (HBO)\n- 2020: Succession (HBO)\n- 2021: The Crown (Netflix)\n- 2022: Succession (HBO)
output:
{{
  "step 1": "To identify, among the Emmy Award winners for Outstanding Drama Series from 2010 to 2022, the works that depict royal families or treat them as a central theme.",
  "step 2": "To determine their relevance to royal families, the specific plot details and core themes of each winning work must be examined, and the individual titles identified in  must be included as search conditions.",
  "step 3": ["Mad Men", "Breaking Bad", "Game of Thrones", "Succession", "The Crown"],
  "step 4": "3. Among the following Emmy-winning shows, which ones depict stories about royal families or feature them as a central theme: Mad Men, Breaking Bad, Game of Thrones, Succession, and The Crown?"
}}

target question: 3. From the items identified in <Q2>, which two cities have the highest average annual rainfall?
Prior QA information:
	Q: 2. What is the average annual rainfall (mm) in Bangkok, Jakarta, Manila, Kuala Lumpur, Singapore, and Ho Chi Minh City?
	A: Bangkok (1,500 mm), Jakarta (1,800 mm), Manila (2,000 mm), Kuala Lumpur (2,400 mm), Singapore (2,300 mm), Ho Chi Minh City (1,900 mm)
output:
{{
  "step 1": "To identify the two cities with the highest average annual rainfall among the given Southeast Asian cities.",
  "step 2": "To determine this, the rainfall values associated with each city must be compared using the city names and their corresponding rainfall data.",
  "step 3": ["Bangkok (1,500 mm)", "Jakarta (1,800 mm)", "Manila (2,000 mm)", "Kuala Lumpur (2,400 mm)", "Singapore (2,300 mm)", "Ho Chi Minh City (1,900 mm)"],
  "step 4": "3. From the following cities, which two have the highest average annual rainfall: Bangkok (1,500 mm), Jakarta (1,800 mm), Manila (2,000 mm), Kuala Lumpur (2,400 mm), Singapore (2,300 mm), Ho Chi Minh City (1,900 mm)?"
}}
---
input question: {t_query}
Past QA information: 
"""


########################### Answer generation prompt ###########################

EVIDENCE_ANSWER_SYS_PROMPT = """
You are a helpful question-answering assistant. Your task is to answer a complex question provided by the user. You may generate a brief explanation before presenting the final answer if necessary.
Even if your information is not up to date, you must answer the question based on the knowledge you have.
The response may consist of multiple sentences, but each sentence must be concise and clear. Since the answer will be passed to another agent for follow-up tasks, ensure that the response preserves sufficient context while remaining concise and clear.

Your response must strictly follow the format below:
Answers: {ANSWERS}

You must end your response immediately after the final answer. Do not include any additional text beyond the specified format. Even if your information is not up to date, you must still provide an answer based on your knowledge.
"""

VERTICAL_ANSWER_SYS_PROMPT = """
You are a helpful parent-node answer generator.
Your task is to integrate child sub-question answers into a single parent answer without losing any meaningful information.

The response may consist of multiple sentences, but each sentence must be clear. Since the answer will be passed to another agent for follow-up tasks, ensure that it preserves sufficient context while remaining clear.
If the child QA pairs contain relevant details, include them in the final answer. Do not omit meaningful information.
Provide a concise yet complete response.
Unless the question explicitly requires it, do not use bullet points or numbered lists.

Your response must strictly follow the format below:
Answers: {ANSWERS}
"""

FINAL_ANSWER_SYS_PROMPT = {
"monaco": """
You are a helpful question answering assistant. Your task is to answer the question with high precision.
- List **only** answers that **directly and clearly** satisfy what the question asks. Do **not** include uncertain, tangential, or "possibly relevant" items.
- Prefer **fewer, correct** answers over a long list that mixes correct and incorrect items. When in doubt, omit the item.
- Your response must use the following format:
Answers: {ANSWERS}
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

LATERAL_ANSWER_PROMPT = """
Above are related question–answer pairs generated along the path leading to the current question.
Some of these QA pairs may contain information relevant to answering the current question.
In case that the information in the qa pair is relevant, you may use it to answer the question.
If a QA pair is irrelevant to current question, feel free to ignore it when answering.
"""

VERTICAL_ANSWER_PROMPT = """
Above are multiple sub question–answer (QA) pairs related to the current question.
Each sub-question represents a more specific formulation of the current question, and its corresponding answer contains information that may contribute to answering the current question.

Using the provided sub QA pairs, generate an answer to the current question that:
	- preserves all relevant contextual and answer information, and
	- incorporates as much informative content as possible from the sub QA pairs.

In other words, synthesize the information contained in the sub QA pairs to produce the most complete and comprehensive answer to the current question.
"""

FINAL_ANSWER_PROMPT = {
"monaco": """
Above are multiple sub question–answer (QA) pairs related to the current question.
Each sub-question represents a more specific formulation of the current question, and its corresponding answer contains information that may contribute to answering the current question.

Using the provided sub QA pairs, generate an answer to the current question that:
    - is **concise and direct**: state the answer without unnecessary explanation, padding, or repetition.
    - preserves relevant context so that each part of the answer is clearly tied to what it refers to (no decontextualized fragments or bare lists that omit what each item describes).
    - includes only what is needed to answer the question; omit filler and redundant phrasing.

Do not produce long explanations or prose when a short, clear answer suffices. Prefer brevity while remaining complete and unambiguous.
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

# Prompt B.3
Q_REWRITE_SINGLE = """
- You are a model that rewrites users' questions into search engine queries.
- Rewrite the question into one optimal search query.
- Examples:
    Please recommend an electric car in a similar price range to the BMW i5.
    Query: recommendation of BMW i5 price range electric car.
    Please tell me the Samsung stock price.
    Query: Samsung stock price

Question: {question}
Query: """

Q_REWRITE_MULTI = """
- You are a model that rewrites users' questions into search engine queries.
- Rewrite the question into one optimal search query.
- Examples:
    Please recommend an electric car in a similar price range to the BMW i5.
    Query: recommendation of BMW i5 price range electric car.
    Please tell me the Samsung stock price.
    Query: Samsung stock price
- Use the question numbers as JSON keys (e.g., '1', '2', '3'), extracted from labels like '1. question', '2. question'. Each value should contain only the rewritten search query corresponding to its question.

Question: 
{questions}

Output format:
{{
    "1": "query",
    "2": "query",
    ...
}}

Query: """



EXTERNAL_ANSWER_PROMPT = """
Question: {question}
Search Query: {query}
Documents: {documents}

*** Above are provided with multiple excerpts of paragraphs and tables, each of document has a title, followed by the
actual content. Some of these documents might contain helpful information to answering the question. In case that the
information in the document is relevant, you may use it to solve the question. If a document is irrelevant feel free to ignore
it when answering.

Answer: """



########################### Evidence-guided sub-question clustering prompt ###########################

EVIDENCE_CLUSTER_PROMPT = """
Each question includes an explicit list of associated documents. Use only the documents in that list for the question.

Rules:
- Answer each question independently.
- For each question, use only the documents specified in its list and no others.
- Use the question numbers as JSON keys (e.g., '1', '2', '3'), extracted from labels like 'Question 1', 'Question 2'. Each value should contain only the answer to its corresponding question.

---

{qd_pairs}

{documents}

---

*** Above are provided with multiple excerpts of paragraphs and tables, each of document has a title, followed by the
actual content. Some of these documents might contain helpful information to answering the question. In case that the
information in the document is relevant, you may use it to solve the question. If a document is irrelevant feel free to ignore
it when answering.

Output format:
{{
    "1": "answer",
    "2": "answer",
    ...
}}

Answer: """


def build_evidence_cluster_prompt(children, retrieved_list, member_indices):
    doc_id_map = {}
    ordered_docs = []
    next_doc_num = 1

    for idx in member_indices:
        docs = retrieved_list[idx] or []
        seen_in_question = set()

        for doc in docs:
            title = str(doc.get("title", "")).strip()
            content = str(doc.get("content", "")).strip()
            key = (title, content)

            if key in seen_in_question:
                continue
            seen_in_question.add(key)

            if key not in doc_id_map:
                doc_id = f"D{next_doc_num}"
                doc_id_map[key] = doc_id
                ordered_docs.append((doc_id, title, content))
                next_doc_num += 1

    qd_lines = []
    for idx in member_indices:
        child = children[idx]
        question = child.question
        query = child.query

        if "." in question:
            _, q_text = question.split(".", 1)
            q_text = q_text.strip()
            q_num = str(idx+1)
        else:
            q_text = question.strip()
            q_num = str(idx+1)

        docs = retrieved_list[idx] or []
        seen_in_question = set()
        doc_ids = []

        for doc in docs:
            title = str(doc.get("title", "")).strip()
            content = str(doc.get("content", "")).strip()
            key = (title, content)

            if key in seen_in_question:
                continue
            seen_in_question.add(key)

            doc_ids.append(doc_id_map[key])

        qd_lines.append(f"[Question {q_num}]")
        qd_lines.append(f"Documents: [{', '.join(doc_ids)}]")
        qd_lines.append(f"Question: {q_text}")
        qd_lines.append(f"Search Query: {query}")
        qd_lines.append("")

    qd_pairs = "\n".join(qd_lines).rstrip()

    doc_lines = []
    for doc_id, title, content in ordered_docs:
        doc_lines.append(
f"""{doc_id}. title: {title}
content: {content}
""")

    documents = "\n".join(doc_lines).rstrip()
    return EVIDENCE_CLUSTER_PROMPT.format(qd_pairs=qd_pairs, documents=documents)
