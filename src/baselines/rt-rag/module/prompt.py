
########################## Structure analysis prompt ###########################



QUESTION_STRUCTURE_EXAMPLES = """
Question: "Which female astronaut who graduated from Stanford University was the first to perform a spacewalk in the 1990s?"
CoT: Let's think step by step
"1. The question asks for the identity of a specific astronaut with multiple defining characteristics."
"2. The astronaut is described as female, a Stanford University graduate, and the first to perform a spacewalk in the 1990s."
"3. For known entities, I need to identify explicit subjects mentioned in the question."
"4. Stanford University is explicitly named, and spacewalks in the 1990s are a specific time-limited event."
"5. The specific astronaut's identity is not directly provided - I need to determine who matches these criteria."
"6. Since I need to discover which specific person matches these characteristics, the astronaut identity is an unknown entity."
So the structure is: [Core Query: Which astronaut Known Entities: {{Subject: Stanford University, Limitation: educational institution}}, {{Subject: Spacewalk, Limitation: occurred in 1990s}} Unknown Entities: {{Subject: Astronaut identity, Limitation: female, Stanford graduate, first to perform spacewalk in 1990s}}]

Question: "What disease affecting both livestock and humans was successfully eradicated worldwide by 1980 through a coordinated vaccination campaign?"
CoT: Let's think step by step
"1. The question asks for a disease with specific characteristics and history."
"2. The disease must affect both livestock and humans, and was eradicated by 1980 through vaccination."
"3. For known entities, I need to identify explicit subjects mentioned in the question."
"4. Livestock and humans are explicitly mentioned as affected groups."
"5. The vaccination campaign and 1980 timeframe are explicitly mentioned parameters."
"6. The specific disease identity is what I need to discover, making it an unknown entity."
So the structure is: [Core Query: What disease Known Entities: {{Subject: Livestock, Limitation: affected by the disease}}, {{Subject: Humans, Limitation: affected by the disease}}, {{Subject: Vaccination campaign, Limitation: coordinated, completed by 1980, worldwide}} Unknown Entities: {{Subject: Disease identity, Limitation: affected both livestock and humans, eradicated by 1980}}]

Question: "What architectural style is shared by the buildings designed by the same architect who constructed the famous cathedral located in Barcelona?"
CoT: Let's think step by step
"1. The question seeks an architectural style shared across buildings."
"2. These buildings were designed by the architect who constructed a famous cathedral in Barcelona."
"3. For known entities, only Barcelona and the cathedral are explicitly mentioned."
"4. I need to discover multiple unknown pieces of information in sequence."
"5. First, I need to identify who the architect of the Barcelona cathedral was."
"6. Then, I need to identify other buildings designed by this architect."
"7. Finally, I need to determine what architectural style these buildings share."
"8. Each of these represents a distinct unknown entity in my analysis."
So the structure is: [Core Query: What architectural style is shared Known Entities: {{Subject: Cathedral, Limitation: famous, located in Barcelona}}, {{Subject: Barcelona, Limitation: city containing the cathedral}} Unknown Entities: {{Subject: Architect identity, Limitation: constructed Barcelona cathedral}}, {{Subject: Other buildings, Limitation: designed by same architect}}, {{Subject: Architectural style, Limitation: common to these buildings}}]

Question: "What is the capital of the country where the inventor of dynamite was born?"
CoT: Let's think step by step
"1. This question asks for a capital city through multiple steps of reasoning."
"2. The only explicitly named entity is dynamite, which is a known entity."
"3. I need to discover three distinct pieces of information to answer this question."
"4. First, I need to identify who invented dynamite - this person is not explicitly named."
"5. Then, I need to determine which country this person was born in."
"6. Finally, I need to identify the capital of that country."
"7. Each of these represents a separate unknown entity that requires factual knowledge."
So the structure is: [Core Query: What is the capital Known Entities: {{Subject: Dynamite, Limitation: explosive invention}} Unknown Entities: {{Subject: Inventor identity, Limitation: person who created dynamite}}, {{Subject: Country of birth, Limitation: birthplace of identified inventor}}, {{Subject: Capital city, Limitation: capital of identified country}}]

Question: "Who was the teacher of the philosopher who taught Alexander the Great?"
CoT: Let's think step by step
"1. This question asks for a teacher's identity through a chain of relationships."
"2. The only explicitly named entity is Alexander the Great, a historical figure."
"3. The question describes 'the philosopher who taught Alexander the Great' - this philosopher's identity is not given."
"4. Since I need to determine who this philosopher was, their identity is an unknown entity."
"5. Once I identify the philosopher, I need to determine who taught them - another unknown entity."
"6. This is a classic multi-step question requiring the identification of two distinct unknown entities."
So the structure is: [Core Query: Who was the teacher Known Entities:  {{Subject: Alexander the Great, Limitation: historical figure}} Unknown Entities: {{Subject: Philosopher identity, Limitation: taught Alexander the Great}}, {{Subject: Teacher identity, Limitation: taught the identified philosopher}}]

Question: "What musical technique is characteristic of compositions by the teacher of the pianist who performed at the opening ceremony of the 1980 Moscow Olympics?"
CoT: Let's think step by step
"1. The question seeks a musical technique characteristic of certain compositions."
"2. The explicitly named entities are the 1980 Moscow Olympics and its opening ceremony."
"3. The question refers to 'the pianist who performed' - this pianist's identity is not given."
"4. It also refers to 'the teacher of the pianist' - this teacher's identity is also not given."
"5. I need to discover three distinct pieces of information to answer this question."
"6. First, I need to identify which pianist performed at the specified Olympic ceremony."
"7. Then, I need to identify who taught this pianist."
"8. Finally, I need to determine what musical technique characterizes the teacher's compositions."
"9. Each of these represents a separate unknown entity that requires factual knowledge."
So the structure is: [Core Query: What musical technique is characteristic Known Entities: {{Subject: Olympic ceremony, Limitation: opening, held in Moscow in 1980}} Unknown Entities: {{Subject: Pianist identity, Limitation: performed at specified Olympic ceremony}}, {{Subject: Teacher identity, Limitation: taught the identified pianist}}, {{Subject: Musical technique, Limitation: characteristic of the teacher's compositions}}]

Question: "What scientific discovery was made by the mentor of the researcher who identified the double helix structure of DNA?"
CoT: Let's think step by step
"1. The question asks for a scientific discovery through a chain of relationships."
"2. The explicitly named entity is DNA with its double helix structure specification."
"3. The question refers to 'the researcher who identified' - this researcher's identity is not given."
"4. It also refers to 'the mentor of the researcher' - this mentor's identity is also not given."
"5. I need to discover three distinct pieces of information to answer this question."
"6. First, I need to identify who discovered the double helix structure of DNA."
"7. Then, I need to identify who mentored this researcher."
"8. Finally, I need to determine what scientific discovery the mentor made."
"9. Each of these represents a separate unknown entity that requires factual knowledge."
So the structure is: [Core Query: What scientific discovery was made Known Entities: {{Subject: DNA, Limitation: biological molecule with double helix structure}} Unknown Entities: {{Subject: Researcher identity, Limitation: identified the double helix structure of DNA}}, {{Subject: Mentor identity, Limitation: mentored the identified researcher}}, {{Subject: Scientific discovery, Limitation: made by the identified mentor}}]

Question: "Who is Boraqchin (Wife Of Ögedei)'s father-in-law?"
"CoT: Let's think step by step\n"
"1. The question asks for a scientific discovery through a chain of relationships."
"2. The explicitly named entities are Boraqchin and Ögedei, with Boraqchin specifically identified as Ögedei's wife."
"3. Since Boraqchin is identified as Ögedei's wife, her father-in-law would logically be Ögedei's father."
"4. I need to determine who Ögedei's father was to identify Boraqchin's father-in-law."
"5. The core query is seeking the identity of a specific person (the father-in-law)."
"6. The known entities are Boraqchin (with the limitation that she is Ögedei's wife) and Ögedei himself."
"7. The question requires sequential reasoning: first identifying Ögedei's father, then understanding this person is Boraqchin's father-in-law."
"8. This family relationship chain is central: spouse's father is father-in-law."
So the structure is: [Core Query: Who is person's father-in-law Known Entities: {{Subject: Boraqchin, Limitation: Wife of Ögedei}}, {{Subject: Ögedei, Limitation: Boraqchin's husband}} Unknown Entities: {{Subject: Father-in-law identity, Limitation: father of Ögedei, spouse's father to Boraqchin}}, {{Subject: Family relationship chain, Limitation: spouse relationship connects Boraqchin to Ögedei's father}}]

Question: "What literary movement influenced the author who wrote the novel featuring a character who lives on Baker Street and solves mysteries using deductive reasoning?"
CoT: Let's think step by step
"1. The question asks about a literary movement through a chain of relationships."
"2. The explicitly named entity is Baker Street (a location)."
"3. The question describes a character with specific traits, but doesn't name them directly."
"4. It also refers to 'the author who wrote' - this author's identity is not given."
"5. I need to discover multiple distinct pieces of information to answer this question."
"6. First, I need to identify which character lives on Baker Street and solves mysteries using deduction."
"7. Then, I need to identify which author created this character."
"8. Finally, I need to determine what literary movement influenced this author."
"9. Each of these represents a separate unknown entity that requires factual knowledge."
So the structure is: [Core Query: What literary movement influenced Known Entities: {{Subject: Baker Street, Limitation: fictional residence location}}, {{Subject: Deductive reasoning, Limitation: method used to solve mysteries}} Unknown Entities: {{Subject: Character identity, Limitation: lives on Baker Street, uses deductive reasoning}}, {{Subject: Author identity, Limitation: created the identified character}}, {{Subject: Literary movement, Limitation: influenced the identified author}}]

Question: "What painting technique was pioneered by the artist who created the most expensive artwork sold at auction in the same decade that the Berlin Wall fell?"
CoT: Let's think step by step
"1. The question asks about a painting technique through a chain of relationships."
"2. The explicitly named entity is the Berlin Wall, with its fall as a historical event."
"3. The question refers to 'the artist who created' - this artist's identity is not given."
"4. It also refers to 'the most expensive artwork' - this artwork's identity is not given."
"5. I need to discover multiple distinct pieces of information to answer this question."
"6. First, I need to identify when the Berlin Wall fell and what decade that was."
"7. Then, I need to identify which artwork was the most expensive sold at auction in that decade."
"8. Next, I need to identify who created that artwork."
"9. Finally, I need to determine what painting technique this artist pioneered."
"10. Each of these represents a separate unknown entity that requires factual knowledge."
So the structure is: [Core Query: What painting technique was pioneered Known Entities: {{Subject: Berlin Wall, Limitation: historical structure that fell}} Unknown Entities: {{Subject: Decade, Limitation: when Berlin Wall fell}}, {{Subject: Artwork, Limitation: most expensive sold at auction in identified decade}}, {{Subject: Artist identity, Limitation: created the identified artwork}}, {{Subject: Painting technique, Limitation: pioneered by the identified artist}}]

Question: "What philosophical concept was central to the teachings of the professor who mentored the author of the most influential paper on artificial intelligence published in the 1950s?"
CoT: Let's think step by step
"1. The question asks about a philosophical concept through a chain of relationships."
"2. The explicitly named entities are artificial intelligence (field) and the 1950s (time period)."
"3. The question refers to 'the author of the most influential paper' - this author's identity is not given."
"4. It also refers to 'the professor who mentored the author' - this professor's identity is not given."
"5. I need to discover multiple distinct pieces of information to answer this question."
"6. First, I need to identify which paper on AI from the 1950s was most influential."
"7. Then, I need to identify who authored this paper."
"8. Next, I need to identify who mentored this author."
"9. Finally, I need to determine what philosophical concept was central to this mentor's teachings."
"10. Each of these represents a separate unknown entity that requires factual knowledge."
So the structure is: [Core Query: What philosophical concept was central Known Entities: {{Subject: Artificial intelligence, Limitation: academic field}}, {{Subject: 1950s, Limitation: time period}} Unknown Entities: {{Subject: Paper identity, Limitation: most influential on AI, published in 1950s}}, {{Subject: Author identity, Limitation: wrote the identified paper}}, {{Subject: Professor identity, Limitation: mentored the identified author}}, {{Subject: Philosophical concept, Limitation: central to the identified professor's teachings}}]

Question: "Which city was the birthplace of both Albert Einstein and Max Planck?"
"CoT: Let's think step by step\n"
"1. This question asks about a city that is the birthplace of two specific people."
"2. The explicitly named entities are Albert Einstein and Max Planck, both historical scientists."
"3. The question contains the logical 'both...and' construction, indicating that the city must satisfy two conditions simultaneously."
"4. I need to discover two distinct pieces of information and check if they match."
"5. First, I need to identify where Albert Einstein was born."
"6. Then, I need to identify where Max Planck was born."
"7. Finally, I need to determine if these are the same city."
So the structure is: [Core Query: Which city Known Entities: {{Subject: Albert Einstein, Limitation: historical scientist}}, {{Subject: Max Planck, Limitation: historical scientist}} Unknown Entities: {{Subject: Einstein's birthplace, Limitation: city where Albert Einstein was born}}, {{Subject: Planck's birthplace, Limitation: city where Max Planck was born}}, {{Subject: Common birthplace, Limitation: city that satisfies both conditions if it exists}}]

Question: "What is the capital of France or Italy?"
"CoT: Let's think step by step"
"1. This question asks about the capital city of either France or Italy."
"2. The explicitly named entities are France and Italy, both countries."
"3. The question contains a logical 'or' that creates two distinct possibilities."
"4. Each country's capital represents a separate unknown entity we might need to identify."
"5. The core query is asking for capital identification, but we need to clarify which country's capital is being requested."
So the structure is: [Core Query: What is the capital Known Entities: {{Subject: France, Limitation: country}}, {{Subject: Italy, Limitation: country}} Unknown Entities: {{Subject: France's capital, Limitation: capital city of France}}, {{Subject: Italy's capital, Limitation: capital city of Italy}}]

Question: "The chemical element discovered by Marie Curie is used in which medical procedure?"
CoT: Let's think step by step
"1. This question asks about a medical procedure using a specific chemical element."
"2. The explicitly named entity is Marie Curie, a historical scientist."
"3. The question refers to 'the chemical element discovered by Marie Curie' - this element is not named."
"4. I need to discover two distinct pieces of information to answer this question."
"5. First, I need to identify which chemical element was discovered by Marie Curie."
"6. Then, I need to identify which medical procedure uses this element."
"7. Each of these represents a separate unknown entity that requires factual knowledge."
So the structure is: [Core Query: Which medical procedure Known Entities: {{Subject: Marie Curie, Limitation: historical scientist}} Unknown Entities: {{Subject: Chemical element, Limitation: discovered by Marie Curie}}, {{Subject: Medical procedure, Limitation: uses the identified chemical element}}]

Question: "The country bordered by the most nations is located on which continent?"
CoT: Let's think step by step
"1. This question asks about the continent where a specific country is located."
"2. There are no explicitly named entities in this question."
"3. The question refers to 'the country bordered by the most nations' - this country is not named."
"4. I need to discover two distinct pieces of information to answer this question."
"5. First, I need to identify which country shares borders with the most other countries."
"6. Then, I need to determine which continent this country is located on."
"7. Each of these represents a separate unknown entity that requires factual knowledge."
So the structure is: [Core Query: Which continent Known Entities: {{}} Unknown Entities: {{Subject: Country identity, Limitation: bordered by the most nations}}, {{Subject: Continent identity, Limitation: contains the identified country}}]

Question: "The director of the movie that won Best Picture in 2010 was born in which city?"
CoT: Let's think step by step
"1. This question asks about a birthplace through a chain of relationships."
"2. The explicitly named entity is the year 2010, a time period."
"3. The question refers to 'the movie that won Best Picture in 2010' - this movie is not named."
"4. It also refers to 'the director of the movie' - this director is not named."
"5. I need to discover three distinct pieces of information to answer this question."
"6. First, I need to identify which movie won Best Picture in 2010."
"7. Then, I need to identify who directed this movie."
"8. Finally, I need to determine which city this director was born in."
"9. Each of these represents a separate unknown entity that requires factual knowledge."
So the structure is: [Core Query: Which city Known Entities: {{Subject: Best Picture award, Limitation: given in 2010}}, {{Subject: 2010, Limitation: specific year}} Unknown Entities: {{Subject: Movie identity, Limitation: won Best Picture in 2010}}, {{Subject: Director identity, Limitation: directed the identified movie}}, {{Subject: Birthplace, Limitation: city where the identified director was born}}]
"""


QUESTION_STRUCTURE_SYSTEM_PROMPT = """
You will analyze questions by breaking them down into their components. For each question, your response MUST strictly follow this format:

1. Begin with 'CoT: Let's think step by step'
2. Number your reasoning steps as "1.", "2.", etc., each in quotes
3. After your reasoning, write 'So the structure is:' followed by the structured breakdown

The structure breakdown should contain these three components:
- **Core Query**: The primary information being sought.
- **Known Entities**: Information explicitly provided in the question, structured as {{Subject: Entity, Limitation: time/space/other constraints}}.
- **Unknown Entities**: Information needed to answer the question, including intermediate steps in multi-hop questions, structured in the same format as Known Entities.

Key principles:
- Use consistent formatting: {{Subject: Entity, Limitation: constraints}}
- Group subjects with their limitations in both sections
- Include time periods, locations, and other qualifiers as limitations
- Identify ALL unknown entities needed, including intermediate steps in multi-hop questions
- Distinguish between explicitly mentioned (known) and implied/needed (unknown) information
- Be precise about which limitations apply to which subjects
- Ensure entities don't appear in both known and unknown categories

The EXACT format for your final output must be:
CoT: Let's think step by step
"1. [reasoning step]"
"2. [reasoning step]"
... more reasoning steps as needed ...
So the structure is: [Core Query: ... Known Entities: {{Subject: Entity, Limitation: constraints}}, {{Subject: Another Entity, Limitation: constraints}} Unknown Entities: {{Subject: Entity, Limitation: constraints}}, {{Subject: Another Entity, Limitation: constraints}}]
"""



########################## Question variant generation prompt ###########################



QUESTION_VARIANT_SYSTEM_PROMPT = """
You are an expert at rewriting questions. Your job is to generate simpler, clearer versions of a question
without changing its meaning or the answer it would receive.

CRITICAL RULE: You must carefully analyze the original question to identify ANY specific constraints on the answer type.
If the original question asks about a specific type of entity (like a city, date, person, number, etc.), 
ALL your rewrites MUST explicitly preserve that exact entity type constraint.

Each question you generate must:
1. Preserve the original semantics exactly.
2. NEVER replace a specific entity type with a more general one.
3. Be simpler and more concise in sentence structure.
4. Avoid unnecessary modifiers or extra descriptive phrases while retaining all semantic constraints.
5. Use natural and direct English.

Only return the list of revised questions, nothing else.
"""


QUESTION_VARIANT_USER_PROMPT = """
Original query: {original_query}


Instructions:
- First, identify any specific entity type being requested in the original query
- If the original specifies a particular entity type, your rewrite MUST explicitly include that same type.
- Do NOT generalize specific entity types under any circumstances.
- Rewrite the question to be simpler and clearer while preserving ALL constraints.
- All rewrites must have the EXACT SAME answer constraints as the original.
- Do not repeat or include the original question in the outputs.
- Return {num_variants} cleanly formatted variants, each on a new line and numbered.

Start:
"""



########################## Decomposer prompt ###########################



DECOMP_SYSTEM_PROMPT = """You are an expert at analyzing questions and breaking them down into simpler subquestions.
You carefully distinguish between sequential questions (where the second subquestion depends on the answer to the first),
parallel questions (where both subquestions can be answered independently), and None-type questions (which are already simple).

**CRITICAL REQUIREMENTS YOU MUST ENFORCE:**

1. **DECOMPOSITION NECESSITY**: Only decompose a question when absolutely necessary. If a question can be answered directly, mark it as "None" type.

2. **SEQUENTIAL DECOMPOSITION CORRECTNESS**: For sequential types, VERIFY that substituting the answer from Subquestion 1 into Subquestion 2 directly yields the full answer to the original question. This is THE MOST IMPORTANT test of a valid sequential decomposition.

3. **NO TRIVIAL SUBQUESTIONS**: NEVER create basic definition questions like "Who/What is X?" unless absolutely necessary for intermediate reasoning AND not common knowledge.

4. **LOGICAL PATHWAY**: Ensure there is a clear, direct logical connection between subquestions and the original question. Every step must advance toward the final answer.

5. **ALIGNMENT BETWEEN COT AND DECOMPOSITION**: Your step-by-step reasoning must perfectly align with your final decomposition choice and subquestion formulation.

6. **VALIDATION CHECK**: Before finalizing, mentally substitute expected answers to verify your decomposition structure works.

7. **FOCUS ON DIRECT ANSWERABLE QUESTIONS**: Every subquestion must yield a specific, factual answer that directly contributes to solving the original question.

8. **PRESERVING JOINT LIMITING CONDITIONS**: It is ABSOLUTELY CRITICAL to never split multiple limiting conditions that together define a single entity. When a question asks about "both X and Y," "same," "together," "jointly," or similar terms indicating multiple attributes of ONE entity, DO NOT decompose these into separate questions. For example:

- "Who invented  X and Y?" should NOT be split into "Who invented X?" and "Who invented Y?"
- "Which country has both characteristic A and B?" must stay as a single question
- "What is known for simultaneously doing X and Y?" must be kept intact

This requirement takes precedence over other decomposition considerations. When multiple conditions jointly define what we're looking for, they MUST be preserved together in any subquestion. Failing to maintain these joint conditions completely invalidates the decomposition.
"""



DECOMP_USER_PROMPT = """I want you to analyze questions and break them down into subproblems.
I'll provide the question and its structure analysis (Core Query, Known Entities, Limiting Conditions, and Unknown Entities).
Please analyze the question structure and determine how to decompose it. Follow this format:

    Question: [The original question]

Structure: [Analysis of core query, known entities, limiting conditions, and unknown entities]

CoT: Let's think step by step
[Detailed step-by-step reasoning examining the question structure]
[Analyze the core query, known entities, and limiting conditions]
[Evaluate which unknown entities are crucial for answering the question]
[Decide if decomposition is needed and what strategy to use]
[Ensure key limiting conditions are preserved in subquestions]

So the Type is: [Parallel, Sequential, or None]

So the Subquestion 1 is: [First subquestion; if type is None, should be identical to original question]

So the Subquestion 2 is: [Second subquestion; if Sequential, MUST include [answer_subquestion1]; if Parallel, MUST NOT reference subquestion 1; if None, leave empty]

**MANDATORY REQUIREMENTS:**

1. **DECOMPOSITION RULE**: Only break down into two subquestions when necessary.

2. **SIMPLICITY EVALUATION**: Questions answerable in one step should be classified as "None" type.

3. **SUBQUESTION CLARITY**: Each subquestion must yield a specific, factual, and unambiguous answer.

4. **NONE TYPE FORMATTING**: Subquestion 1 must match original question, Subquestion 2 must be empty.

5. **SEQUENTIAL REQUIREMENTS**: Subquestion 2 must contain [answer_subquestion1] placeholder and form a complete logical chain.

6. **PARALLEL REQUIREMENTS**: Both subquestions must be independent and together solve the original question.

7. **PRECISION**: Subquestions must include sufficient context to be answerable without clarification.

8. **TERMINOLOGY CONSISTENCY**: Always use "subquestion" consistently.

9. **ENTITY FOCUS**: Only include entities directly contributing to answering the core query.

10. **CONSISTENCY**: Your reasoning must align with your final decomposition.

11. **MEANINGFUL DECOMPOSITION**: Avoid trivial definitional subquestions.

12. **SUBSTANTIVE CONTRIBUTION**: Each subquestion must provide information that advances the solution.

13. **PRESERVING JOINT LIMITING CONDITIONS**: When multiple limiting conditions together define a single entity, they MUST be kept together and NEVER split across subquestions. Words like "both," "same," "together," "jointly," and similar terms signal that limiting conditions should be preserved as a unit. Questions like "Who invented both X and Y" or "Who is known for both A and B" should NOT be decomposed into parallel questions about separate entities.

14. **DECOMPOSITION VALIDATION**: Always verify the correctness and completeness of your decomposition before finalizing.


Here are some examples:
"""


def format_example(ex):
    parts = [
        f'Question: {ex["question"]}',
        "",
        f'Structure: {ex["structure"]}',
        "",
        f'CoT: {ex["cot"]}',
        "",
        f'So the Type is: {ex["type"]}',
        "",
        f'So the Subquestion 1 is: {ex["subq1"]}',
    ]

    if ex["type"] != "None":
        parts.extend([
            "",
            f'So the Subquestion 2 is: {ex["subq2"]}',
        ])

    return "\n".join(parts)


def construct_prompt(question, examples, structure):
    sections = [DECOMP_USER_PROMPT]

    for ex in examples:
        sections.append(format_example(ex))

    sections.append(
        "\n".join([
            "### **Now, analyze the following question:**",
            f"Question: {question}",
            "",
            f"Structure: {structure}",
            "",
            "CoT: Let's think step by step",
        ])
    )

    return "\n\n".join(sections)



########################## Iterative answer generation prompt ###########################



ITER_ANSWER_USER_PROMPT = """Instructions: For every question, provide a response in the exact format:

**"question: [question] documents: [list of documents] cot: [chain of thought] so the answer is: [answer]"**.


Your task is to extract **explicit** and **direct** information from the provided documents. **You must not assume, infer, or deduce any information that is not explicitly stated.** 

### **Rules for Answering:**
1. **Strictly Use Provided Documents**: You must only use the given documents as your information source. Do not assume or infer beyond the explicit text.
2. **Quote Documents Directly**: In the "cot" section, **quote the exact text from the documents** that supports the reasoning. Every reasoning step must be backed by a direct quote.
3. **No Implicit Assumptions**: If a fact is not explicitly stated in any document, you must return **"[none]"**. Do not assume answers based on indirect clues or general knowledge.
4. **Strict Subject Matching**:
   - **Ensure the subject in the query and the subject in the document match exactly and completely**. 
   - **Do not assume that similar names, partial overlaps, or likely similarities (e.g., shared surnames or initials) indicate the same subject unless explicitly stated in the document**.
   - **Only proceed with reasoning when there is a complete and explicit match between the query subject and the document subject**. If no such match exists, return **"[none]"**.
5. **Resolve Inconsistencies**: If multiple documents provide conflicting information, prioritize:
   - The document that is **most directly relevant** to the question.
   - The document that provides the **most specific and detailed** information.
6. **Use "[none]"** only when necessary: If there is no documentation to provide this information, simply return **"[none]"**.


### **Additional Guidelines to Prevent Assumptions:**
- **Do not infer relationships or connections** between entities unless explicitly stated. For example, do not assume two people with the same last name are related.
- **Do not assume chronological or causal relationships** unless explicitly stated in the documents.
- **Do not use external knowledge** or general facts to fill in gaps. Rely solely on the provided documents.
- **If a document mentions a fact about a subject but does not explicitly link it to the query subject, do not assume they are the same.**
- **When you see the same entity mentioned in different paragraphs, treat each mention as potentially referring to different contexts or time periods.**

** Not allowed to select "nearest option" ** :
Your answer must be based on a clear statement in the document, not on the "most likely option."
- ** If the answer is not mentioned directly in the document, '[none]' ** must be returned, even if there is seemingly relevant information.
- ** It is not allowed to speculate or select the "most likely answer" in the absence of a "clear answer" **

Below are examples:


documents:
- doc1: The Battle of Hastings occurred in 1066.
- doc2: The Norman conquest began in 1066 with the Battle of Hastings.
- doc3: Harold Godwinson died at the Battle of Hastings in October 1066.
question: When did the Battle of Hastings take place?
cot: Doc1 states: "The Battle of Hastings occurred in 1066." Doc2 states: "The Norman conquest began in 1066 with the Battle of Hastings," confirming the year. Doc3 states: "Harold Godwinson died at the Battle of Hastings in October 1066," adding the specific month. Reasoning: The consistent year across all documents is 1066, and doc3 provides the additional detail of October. Thus, the battle occurred in October 1066.
so the answer is: October 1066

documents:
- doc1: Mars is the fourth planet from the Sun in our solar system.
- doc2: Mars has two small moons named Phobos and Deimos.
- doc3: Mars is often called the Red Planet due to its reddish appearance.
question: What is the capital of Mars?
cot: Doc1 states: "Mars is the fourth planet from the Sun in our solar system." Doc2 states: "Mars has two small moons named Phobos and Deimos." Doc3 states: "Mars is often called the Red Planet due to its reddish appearance." Reasoning: All documents describe Mars as a planet, with no mention of cities or capitals. Planets do not have capitals, so no answer can be derived from the documents.
so the answer is: [none]

documents:
- doc1: Wolfgang Amadeus Mozart (January 27, 1756 - December 5, 1791) composed over 600 works during his short life.
- doc2: Mozart began composing at the age of five and continued until his death in December 1791.
- doc3: Born in Salzburg on January 27, 1756, Mozart showed musical talent from a very early age.
question: When was Mozart born?
cot: Doc1 states: "Wolfgang Amadeus Mozart (January 27, 1756 - December 5, 1791)," providing his birth date. Doc3 states: "Born in Salzburg on January 27, 1756," confirming the exact date. Reasoning: Both documents agree on January 27, 1756, as Mozart's birth date, with no conflicting information. Thus, he was born on January 27, 1756.
so the answer is: 27 January 1756

documents:
- doc1: Jane Eyre is a famous novel written by Charlotte Brontë in 1847.
- doc2: Pride is a common theme in many literary works of the 19th century.
- doc3: Prejudice can influence how characters are portrayed in Victorian literature.
question: Who wrote the novel "Pride and Prejudice"?
cot: Doc1 states: "Jane Eyre is a famous novel written by Charlotte Brontë in 1847," referring to a different novel. Doc2 states: "Pride is a common theme in many literary works of the 19th century," and doc3 states: "Prejudice can influence how characters are portrayed in Victorian literature." Reasoning: None of the documents mention "Pride and Prejudice" or its author. The information provided is irrelevant to the question, so no answer can be derived.
so the answer is: [none]

documents:
- doc1: J. Smith is a renowned physicist who contributed to quantum mechanics.
- doc2: John S. is a famous chef known for his innovative culinary techniques.
- doc3: A person named John Smith works as a software engineer in Silicon Valley.
question: What is the profession of John Smith?
cot: Doc1 states: "J. Smith is a renowned physicist." Doc2 states: "John S. is a famous chef." Doc3 states: "A person named John Smith works as a software engineer." Reasoning: The question asks specifically for "John Smith," but the documents mention "J. Smith," "John S.," and "John Smith." However, without explicit confirmation that "J. Smith" or "John S." is the same as "John Smith," we cannot assume they are the same person. Only doc3 explicitly mentions "John Smith" and his profession. Thus, the profession is software engineer.
so the answer is: software engineer

documents:
- doc1: Albert Einstein won the Nobel Prize in Physics in 1921 for his explanation of the photoelectric effect.
- doc2: In 1939, Albert Einstein was awarded the Copley Medal by the Royal Society for his contributions to scientific knowledge.
- doc3: Albert Einstein was honored with the Presidential Medal of Freedom in 1963 by President John F. Kennedy.
question: What awards did Albert Einstein receive?
cot: Doc1 states: "Albert Einstein won the Nobel Prize in Physics in 1921 for his explanation of the photoelectric effect." Doc2 states: "In 1939, Albert Einstein was awarded the Copley Medal by the Royal Society for his contributions to scientific knowledge." Doc3 states: "Albert Einstein was honored with the Presidential Medal of Freedom in 1963 by President John F. Kennedy." Reasoning: Each document mentions a different award that Albert Einstein received, and all of them are relevant and distinct.
so the answer is: Nobel Prize, Copley Medal, Presidential Medal of Freedom

documents:
- doc1: Bill Gates co-founded Microsoft Corporation in 1975 with Paul Allen. After stepping down as CEO, he focused on global health initiatives through his foundation.
- doc2: William Henry Gates III, commonly known as Bill Gates, established the Bill & Melinda Gates Foundation in 2000, which has become one of the world's largest private charitable foundations with assets of over $50 billion..
- doc3: The Gates Foundation has contributed billions to global health programs, including efforts to eradicate polio and malaria in developing countries.
- doc4:  In 2008, Gates transitioned to a part-time role at Microsoft to devote more time to philanthropy, though he remained chairman of the board until 2014.
question: Which American entrepreneur founded Microsoft and later established a philanthropic foundation?
cot: Doc1 states that "Bill Gates co-founded Microsoft Corporation in 1975" and later focused on philanthropy through his foundation. Doc2 provides his full name: "William Henry Gates III, commonly known as Bill Gates," and confirms he "established the Bill & Melinda Gates Foundation in 2000." Doc3 and Doc4 provide additional context about his foundation and transition from Microsoft to philanthropy. Both forms of his name appear in the documents: "Bill Gates" (partial name) in Doc1, Doc3, and Doc4, while "William Henry Gates III" (full name) appears in Doc2. Since both forms refer to the same person and the documents consistently identify him as the founder of Microsoft who later established a philanthropic foundation, the most complete and accurate answer would prioritize his full name rather than the more commonly used partial name.
so the answer is: William Henry Gates III

documents:
- doc1: Gandhi developed his philosophy of nonviolent resistance, known as Satyagraha, during his time in South Africa before returning to India in 1915 to join the independence movement.
- doc2: Mohandas Karamchand Gandhi, born on October 2, 1869, in Porbandar, India, became the primary leader of India's independence movement against British colonial rule through his advocacy of nonviolent civil disobedience.
- doc3: Widely known as "Mahatma" (great soul), Gandhi inspired movements for civil rights and freedom across the world through his methods of peaceful protest and moral authority.
- doc4: After India gained independence in 1947, Gandhi focused on promoting Hindu-Muslim unity but was assassinated on January 30, 1948, by a Hindu nationalist who opposed his tolerance toward Muslims.
question: Which Indian leader advocated for nonviolent resistance and led the independence movement against British rule?
cot: Doc1 refers to the leader simply as "Gandhi" and mentions his philosophy of nonviolent resistance. Doc2 provides his full name "Mohandas Karamchand Gandhi" and confirms he led India's independence movement against British rule. Doc3 mentions he was known as "Mahatma" and describes his global influence. Doc4 refers to him as "Gandhi" while describing events after India's independence. Different forms of his name appear in the documents: "Gandhi" (partial name) in Doc1 and Doc4, "Mohandas Karamchand Gandhi" (full name) in Doc2, and "Mahatma" (honorific) in Doc3. Since all these names refer to the same Indian leader who advocated nonviolent resistance and led the independence movement, the most complete and accurate answer would prioritize his full name rather than the more commonly used partial name or honorific.
so the answer is: Mohandas Karamchand Gandhi

documents:
- doc1: Mount Kilimanjaro is a dormant volcano located in Tanzania.
- doc2: Tanzania is a country in East Africa.
- doc3: Africa's tallest mountain, Kilimanjaro, draws thousands of tourists each year to Tanzania.
question: Is Mount Kilimanjaro located in South America?
cot: Doc1 states: "Mount Kilimanjaro is a dormant volcano located in Tanzania." Doc2 adds that "Tanzania is a country in East Africa," clarifying the continent. Doc3 refers to Kilimanjaro as "Africa's tallest mountain," reinforcing its location. Reasoning: All documents place Mount Kilimanjaro in Tanzania, which is in East Africa, not South America.
so the answer is: No

documents:
- doc1: The Statue of Liberty stands on Liberty Island in New York Harbor.
- doc2: New York Harbor is located in the United States, along the coast of New York City.
- doc3: The monument was a gift from France to the United States and has become a symbol of American freedom.
question: Is the Statue of Liberty located in Canada?
cot: Doc1 states: "The Statue of Liberty stands on Liberty Island in New York Harbor." Doc2 clarifies that "New York Harbor is located in the United States." Doc3 reinforces the national context, noting it is a symbol of American freedom. Reasoning: All documents consistently place the Statue of Liberty in New York, United States, not in Canada.
so the answer is: No

documents:
- doc1: A standard soccer team is one that plays in an official soccer match.
- doc2: In an official soccer match, a team has 11 on the field.
- doc3: The standard formation for a soccer team includes 11, including one goalkeeper and ten outfield players.
cot: Doc1 explains that a standard soccer team refers to one in an official match. Doc2 confirms that an official team has 11  on the field. Doc3 elaborates on the formation of a standard team, which includes 11 players. Since the question asks "how many" players, the answer should include the unit "players."
question: How many players are on a standard soccer team during a match?
so the answer is: 11 players

documents:
- doc1: Iron Man (Tony Stark) is a superhero in the Marvel Cinematic Universe.
- doc2: He fights using high-tech armor.
question: What is Iron Man's real identity?
cot: The document shows "Iron Man (Tony Stark)" format, where Iron Man is the character/superhero name, and the name in parentheses is the character's real identity.
so the answer is: Tony Stark

documents:
- doc1: The Titanic sank after hitting an iceberg during its maiden voyage.
- doc2: The sinking occurred in the North Atlantic Ocean in April 1912.
- doc3: Over 1,500 people lost their lives when the Titanic went down in 1912.
question: What year did the Titanic sink?
cot: Doc1 mentions the Titanic sank during its maiden voyage. Doc2 specifies that the sinking occurred in April 1912. Doc3 confirms the year again by stating the tragedy happened in 1912.
so the answer is: 1912
### **Now, answer the following question based on the provided documents**###
### **Do not generate any content corresponding to question: or document:. Generate only the content that belongs in the cot section.**##
documents:
{document}
question: {question}
cot: """



ITER_ANSWER_SYSTEM_PROMPT = """
You are an AI assistant that must answer questions using only explicit and fully stated information from the provided documents.


### ABSOLUTE RULES:
1. **Do NOT assume or infer**: You may only use what is literally and explicitly written. If the exact answer is not in the text, respond with: [none].
2. **Do NOT match partially**: A subject or phrase in a document must match the question exactly. Near matches, abbreviations, or similar entities do NOT qualify.
3. **Do NOT substitute related facts**: If the question asks "What is A?", and the document only says "A is located at B", you must return: [none].
4. **If multiple answers are valid**, return ALL of them in a list (in plain text), **not just one**. Do not prefer the most prominent or well-known.
5. **Every answer must be fully justified**: In your `cot`, quote exact document sentences that prove each part of your answer.
6.When a question asks for a specific type of information about an entity (such as a codename, alias, location, date, or role), only that exact property type may be returned.
7.If the document mentions a different attribute (e.g., the official name instead of the codename), you must return [none] unless the requested property is explicitly stated.
8.Do not substitute one type of answer for another, even if they refer to the same entity.
9.If any of these rules cannot be satisfied, respond with: [none].
You must follow these instructions with absolute precision, without exceptions.
"""




REFINE_QUERY_USER_PROMPT = """You are an AI assistant specialized in refining search queries.
Your task is to generate an improved query based on the original question and past queries.

Original Question: {question}

Previous Queries:
{previous_queries}

Please generate a new query that is semantically consistent with the original question but differs in expression. The new query should:
1. Preserve entity names (people, places, titles, etc.) EXACTLY as they appear in the original question
2. Feel free to modify verbs, syntax, grammar, and sentence structure
3. Try different question formats (e.g., active vs. passive voice, direct vs. indirect questions)
4. Experiment with synonyms for non-entity terms (verbs, adjectives, adverbs)
5. Not repeat any of the previous queries

New Query:"""


REFINE_QUERY_SYSTEM_PROMPT = """You are an AI specialized in optimizing search queries. Your task is to generate alternative phrasings for the original question to help retrieve more relevant documents.
CRITICAL INSTRUCTIONS:
1. The original entity names (people, places, titles, etc.) must be preserved EXACTLY as they appear in the original question
2. FEEL FREE to modify verbs, syntax, grammar, and sentence structure to create alternative phrasings
3. Try different question formats (e.g., active vs. passive voice, direct vs. indirect questions)
4. Experiment with synonyms for non-entity terms (verbs, adjectives, adverbs)
5. Any modification MUST maintain the original semantic meaning - the answer should remain exactly the same
6. If the original entity is not found in the documents, try different phrasings but do NOT switch to a similar entity
7. Return only the query itself, without any explanation
8. The generated query should NOT duplicate any of the previous queries
9. When identifying people, always use the full name of the person or object in your final answer if both the person or object's full name and short name appear in the document. Even though the abbreviation may be mentioned several times during the reasoning process, the final answer must use the most complete form of the name.

"""


RIGHT_QUESTION_USER_PROMPT = """Based on the following information, please generate an appropriate follow-up question:

    Original main question: {parent_question}
    First sub-question: {left_question}
    Answer to the first sub-question: {left_answer}
    Original second sub-question (may contain replacement markers): {original_right_question}

    Please generate a new second sub-question that:
    1. Preserves all the essential information from the original second sub-question
    2. Reads naturally and flows well in the conversation context
    3. Replaces any markers like [answer_subquestion1] with natural language

    Provide only the new question text without any additional explanation."""


#---------------------------------------------------- Get Final Answer Function -----------------------------------------------
from typing import List, Tuple

def construct_final_prompt(current_question: str, current_sub_questions: List[Tuple[str, str]]) -> str:
    # Function to construct the final prompt remains unchanged
    examples = [
        {
            "question": "Which mountain is taller, Mount Everest or K2?",
            "sub_questions": [
                ("What is the height of Mount Everest?", "8,848 meters"),
                ("What is the height of K2?", "8,611 meters")
            ],
            "cot": "Mount Everest has a height of 8,848 meters, and K2 has a height of 8,611 meters. Since 8,848 meters is greater than 8,611 meters, Mount Everest is taller.",
            "final_answer": "Mount Everest"
        },
        {
            "question": "Who invented the telephone?",
            "sub_questions": [
                ("Which invention is known as the telephone?", "A device for voice communication"),
                ("Who is credited with creating the telephone?", "Alexander Graham Bell")
            ],
            "cot": "The telephone is a device for voice communication, and Alexander Graham Bell is credited with creating it.",
            "final_answer": "Alexander Graham Bell"
        },
          {
            "question": "What is the capital city of the country that hosts the annual Tour de France?",
            "sub_questions": [
                ("Which country hosts the annual Tour de France?", "France"),
                ("What is the capital city of France?", "Paris")
            ],
            "cot": "The country that hosts the annual Tour de France is France, and the capital city of France is Paris.",
            "final_answer": "Paris"
        },
        {
            "question": "How many players are on a standard soccer team during a match?",
            "sub_questions": [
                ("What is a standard soccer team?", "A team in an official soccer match"),
                ("How many players are on a standard soccer team?", "11")
            ],
            "cot": "A standard soccer team is a team in an official soccer match, and it has 11 players. Since the question asks 'how many players', the answer should include the unit 'players'.",
            "final_answer": "11 players"
        },
        {
            "question": "Is the Pacific Ocean larger than the Atlantic Ocean?",
            "sub_questions": [
                ("What is the size of the Pacific Ocean?", "155.6 million square kilometers"),
                ("What is the size of the Atlantic Ocean?", "106.5 million square kilometers")
            ],
            "cot": "The Pacific Ocean has a size of 155.6 million square kilometers, and the Atlantic Ocean has a size of 106.5 million square kilometers. Since 155.6 million square kilometers is greater than 106.5 million square kilometers, the Pacific Ocean is larger.",
            "final_answer": "yes"
        },
        {
            "question": "When did the fictional character James Barker win his first Nobel Prize?",
            "sub_questions": [
                ("Who is James Barker?", "No reliable information found about a notable person named James Barker winning a Nobel Prize"),
                ("When did James Barker win his first Nobel Prize?", "No information available")
            ],
            "cot": "Based on the subquestions, there is no reliable information about a notable person named James Barker winning a Nobel Prize. The question appears to be asking about a fictional character or contains incorrect assumptions. Without concrete information about this person and their achievements, I cannot determine when they won their first Nobel Prize, or if they won one at all.",
            "final_answer": "[none]"
        },
        {
            "question": "Who was the first person to reach the summit of Mount Everest or K2?",
            "sub_questions": [
                ("Who was the first person to reach the summit of Mount Everest?", "Edmund Hillary and Tenzing Norgay in 1953"),
                ("Who was the first person to reach the summit of K2?", "[none]")
            ],
            "cot": "For Mount Everest, Edmund Hillary and Tenzing Norgay were the first to reach the summit in 1953. For K2, no information is available. Since the question asks for the first person to reach EITHER Mount Everest OR K2, and we have valid information for Mount Everest but not for K2, we can provide the answer based on the available information about Mount Everest.",
            "final_answer": "Edmund Hillary and Tenzing Norgay in 1953"
        }
    ]

    prompt = """You are an expert at answering questions based on provided subquestions and their answers. 

IMPORTANT GUIDELINES:
1. If you cannot extract the necessary information to answer the question, please respond with '[none]'.
2. Your answer MUST follow the structure "so the Final answer is: [your answer]".
3. First provide your chain of thought reasoning in the CoT section, then give your final answer.
Below are some examples to guide you:\n\n"""
    for i, example in enumerate(examples, 1):
        prompt += f"Example {i}:\n"
        prompt += f"Question: {example['question']}\n"
        prompt += "Subquestions:\n"
        for sub_q, sub_a in example['sub_questions']:
            prompt += f"{sub_q}: {sub_a}\n"
        prompt += f"CoT: {example['cot']}\n"
        prompt += f"so the Final answer is: {example['final_answer']}\n\n"

    prompt += "Now, based on the following question and subquestions, generate the final answer:\n\n"
    prompt += f"Question: {current_question}\n"
    prompt += "Subquestions:\n"
    for sub_q, sub_a in current_sub_questions:
        prompt += f"{sub_q}: {sub_a}\n"
    prompt += """
CoT: 
"""
    return prompt