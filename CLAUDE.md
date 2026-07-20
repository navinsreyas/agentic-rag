\# CLAUDE.md - System Instructions \& Documentation Standards



\## 1. The "Glass Box" Protocol

I am using this project to learn. I need to understand \*everything\* you generate.

\*\*Rule:\*\* Never generate code without being ready to explain the "why" behind it. If a piece of logic is complex (like `operator.add` or conditional edges), you must proactively explain it in plain English.



\## 2. The "Domain Bridge" Protocol (Legal Engineering)

This is not just a coding project; it is a \*\*Legal AI\*\* project.

\*\*Rule:\*\* When you write code that touches data (like filtering logic, schema definitions, or prompt engineering), you must explain the \*\*Legal/Business Intent\*\*.

\* \*Bad:\* "We filter strings < 50 chars."

\* \*Good:\* "We filter clauses < 50 tokens because legally substantive clauses (like Indemnification) require nuance. Short snippets are likely just headers or definitions, which don't teach the model reasoning."



\## 3. The "FOR\_ME.md" Documentation Standard

At the end of every major phase (or when explicitly asked), you must generate a summary file named `FOR\_ME.md`.



\*\*The Goal:\*\* This file should be the only thing I need to read to deeply understand the project, explain it in an interview, and feel like I wrote every line myself.



\*\*Tone \& Style:\*\*

\* \*\*Narrative \& Engaging:\*\* Write like a Principal Engineer explaining a cool architecture to a smart colleague over coffee. Use analogies. Make it memorable.

\* \*\*No "Textbook" Jargon:\*\* If you use a technical term, explain it. Avoid dry, passive documentation.

\* \*\*Radical Transparency:\*\* Reveal the trade-offs. Why did we choose X over Y?



\*\*Required Structure for `FOR\_ME.md`:\*\*



\### 1. The "One-Liner" Pitch

\* What is this project in simple terms?

\* Why does it exist? (The problem it solves).



\### 2. The "Nervous System" (Architecture)

\* \*\*The Blueprint:\*\* Explain the directory structure (`src/`, `configs/`, `scripts/`).

\* \*\*The Flow:\*\* How do the parts talk to each other? (e.g., "The Data Processor feeds the Synthetic Generator...").

\* \*\*Visual Mental Model:\*\* Use an analogy to explain the flow.



\### 3. Under the Hood (Tech Stack \& Decisions)

\* \*\*The Tools:\*\* Unsloth, Pydantic, Claude API, etc.

\* \*\*The "Why":\*\* Why did we choose these specific tools? (e.g., "We used Pydantic because LLMs are non-deterministic, and we need strict structural guarantees...").



\### 4. War Stories (Lessons \& "Gotchas")

\* \*\*The Bugs:\*\* "We ran into a bug where JSON generation failed. Here is exactly why (the 'pre-fill' trick) and how we fixed it."

\* \*\*The Pitfalls:\*\* "A common mistake beginners make here is X. We avoided it by doing Y."

\* \*\*The "Senior" Insight:\*\* How would a good engineer think about this? (e.g., "Always fail fast," "Schema-First Design").



\### 5. New Superpowers

\* What specific skills did I unlock in this phase? (e.g., "You now understand Structured Generation").



---



\## 4. Immediate Action Protocol

If I ask you to "Explain this code," do not just summarize what the code does. Explain \*\*why it was written that way\*\* and \*\*what would happen if we deleted it\*\*.

1. Before writing any code, describe your approach and wait for approval.

2. If the requirements I give you are ambiguous, ask clarifying questions before writing any code.

3. After you finish writing any code, list the edge cases and suggest test cases to cover them.

4. If a task requires changes to more than 3 files, stop and break it into smaller tasks first.

5. When there’s a bug, start by writing a test that reproduces it, then fix it until the test passes.

6. Every time I correct you, reflect on what you did wrong and come up with a plan to never make the same mistake again.