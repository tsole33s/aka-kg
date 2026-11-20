\# aka — Autonomous Knowledge Acquisition



A minimal, modular library for \*\*autonomous knowledge acquisition\*\*: a frozen

LLM is used as an environment, while a lightweight outer learner decides what

to ask next, extracts concepts from the answers, and builds a knowledge graph

whose structural gaps drive the next round of questions.



The design follows the "outer layer" architecture validated in

\[Dream-Layer](https://github.com/aggarwamnew/dream-layer): a frozen LLM plus a

separate learning loop that observes, forms concepts, and selects questioning

strategies. The key empirical finding it operationalizes is that

\*\*feedback-driven questioning produces substantially denser knowledge graphs

than structure-only or random baselines\*\*, and that keyword accumulation

without semantic representation fails to build real connections.



\---



\## What it does



Given a seed corpus and any chat-completion LLM, `aka` will:



1\. Extract semantic features from text and promote frequently-seen ones into

&#x20;  stable \*\*concepts\*\* (Module 1).

2\. Build an undirected \*\*knowledge graph\*\* where edges encode shared features,

&#x20;  weighted by Jaccard similarity (Module 2).

3\. Read structural gaps from that graph and turn them into LLM-ready

&#x20;  \*\*question prompts\*\* using five fixed strategies (Module 3).

4\. Choose which strategy to fire next using \*\*Discounted Thompson Sampling\*\*,

&#x20;  rewarded by the information gain each answer produces in the graph

&#x20;  (Module 4).

5\. Run the whole thing against a \*\*cached, retried, provider-neutral LLM\*\*

&#x20;  wrapper (Module 5a), with \*\*atomic checkpointing\*\* so runs resume cleanly

&#x20;  (Module 5b).



The loop in one line:




`observe(text)` -> `ConceptLayer` (features & concepts) -> `KnowledgeGraph` (`GraphDelta` & Jaccard edges) -> `QuestionGenerator` (5 strategies) -> `StrategySelector` (Discounted Thompson Sampling) -> repeat.
