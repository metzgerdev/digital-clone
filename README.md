# Digital Clone — Multi-Agent Email Responder

A multi-agent system that answers a question in the email voice of a real
employee, grounds every claim in a knowledge base, checks its own work, and
offers to book a call when it is not confident.

- **Style** comes from the Enron email corpus. The system profiles one employee —
  Kay Mann, the corpus's most prolific sender (~8,900 sent emails).
- **Knowledge** comes from four cognitive-science textbooks (the
  `open-phi/textbooks` dataset, filtered to cognitive science).

See [PLAN.md](PLAN.md) for the full build plan and [project.md](project.md) for the brief.

## How it works

Ask a cognitive-science question, as if you emailed the employee. The system:

1. **Learns the voice** — a style profile plus the author's most representative emails.
2. **Retrieves facts** — semantic search (BGE-large + FAISS) over the textbooks.
3. **Drafts a reply** — states the retrieved facts in the author's voice, with citations.
4. **Grades itself** — style match, factual grounding, and confidence.
5. **Falls back** — offers a calendar booking when confidence is low.

```
question --> Orchestrator (LangGraph)
   |- Style Agent      profile + nearest exemplar emails
   |- Knowledge Agent  semantic search -> grounded passages + citations
   Draft (Claude) --> Evaluator (style / grounding / confidence)
   confident?   yes -> send    no -> reflect & retry    unsupported -> book a call
```

## Status

| Phase | Component | State |
|---|---|---|
| 0 | Style Extractor + Profiler | done |
| 1 | Knowledge Agent (cognitive-science retrieval) | done |
| 2 | Style-conditioned drafting | next |
| 3 | Evaluator (style / grounding / confidence) | planned |
| 4 | Orchestrator + calendar fallback | planned |
| 5 | Demo + evaluation harness | planned |

## Layout

```
agents/
  style_extractor.py    style features + StyleProfile (Agent 1)
  build_profile.py      build an author profile from a maildir (Agent 2)
  knowledge_agent.py    semantic retrieval over the textbook corpus
profiles/
  profile_mann-k.json   Kay Mann's learned style profile
knowledge/              cached FAISS index (generated, gitignored)
PLAN.md  project.md
```

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Build the knowledge index once (downloads the dataset and BGE-large, then embeds):

```bash
python agents/knowledge_agent.py
```

Profile an employee from an Enron maildir:

```bash
python agents/build_profile.py <maildir_root>              # picks the top sender
TARGET=kaminski-v python agents/build_profile.py <maildir_root>
```

Phase 2 reads an OpenRouter key from `.env`:

```
OPENROUTER_API_KEY=sk-...
```

## Reused components and data

- **Retrieval** — `DenseRetriever` (BGE-large + FAISS) from a sibling RAG project.
- **Style** — Enron Email Dataset (public).
- **Knowledge** — `open-phi/textbooks` (Hugging Face), cognitive-science subset.
- **LLM** — Claude via OpenRouter.

Generated artifacts (the FAISS index, extracted emails) are gitignored and rebuilt
from the scripts above.
