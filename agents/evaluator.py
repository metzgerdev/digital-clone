"""Evaluator — grade a draft on style, grounding, and confidence (Phase 3).

Three scores per draft:
  - style       : how typical the draft is of the author (from StyleAgent).
  - grounding   : fraction of the draft's claims supported by the retrieved
                  chunks, judged by an LLM fact-checker.
  - retrieval   : how strong the supporting evidence was (top chunk scores).
Combined into a single confidence the Orchestrator (Phase 4) uses to decide
send / reflect / fall back.

    from agents.evaluator import Evaluator
    ev = Evaluator(style_agent, client)
    verdict = ev.evaluate(style_agent.draft(question, kb, client))
"""

from __future__ import annotations

import json
import os
import re
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from .observability import trace
except ImportError:
    from observability import trace

DEFAULT_MODEL = "anthropic/claude-sonnet-4-6"
DEFAULT_WEIGHTS = {"grounding": 0.5, "retrieval": 0.25, "style": 0.25}

_GROUND_PROMPT = """You are a strict fact-checker. Decide whether the ANSWER is supported by \
the CONTEXT. Treat the CONTEXT as the only source of truth. Check each factual claim in the \
ANSWER against the CONTEXT.

Return ONLY JSON, no prose:
{{"grounding_score": <float 0.0-1.0 = fraction of claims supported by the context>,
  "unsupported_claims": [<short strings>],
  "notes": "<one sentence>"}}

CONTEXT:
{context}

ANSWER:
{answer}"""


def _parse_json(text: str) -> dict:
    s = text.strip()
    if s.startswith("```"):
        s = re.sub(r"^```[a-z]*\n?", "", s)
        s = re.sub(r"\n?```$", "", s)
    try:
        return json.loads(s)
    except Exception:
        m = re.search(r"\{.*\}", s, re.S)
        if m:
            try:
                return json.loads(m.group(0))
            except Exception:
                return {}
        return {}


def _retrieval_signal(chunks: list[dict]) -> float:
    """Normalise mean top-3 cosine into 0..1 (0.55 -> 0, 0.85 -> 1)."""
    if not chunks:
        return 0.0
    mean_top = float(np.mean([c["score"] for c in chunks[:3]]))
    return min(1.0, max(0.0, (mean_top - 0.55) / 0.30))


class Evaluator:
    def __init__(
        self, style_agent, client, *, model: str = DEFAULT_MODEL, weights: dict | None = None
    ):
        self.style_agent = style_agent
        self.client = client
        self.model = model
        self.weights = dict(weights or DEFAULT_WEIGHTS)

    def grounding(self, answer: str, chunks: list[dict]) -> dict:
        """LLM fact-check: is the answer supported by the chunks?"""
        with trace("eval.grounding", model=self.model) as span:
            context = self.style_agent._format_refs(chunks)
            prompt = _GROUND_PROMPT.format(context=context, answer=answer)
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                max_tokens=400,
            )
            data = _parse_json(resp.choices[0].message.content)
            score = float(data.get("grounding_score", 0.0))
            score = min(1.0, max(0.0, score))
            span["outputs"]["grounding"] = round(score, 3)
        return {
            "score": score,
            "unsupported": data.get("unsupported_claims", []),
            "notes": data.get("notes", ""),
        }

    def evaluate(self, result: dict) -> dict:
        """result = StyleAgent.draft(...) output. Returns the three scores + confidence."""
        with trace("eval.evaluate") as span:
            draft, chunks = result["draft"], result["chunks"]
            style = result.get("style_score") or self.style_agent.style_score(draft)
            ground = self.grounding(draft, chunks)
            retrieval = _retrieval_signal(chunks)
            w = self.weights
            confidence = (
                w["grounding"] * ground["score"] + w["retrieval"] * retrieval + w["style"] * style
            )
            verdict = {
                "style": round(style, 3),
                "grounding": round(ground["score"], 3),
                "retrieval": round(retrieval, 3),
                "confidence": round(float(confidence), 3),
                "unsupported": ground["unsupported"],
                "notes": ground["notes"],
            }
            span["outputs"].update(confidence=verdict["confidence"], grounding=verdict["grounding"])
        return verdict


if __name__ == "__main__":
    from knowledge_agent import KnowledgeAgent
    from observability import log_config_change
    from style_agent import StyleAgent, make_client

    sa = StyleAgent(name="Kay Mann")
    kb = KnowledgeAgent().build()
    client = make_client()
    ev = Evaluator(sa, client)

    questions = [
        "How do Bayesian models explain human cognition?",
        "What is the difference between symbolic and connectionist models of the mind?",
        "What role does working memory play in problem solving?",
        "What were Enron's Q3 2001 earnings?",  # out-of-domain -> should be low
    ]

    print(f"\n{'question':<52} style  grnd  retr  CONF")
    print("-" * 78)
    verdicts = []
    for q in questions:
        v = ev.evaluate(sa.draft(q, kb, client))
        verdicts.append(v)
        print(
            f"{q[:50]:<52}{v['style']:>5} {v['grounding']:>5} "
            f"{v['retrieval']:>5} {v['confidence']:>5}"
        )

    confs = [v["confidence"] for v in verdicts]

    # A real tuning experiment: how many fall back at 0.75 vs 0.70?
    def fallback_rate(threshold):
        return round(sum(c < threshold for c in confs) / len(confs), 2)

    before, after = fallback_rate(0.75), fallback_rate(0.70)
    print(f"\nfallback rate @0.75 = {before}   @0.70 = {after}")
    log_config_change(
        change="Adjusted fallback threshold from 0.75 to 0.70",
        reason="calibrating the send/fallback boundary on the demo question set",
        metric_before={"fallback_rate": before},
        metric_after={"fallback_rate": after},
        keep=(after <= before),
        keep_reason="keeps genuinely-unsupported answers in fallback while sending grounded ones",
        actor="nam",
    )
    print("\nlogged the threshold experiment to logs/config_changes.md")
