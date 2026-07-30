"""Orchestrator — the LangGraph state machine that ties it all together (Phase 4).

    question -> draft -> evaluate -> decide
                                       send      : confidence >= threshold
                                       reflect   : grounded but below bar, attempts left -> re-draft
                                       fallback  : unsupported / out of attempts -> offer a call

The reflection loop feeds the evaluator's critique back into the next draft. The
fallback offers a calendar booking in the author's voice. Every node is traced.

    from agents.orchestrator import Orchestrator
    orch = Orchestrator(style_agent, kb, evaluator, client)
    result = orch.run("How do Bayesian models explain cognition?")
    print(result["decision"], result["reply"])
"""

from __future__ import annotations

import os
import sys
from typing import TypedDict

from langgraph.graph import END, START, StateGraph

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fallback import calendar_fallback  # noqa: E402

try:
    from .observability import trace
except ImportError:
    from observability import trace


class State(TypedDict, total=False):
    question: str
    draft: str
    chunks: list
    style_score: float
    citations: list
    verdict: dict
    attempts: int
    feedback: str
    decision: str
    reply: str


class Orchestrator:
    def __init__(
        self,
        style_agent,
        kb,
        evaluator,
        client,
        *,
        threshold: float = 0.70,
        retrieval_floor: float = 0.30,
        max_attempts: int = 2,
    ):
        self.sa = style_agent
        self.kb = kb
        self.ev = evaluator
        self.client = client
        self.threshold = threshold
        self.retrieval_floor = retrieval_floor
        self.max_attempts = max_attempts
        self.graph = self._build()

    # --- nodes ------------------------------------------------------------- #
    def _draft(self, state: State) -> dict:
        attempt = state.get("attempts", 0) + 1
        with trace("orch.draft", attempt=attempt):
            out = self.sa.draft(
                state["question"], self.kb, self.client, feedback=state.get("feedback", "")
            )
        return {
            "draft": out["draft"],
            "chunks": out["chunks"],
            "style_score": out["style_score"],
            "citations": out["citations"],
            "attempts": attempt,
        }

    def _evaluate(self, state: State) -> dict:
        verdict = self.ev.evaluate(
            {
                "draft": state["draft"],
                "chunks": state["chunks"],
                "style_score": state.get("style_score"),
            }
        )
        return {"verdict": verdict}

    def _reflect(self, state: State) -> dict:
        v = state["verdict"]
        notes = []
        if v["style"] < 0.6:
            notes.append(
                f"Sound more like {self.sa.name}: shorter, warmer, open with "
                '"Hi", end with a question, sign off "Thanks, '
                f'{self.sa.name.split()[0]}".'
            )
        if v["unsupported"]:
            notes.append(
                "Remove or hedge these unsupported claims: " + "; ".join(v["unsupported"][:3])
            )
        if not notes:
            notes.append("Tighten the answer and stay closer to the references.")
        return {"feedback": "\n".join(f"- {n}" for n in notes)}

    def _send(self, state: State) -> dict:
        return {"decision": "send", "reply": state["draft"]}

    def _fallback(self, state: State) -> dict:
        return {"decision": "fallback", "reply": calendar_fallback(state["question"], self.sa.name)}

    # --- routing ----------------------------------------------------------- #
    def _decide(self, state: State) -> str:
        v = state["verdict"]
        if v["confidence"] >= self.threshold:
            return "send"
        if v["retrieval"] >= self.retrieval_floor and state["attempts"] < self.max_attempts:
            return "reflect"
        return "fallback"

    def _build(self):
        g = StateGraph(State)
        g.add_node("draft", self._draft)
        g.add_node("evaluate", self._evaluate)
        g.add_node("reflect", self._reflect)
        g.add_node("send", self._send)
        g.add_node("fallback", self._fallback)
        g.add_edge(START, "draft")
        g.add_edge("draft", "evaluate")
        g.add_conditional_edges(
            "evaluate", self._decide, {"send": "send", "reflect": "reflect", "fallback": "fallback"}
        )
        g.add_edge("reflect", "draft")
        g.add_edge("send", END)
        g.add_edge("fallback", END)
        return g.compile()

    def run(self, question: str) -> dict:
        with trace("orch.run", question=question[:80]) as span:
            final = self.graph.invoke({"question": question, "attempts": 0, "feedback": ""})
            span["outputs"].update(
                decision=final["decision"],
                confidence=final["verdict"]["confidence"],
                attempts=final["attempts"],
            )
        return final


if __name__ == "__main__":
    from evaluator import Evaluator
    from knowledge_agent import KnowledgeAgent
    from style_agent import StyleAgent, make_client

    sa = StyleAgent(name="Kay Mann")
    kb = KnowledgeAgent().build()
    client = make_client()
    ev = Evaluator(sa, client)
    orch = Orchestrator(sa, kb, ev, client, threshold=0.70)

    questions = [
        "How do Bayesian models explain human cognition?",  # expect: send
        "What role does working memory play in problem solving?",  # expect: send
        "What were Enron's Q3 2001 earnings?",  # expect: fallback
    ]
    for q in questions:
        r = orch.run(q)
        print("=" * 78)
        print(f"Q: {q}")
        print(
            f"decision={r['decision'].upper()}  confidence={r['verdict']['confidence']}  "
            f"attempts={r['attempts']}  style={r['verdict']['style']}  "
            f"grounding={r['verdict']['grounding']}"
        )
        print("-" * 78)
        print(r["reply"])
        print()
