"""Evaluation harness — run a labelled question set through the full system and
report how it behaves (Phase 5).

Measures the things that matter for a digital clone: does it SEND grounded
answers on in-domain questions and FALL BACK on out-of-domain ones, how confident
is it, how well does the draft match the author's style, and how often does the
reflection loop kick in.

    python evaluate.py
"""

import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "agents"))

from evaluator import Evaluator  # noqa: E402
from knowledge_agent import KnowledgeAgent  # noqa: E402
from orchestrator import Orchestrator  # noqa: E402
from style_agent import StyleAgent, make_client  # noqa: E402

# (question, expected decision) — in-domain cognitive science should SEND,
# out-of-domain should FALL BACK.
QUESTIONS = [
    ("How do Bayesian models explain human cognition?", "send"),
    ("What is the difference between symbolic and connectionist models of the mind?", "send"),
    ("What role does working memory play in problem solving?", "send"),
    ("How is reinforcement learning used to model decision making?", "send"),
    ("What were Enron's Q3 2001 earnings?", "fallback"),
    ("What is the capital of France?", "fallback"),
]

LOG = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs", "eval_results.jsonl")


def main():
    sa = StyleAgent(name="Kay Mann")
    kb = KnowledgeAgent().build()
    client = make_client()
    orch = Orchestrator(sa, kb, Evaluator(sa, client), client)

    rows = []
    print(f"\n{'question':<50} exp   dec    conf  style  grnd  att")
    print("-" * 84)
    for q, expected in QUESTIONS:
        r = orch.run(q)
        v = r["verdict"]
        row = {
            "question": q,
            "expected": expected,
            "decision": r["decision"],
            "confidence": v["confidence"],
            "style": v["style"],
            "grounding": v["grounding"],
            "retrieval": v["retrieval"],
            "attempts": r["attempts"],
            "correct": r["decision"] == expected,
        }
        rows.append(row)
        flag = "ok " if row["correct"] else "MISS"
        print(
            f"{q[:48]:<50}{expected:<6}{r['decision']:<7}{v['confidence']:>5}"
            f"{v['style']:>7}{v['grounding']:>6}{r['attempts']:>5}  [{flag}]"
        )

    # ---- aggregate report ------------------------------------------------- #
    n = len(rows)
    sent = [r for r in rows if r["decision"] == "send"]
    fell = [r for r in rows if r["decision"] == "fallback"]
    reflected = [r for r in rows if r["attempts"] > 1]
    correct = [r for r in rows if r["correct"]]
    mean = lambda key, rs: round(sum(r[key] for r in rs) / len(rs), 3) if rs else 0.0

    print("\n" + "=" * 40 + " summary " + "=" * 35)
    print(f"  questions              : {n}")
    print(f"  routing accuracy       : {len(correct)}/{n} ({round(100 * len(correct) / n)}%)")
    print(
        f"  send rate              : {round(100 * len(sent) / n)}%   "
        f"fallback rate: {round(100 * len(fell) / n)}%"
    )
    print(
        f"  reflection rate        : {round(100 * len(reflected) / n)}% (re-drafted at least once)"
    )
    print(f"  mean confidence (sent) : {mean('confidence', sent)}")
    print(f"  mean style (sent)      : {mean('style', sent)}")
    print(f"  mean grounding (sent)  : {mean('grounding', sent)}")

    os.makedirs(os.path.dirname(LOG), exist_ok=True)
    with open(LOG, "w") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    print(f"\n  saved per-question results to {os.path.relpath(LOG)}")


if __name__ == "__main__":
    main()
