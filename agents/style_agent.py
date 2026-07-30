"""Style Agent — draft a grounded reply in a cloned author's email voice (Phase 2).

Turns a learned style profile + the author's real emails into a drafting prompt:
style directives (from the profile numbers) + the most representative emails as
few-shot exemplars + retrieved knowledge chunks. Drafts via Claude on OpenRouter.

    from agents.style_agent import StyleAgent, make_client
    from agents.knowledge_agent import KnowledgeAgent

    sa = StyleAgent(name="Kay Mann")
    kb = KnowledgeAgent().build()
    out = sa.draft("How do Bayesian models explain cognition?", kb, make_client())
    print(out["draft"])
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from style_extractor import StyleExtractor  # noqa: E402

try:
    from .observability import trace
except ImportError:
    from observability import trace

ROOT = Path(__file__).resolve().parent.parent
PROFILE_PATH = ROOT / "profiles" / "profile_mann-k.json"
EMAILS_PATH = ROOT / "profiles" / "mann-k_emails.jsonl"
DEFAULT_MODEL = "anthropic/claude-sonnet-4-6"


# --------------------------------------------------------------------------- #
# OpenRouter client                                                           #
# --------------------------------------------------------------------------- #
def load_key(env_path: Path = ROOT / ".env") -> str:
    """Read OPENROUTER_API_KEY from the environment or the project .env."""
    key = os.getenv("OPENROUTER_API_KEY")
    if not key and env_path.exists():
        for line in env_path.read_text().splitlines():
            if line.strip().startswith("OPENROUTER_API_KEY"):
                key = line.split("=", 1)[1].strip().strip('"').strip("'")
                break
    if not key:
        raise SystemExit("OPENROUTER_API_KEY not set (put it in multi-agent/.env).")
    return key


def make_client():
    from openai import OpenAI

    return OpenAI(base_url="https://openrouter.ai/api/v1", api_key=load_key())


# --------------------------------------------------------------------------- #
# Style Agent                                                                 #
# --------------------------------------------------------------------------- #
class StyleAgent:
    def __init__(
        self,
        name: str | None = None,
        profile_path: Path = PROFILE_PATH,
        emails_path: Path = EMAILS_PATH,
    ):
        blob = json.loads(Path(profile_path).read_text())
        self.profile = blob["profile"]
        self.name = name or blob.get("employee", "the author")

        self.emails = [
            json.loads(line) for line in Path(emails_path).read_text().splitlines() if line.strip()
        ]
        self.extractor = StyleExtractor()
        self._fit_style_space()

    # --- style space: vectors, standardisation, typicality ----------------- #
    def _fit_style_space(self):
        feats = [self.extractor.extract(e["body"]) for e in self.emails]
        self._keys = sorted({k for f in feats for k in f})
        self._X = np.array([[f.get(k, 0.0) for k in self._keys] for f in feats], float)
        self._mu = self._X.mean(axis=0)
        self._sd = self._X.std(axis=0) + 1e-9
        z = np.abs((self._X - self._mu) / self._sd)
        self._typicality = z.mean(axis=1)  # smaller = more typical
        self._order = np.argsort(self._typicality)  # most representative first

    def _vec(self, feats: dict) -> np.ndarray:
        return np.array([feats.get(k, 0.0) for k in self._keys], float)

    def style_score(self, text: str) -> float:
        """0..1 — how typical `text` is of the author's style (1 = perfectly typical)."""
        z = np.abs((self._vec(self.extractor.extract(text)) - self._mu) / self._sd)
        return round(1.0 / (1.0 + float(z.mean())), 3)

    def exemplars(self, n: int = 5, min_words: int = 30, max_words: int = 160) -> list[str]:
        """The author's most representative emails within a sane length range."""
        out = []
        for i in self._order:
            e = self.emails[int(i)]
            if min_words <= e.get("n_words", 0) <= max_words:
                out.append(e["body"].strip())
            if len(out) >= n:
                break
        return out

    # --- prompt --------------------------------------------------------------- #
    def directives(self) -> str:
        """Turn the profile numbers into explicit style instructions."""
        p = self.profile
        lines = []
        greet = p.get("greeting_patterns") or {}
        if greet:
            top = max(greet, key=greet.get)
            lines.append(f'Usually open with "{top}", or no greeting at all.')
        lines.append(f"Keep it short — around {round(p.get('average_message_length', 60))} words.")
        f = p.get("formality_level", 0.5)
        tone = "formal" if f > 0.66 else "warm but professional" if f > 0.5 else "casual and direct"
        lines.append(f"Tone: {tone}.")
        if p.get("question_frequency", 0) >= 0.5:
            lines.append("Often ask a question, especially to move things forward.")
        rm = p.get("reasoning_patterns") or []
        if rm:
            lines.append(f"Reason things through explicitly, using words like {', '.join(rm[:5])}.")
        punc = p.get("punctuation_patterns") or {}
        if punc.get("comma", 0) > 0.4:
            lines.append("Write comma-rich, flowing sentences.")
        if punc.get("exclaim", 0) > 0.3:
            lines.append("Use the occasional exclamation point.")
        if (p.get("sentiment_distribution") or {}).get("pos", 0) >= 0.35:
            lines.append("Keep a positive, courteous tone.")
        signoff = next(
            (ph for ph in (p.get("common_phrases") or []) if "thank" in ph or "regard" in ph), None
        )
        if signoff:
            lines.append(f'Sign off the way you usually do (e.g., "{signoff.title()}").')
        return "\n".join(f"- {ln}" for ln in lines)

    @staticmethod
    def _format_refs(chunks: list[dict]) -> str:
        blocks = []
        for i, c in enumerate(chunks, 1):
            sec = f" · {c['section']}" if c.get("section") else ""
            blocks.append(f"[{i}] {c['title']}{sec}\n{c['content']}")
        return "\n\n".join(blocks)

    def build_messages(
        self, question: str, chunks: list[dict], n_exemplars: int = 5, feedback: str = ""
    ) -> list[dict]:
        system = (
            f"You are drafting an email reply as {self.name}. Write in {self.name}'s voice, "
            "matching the style guide below. Answer using ONLY the reference material, and cite "
            "sources inline by their [n] number. If the references do not support an answer, say "
            "so briefly rather than inventing facts. Return only the email body — no subject line, "
            "no preamble, no meta commentary."
        )
        exemplars = "\n\n---\n\n".join(self.exemplars(n_exemplars))
        user = (
            f"{self.name}'s writing style:\n{self.directives()}\n\n"
            f"Real examples of {self.name}'s emails:\n---\n{exemplars}\n---\n\n"
            f"Reference material (cite as [n]):\n{self._format_refs(chunks)}\n\n"
            f'Someone emailed this question:\n"{question}"\n\n'
            f"Write {self.name}'s reply."
        )
        if feedback:
            user += f"\n\nRevision notes — a previous draft fell short, fix these:\n{feedback}"
        return [{"role": "system", "content": system}, {"role": "user", "content": user}]

    # --- draft --------------------------------------------------------------- #
    def draft(
        self,
        question: str,
        kb,
        client,
        *,
        model: str = DEFAULT_MODEL,
        k: int = 6,
        n_exemplars: int = 5,
        temperature: float = 0.7,
        max_tokens: int = 600,
        feedback: str = "",
    ) -> dict:
        with trace("style.draft", model=model, k=k) as span:
            hits = kb.retrieve(question, k=k)
            messages = self.build_messages(question, hits, n_exemplars, feedback=feedback)
            with trace("llm.call", model=model):
                resp = client.chat.completions.create(
                    model=model, messages=messages, temperature=temperature, max_tokens=max_tokens
                )
            text = resp.choices[0].message.content.strip()
            score = self.style_score(text)
            span["outputs"].update(
                style_score=score,
                chars=len(text),
                top_ref=round(hits[0]["score"], 3) if hits else None,
            )
        return {
            "question": question,
            "draft": text,
            "style_score": score,
            "citations": [h["citation"] for h in hits],
            "chunks": hits,
        }


if __name__ == "__main__":
    from knowledge_agent import KnowledgeAgent

    sa = StyleAgent(name="Kay Mann")
    kb = KnowledgeAgent().build()
    client = make_client()

    print(
        f"Author style score of a real exemplar: "
        f"{sa.style_score(sa.exemplars(1)[0]):.3f}  (1.0 = perfectly typical)\n"
    )

    for q in [
        "How do Bayesian models explain human cognition?",
        "What is the difference between symbolic and connectionist models of the mind?",
    ]:
        out = sa.draft(q, kb, client)
        print("=" * 78)
        print("Q:", q)
        print(f"style_score={out['style_score']}   refs={out['citations'][:2]}")
        print("-" * 78)
        print(out["draft"])
        print()
