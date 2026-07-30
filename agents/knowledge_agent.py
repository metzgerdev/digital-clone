"""Knowledge Agent — semantic retrieval over a cognitive-science textbook corpus.

Corpus: the `open-phi/textbooks` HuggingFace dataset, filtered to cognitive
science (topic contains "cognitive") — 4 full "Computational Cognitive Science"
textbooks. Reuses RAG-Research-Assistant's DenseRetriever (BGE-large + FAISS
cosine): chunks each book's markdown by section, builds a cached index, and
exposes retrieve(query) -> grounded passages with scores and citations.

    from agents.knowledge_agent import KnowledgeAgent
    kb = KnowledgeAgent().build()          # loads cache, or embeds + caches once
    hits = kb.retrieve("what is a Bayesian model of cognition?", k=5)
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

# Reuse the retriever from the RAG research project.
RAG = Path("/Users/zynarai/Documents/Code/LLMs-from-scratch/RAG-Research-Assistant")
if str(RAG) not in sys.path:
    sys.path.insert(0, str(RAG))
from rag_utils.dense_retriever import DenseRetriever  # noqa: E402

try:
    from .observability import trace
except ImportError:  # run as a script (agents/ on path)
    from observability import trace

DATASET = "open-phi/textbooks"
TOPIC_FILTER = re.compile(r"cognitive", re.I)  # -> the 4 cognitive-science books
INDEX_DIR = Path(__file__).resolve().parent.parent / "knowledge" / "index_cogsci_bge-large"


def chunk_text(text: str, max_chars: int = 800) -> list[str]:
    """Greedy sentence-packing into ~max_chars windows. No heavy deps."""
    sents = re.split(r"(?<=[.!?])\s+", text.strip())
    chunks, cur = [], ""
    for s in sents:
        if len(cur) + len(s) + 1 <= max_chars:
            cur = (cur + " " + s).strip()
        else:
            if cur:
                chunks.append(cur)
            cur = s
    if cur:
        chunks.append(cur)
    return chunks


def split_markdown_sections(md: str) -> list[tuple[str, str]]:
    """Split markdown into (heading, body) pairs on ATX headings (#..######)."""
    parts = re.split(r"(?m)^(#{1,6}\s+.*)$", md)
    sections = []
    if parts and parts[0].strip():
        sections.append(("(intro)", parts[0]))
    for i in range(1, len(parts), 2):
        heading = parts[i].lstrip("#").strip()
        body = parts[i + 1] if i + 1 < len(parts) else ""
        if body.strip():
            sections.append((heading, body))
    return sections


_SKIP_HEADINGS = re.compile(r"table of contents|foreward|foreword|references|bibliography", re.I)


def build_chunks(max_chars: int = 800) -> list[dict]:
    """Load the cognitive-science textbooks and chunk them for retrieval."""
    from datasets import load_dataset

    ds = load_dataset(DATASET, split="train")
    books = [r for r in ds if TOPIC_FILTER.search(r.get("topic") or "")]
    print(f"cognitive-science books: {len(books)}")

    chunks = []
    seen = set()  # dedup identical passages across books
    for bi, r in enumerate(books):
        topic = (r["topic"] or "").strip().rstrip(".")
        book_id = f"cogsci-{bi}"
        for si, (heading, body) in enumerate(split_markdown_sections(r["markdown"])):
            if _SKIP_HEADINGS.search(heading):
                continue
            for j, piece in enumerate(chunk_text(body, max_chars=max_chars)):
                piece = piece.strip()
                if len(piece) < 60:  # drop tiny fragments
                    continue
                key = re.sub(r"\s+", " ", piece).lower()
                if key in seen:  # skip exact duplicates
                    continue
                seen.add(key)
                chunks.append(
                    {
                        "chunk_id": f"{book_id}:{si}:{j}",
                        "content": piece,
                        "paper_id": book_id,
                        "title": f"{topic} (vol {bi + 1})",
                        "section": heading[:80],
                        "page_number": 0,  # retriever expects this key
                    }
                )
    return chunks


class KnowledgeAgent:
    """Grounded semantic search over the cognitive-science textbook corpus."""

    def __init__(self, size: str = "bge-large", index_dir: Path = INDEX_DIR):
        self.size = size
        self.index_dir = Path(index_dir)
        self.retriever: DenseRetriever | None = None

    def build(self, rebuild: bool = False) -> "KnowledgeAgent":
        with trace("kb.build", size=self.size, rebuild=rebuild) as span:
            if self.index_dir.exists() and not rebuild:
                self.retriever = DenseRetriever.load(self.index_dir)
                span["outputs"]["source"] = "cache"
            else:
                chunks = build_chunks()
                print(f"Building index over {len(chunks)} chunks...")
                self.retriever = DenseRetriever(size=self.size).index(chunks)
                self.retriever.save(self.index_dir)
                span["outputs"].update(source="built", chunks=len(chunks))
            span["outputs"]["vectors"] = len(self.retriever.chunks)
        return self

    def retrieve(self, query: str, k: int = 5) -> list[dict]:
        if self.retriever is None:
            self.build()
        with trace("kb.retrieve", k=k, query=query[:80]) as span:
            scores, idx = self.retriever.retrieve(query, k=k)
            hits = []
            for s, i in zip(scores, idx):
                if i < 0:
                    continue
                c = self.retriever.chunks[i]
                section = c.get("section", "")
                hits.append(
                    {
                        "score": float(s),
                        "content": c["content"],
                        "paper_id": c["paper_id"],
                        "title": c["title"],
                        "section": section,
                        "citation": f"{c['title']}" + (f" · {section}" if section else ""),
                    }
                )
            span["outputs"]["hits"] = len(hits)
            span["outputs"]["top_score"] = round(hits[0]["score"], 3) if hits else None
        return hits


if __name__ == "__main__":
    kb = KnowledgeAgent().build()
    for r in kb.retrieve("What is a Bayesian model of human cognition?", k=3):
        print(f"[{r['score']:.3f}] {r['citation']}")
        print("    ", r["content"][:180].replace("\n", " "), "...\n")
