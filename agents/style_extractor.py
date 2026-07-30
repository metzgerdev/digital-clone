"""Style extraction — Agent 1 of the multi-agent system.

Turn raw email text into style features, and aggregate an author's emails into a
single style profile. Factored out of 01_style_extractor.ipynb so later agents
(Profiler, Attributor, Generator, Critic) can import instead of redefining.

    from style_extractor import StyleExtractor, StyleProfile, parse_enron_message

Design rule: style is *form*, not content. Features lean on habits an author uses
unconsciously — function words, punctuation, sentence rhythm, structure.
"""

from __future__ import annotations

import email
import re
from collections import Counter

import nltk
import numpy as np


# --------------------------------------------------------------------------- #
# NLTK data (idempotent). Every step below has a fallback if a download fails. #
# --------------------------------------------------------------------------- #
def _ensure_nltk():
    for pkg in [
        "punkt",
        "punkt_tab",
        "averaged_perceptron_tagger",
        "averaged_perceptron_tagger_eng",
        "stopwords",
        "vader_lexicon",
        "words",
    ]:
        try:
            nltk.download(pkg, quiet=True)
        except Exception as exc:  # pragma: no cover
            print(f"[warn] could not download {pkg}: {exc}")


_ensure_nltk()

from nltk import pos_tag, sent_tokenize, word_tokenize

try:
    from nltk.corpus import stopwords

    STOPWORDS = set(stopwords.words("english"))
except Exception:
    STOPWORDS = set()

try:
    from nltk.sentiment import SentimentIntensityAnalyzer

    _SIA = SentimentIntensityAnalyzer()
except Exception:
    _SIA = None

try:
    from nltk.corpus import words as _nltk_words

    COMMON_WORDS = set(w.lower() for w in _nltk_words.words())
except Exception:
    COMMON_WORDS = set()


FUNCTION_WORDS = set(
    """
a an the this that these those
i me my mine we us our ours you your yours he him his she her hers it its they them their theirs
who whom whose which what
and but or nor for yet so as if than because although though while whereas since unless until whether
of in on at by to from with about against between into through during before after above below
up down out off over under again further then once here there
all any both each few more most other some such no not only own same too very
can will just should now
is am are was were be been being have has had do does did doing
would could may might must shall
onto upon within without
""".split()
)

if not STOPWORDS:
    STOPWORDS = set(FUNCTION_WORDS)


# --------------------------------------------------------------------------- #
# Email parsing and cleaning                                                  #
# --------------------------------------------------------------------------- #
def parse_enron_message(raw: str) -> str:
    """Extract the plain-text body from a raw RFC822 Enron message."""
    try:
        msg = email.message_from_string(raw)
        if msg.is_multipart():
            parts = [
                p.get_payload(decode=False)
                for p in msg.walk()
                if p.get_content_type() == "text/plain"
            ]
            body = "\n".join(parts) if parts else msg.get_payload()
        else:
            body = msg.get_payload()
        return body if isinstance(body, str) else raw
    except Exception:
        return raw


_FWD_MARKERS = re.compile(
    r"^\s*(-----\s*Original Message\s*-----"
    r"|-{2,}\s*Forwarded by"
    r"|_{5,})",
    re.IGNORECASE,
)
_HEADER_ECHO = re.compile(r"^\s*(To|From|Cc|Bcc|Sent|Subject|Date):\s", re.IGNORECASE)


def clean_email(raw: str) -> str:
    """Keep only what the author wrote: drop quoted replies, forwarded chains,
    reply headers, header echoes, and the signature block."""
    lines = raw.splitlines()
    kept = []
    for line in lines:
        stripped = line.strip()
        if _FWD_MARKERS.match(line):  # forwarded / original-message chain: stop here
            break
        if stripped in ("--", "-- "):  # signature delimiter: stop here
            break
        if stripped.startswith(">"):  # quoted reply line
            continue
        if re.match(r"^On .+wrote:$", stripped):
            continue
        if _HEADER_ECHO.match(line):  # echoed header line
            continue
        kept.append(line)
    text = "\n".join(kept)
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    return text


# --------------------------------------------------------------------------- #
# Tokenizers                                                                  #
# --------------------------------------------------------------------------- #
_WORD_RE = re.compile(r"\b\w+\b")


def to_words(text: str, keep_case: bool = False):
    try:
        toks = [t for t in word_tokenize(text) if _WORD_RE.fullmatch(t)]
    except Exception:
        toks = _WORD_RE.findall(text)
    return toks if keep_case else [t.lower() for t in toks]


def to_sentences(text: str):
    try:
        sents = sent_tokenize(text)
    except Exception:
        sents = re.split(r"(?<=[.!?])\s+", text)
    return [s for s in (s.strip() for s in sents) if s]


def to_paragraphs(text: str):
    return [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]


# --------------------------------------------------------------------------- #
# Low-level stylometric features                                              #
# --------------------------------------------------------------------------- #
def f_average_word_length(words):
    return float(np.mean([len(w) for w in words])) if words else 0.0


def f_character_count(text):
    return {
        "char_count": float(len(text)),
        "char_letters": float(sum(c.isalpha() for c in text)),
        "char_digits": float(sum(c.isdigit() for c in text)),
        "char_symbols": float(sum((not c.isalnum()) and (not c.isspace()) for c in text)),
    }


def f_type_token_ratio(words):
    return len(set(words)) / len(words) if words else 0.0


def f_stopword_frequency(words):
    return sum(w in STOPWORDS for w in words) / len(words) if words else 0.0


def f_average_sentence_length(sentences):
    lens = [len(to_words(s)) for s in sentences]
    return float(np.mean(lens)) if lens else 0.0


PUNCT_MAP = {
    "comma": ",",
    "period": ".",
    "semicolon": ";",
    "question": "?",
    "exclaim": "!",
    "colon": ":",
}


def f_punctuation_frequency(text, n_words):
    n = max(n_words, 1)
    return {f"punc_{k}_per_word": text.count(v) / n for k, v in PUNCT_MAP.items()}


def f_paragraph_length(paragraphs):
    if not paragraphs:
        return {"para_sentences_avg": 0.0, "para_words_avg": 0.0}
    return {
        "para_sentences_avg": float(np.mean([len(to_sentences(p)) for p in paragraphs])),
        "para_words_avg": float(np.mean([len(to_words(p)) for p in paragraphs])),
    }


def f_pos_distribution(words_cased):
    empty = {"pos_noun": 0.0, "pos_verb": 0.0, "pos_adj": 0.0, "pos_adv": 0.0}
    if not words_cased:
        return empty
    try:
        tags = [t for _, t in pos_tag(words_cased)]
    except Exception:
        return empty
    n = len(tags)
    frac = lambda pre: sum(t.startswith(pre) for t in tags) / n
    return {
        "pos_noun": frac(("NN",)),
        "pos_verb": frac(("VB",)),
        "pos_adj": frac(("JJ",)),
        "pos_adv": frac(("RB",)),
    }


def f_uppercase_ratio(text):
    letters = [c for c in text if c.isalpha()]
    return sum(c.isupper() for c in letters) / len(letters) if letters else 0.0


SPECIAL_CHARS = "#$%&@*+=/\\|<>~^`()[]{}"


def f_special_char_frequency(text):
    n = max(len(text), 1)
    out = {"special_total_per_char": sum(c in SPECIAL_CHARS for c in text) / n}
    for name, ch in [
        ("hash", "#"),
        ("dollar", "$"),
        ("paren", "("),
        ("bracket", "["),
        ("brace", "{"),
        ("amp", "&"),
    ]:
        out[f"special_{name}_per_char"] = text.count(ch) / n
    return out


def function_word_sequence(words):
    return [w for w in words if w in FUNCTION_WORDS]


# --------------------------------------------------------------------------- #
# Profile features                                                            #
# --------------------------------------------------------------------------- #
GREETINGS = {
    "Hi": r"\bhi\b",
    "Hello": r"\bhello\b",
    "Hey": r"\bhey\b",
    "Dear": r"\bdear\b",
    "Greetings": r"\bgreetings\b",
    "Good morning": r"\bgood morning\b",
    "Good afternoon": r"\bgood afternoon\b",
    "Good evening": r"\bgood evening\b",
}


def detect_greeting(text):
    head = text.strip().lower()[:40]
    for label, pat in GREETINGS.items():
        if re.search(pat, head):
            return label
    return "none"


def punctuation_patterns(text, n_sentences):
    n = max(n_sentences, 1)
    return {
        "ellipsis": len(re.findall(r"\.\.\.|…", text)) / n,
        "dashes": len(re.findall(r"—|–|--|\s-\s", text)) / n,
        "exclaim": text.count("!") / n,
        "question": text.count("?") / n,
        "comma": text.count(",") / n,
        "semicolon": text.count(";") / n,
    }


def question_frequency(text):
    return float(text.count("?"))


REASONING_MARKERS = [
    "because",
    "therefore",
    "thus",
    "hence",
    "since",
    "so that",
    "as a result",
    "consequently",
    "due to",
    "in order to",
    "for this reason",
    "that is why",
]


def reasoning_markers(text):
    low = text.lower()
    return [m for m in REASONING_MARKERS if re.search(r"\b" + re.escape(m) + r"\b", low)]


def sentiment_label(text):
    if _SIA is None:
        return "neu"
    c = _SIA.polarity_scores(text)["compound"]
    return "pos" if c >= 0.05 else "neg" if c <= -0.05 else "neu"


def formality_score(words_cased):
    """Heylighen-Dewaele F-score from POS tags, scaled to 0..1 (higher = formal)."""
    if not words_cased:
        return 0.0
    try:
        tags = [t for _, t in pos_tag(words_cased)]
    except Exception:
        return 0.0
    n = len(tags)
    pct = lambda pre: 100.0 * sum(t.startswith(pre) for t in tags) / n
    f = (
        pct(("NN",))
        + pct(("JJ",))
        + pct(("IN",))
        + pct(("DT",))
        - pct(("PRP", "WP"))
        - pct(("VB",))
        - pct(("RB",))
        - pct(("UH",))
        + 100
    ) / 2
    return max(0.0, min(1.0, f / 100.0))


_TECH_PATTERN = re.compile(r"[_/]|[a-z][A-Z]|\d")
_NON_TECH_CAPS = {"OMG", "WOW", "LOL", "OK", "LMK", "ASAP", "PS", "FYI", "BTW", "AKA"}


def technical_terminology(text):
    toks = to_words(text, keep_case=True)
    if not toks:
        return 0.0
    tech = 0
    for t in toks:
        if _TECH_PATTERN.search(t):
            tech += 1
        elif t.isupper() and len(t) >= 2 and t not in _NON_TECH_CAPS:
            tech += 1
        elif (
            len(t) >= 5
            and COMMON_WORDS
            and t.lower() not in COMMON_WORDS
            and t.lower() not in STOPWORDS
        ):
            tech += 1
    return tech / len(toks)


def content_ngrams(words, lo=2, hi=4):
    grams = []
    for n in range(lo, hi + 1):
        grams += [" ".join(words[i : i + n]) for i in range(len(words) - n + 1)]
    return grams


# --------------------------------------------------------------------------- #
# Agents                                                                      #
# --------------------------------------------------------------------------- #
class StyleExtractor:
    """Agent 1: one email -> a flat dict of numeric style features."""

    def extract(self, raw_email: str) -> dict:
        text = clean_email(raw_email)
        words = to_words(text)
        words_cased = to_words(text, keep_case=True)
        sentences = to_sentences(text)
        paragraphs = to_paragraphs(text)

        feats = {}
        feats["avg_word_length"] = f_average_word_length(words)
        feats.update(f_character_count(text))
        feats["type_token_ratio"] = f_type_token_ratio(words)
        feats["stopword_freq"] = f_stopword_frequency(words)
        feats["avg_sentence_length"] = f_average_sentence_length(sentences)
        feats.update(f_punctuation_frequency(text, len(words)))
        feats.update(f_paragraph_length(paragraphs))
        feats.update(f_pos_distribution(words_cased))
        feats["uppercase_ratio"] = f_uppercase_ratio(text)
        feats.update(f_special_char_frequency(text))
        feats["n_words"] = float(len(words))
        feats["n_sentences"] = float(len(sentences))
        feats["n_paragraphs"] = float(len(paragraphs))
        return feats

    def function_word_skeleton(self, raw_email: str) -> str:
        return " ".join(function_word_sequence(to_words(clean_email(raw_email))))


class StyleProfile:
    """Agent 2 preview: aggregate an author's emails into one style profile."""

    def build(self, emails, min_words: int = 5) -> dict:
        cleaned = [clean_email(e) for e in emails]
        cleaned = [c for c in cleaned if len(to_words(c)) >= min_words]
        if not cleaned:
            return {}
        n_emails = len(cleaned)

        per_msg, greet, sentiment = [], Counter(), Counter()
        reasoning, punct, phrases = Counter(), Counter(), Counter()

        for text in cleaned:
            words = to_words(text)
            words_cased = to_words(text, keep_case=True)
            sentences = to_sentences(text)

            per_msg.append(
                {
                    "msg_len": len(words),
                    "cap_ratio": f_uppercase_ratio(text),
                    "question": question_frequency(text),
                    "vocab": f_type_token_ratio(words),
                    "formality": formality_score(words_cased),
                    "technical": technical_terminology(text),
                }
            )
            greet[detect_greeting(text)] += 1
            for m in reasoning_markers(text):
                reasoning[m] += 1
            for k, v in punctuation_patterns(text, len(sentences)).items():
                punct[k] += v
            for s in sentences:
                sentiment[sentiment_label(s)] += 1
            for g in set(content_ngrams(words)):  # doc-frequency
                phrases[g] += 1

        avg = lambda k: round(float(np.mean([m[k] for m in per_msg])), 3)
        total_sent = sum(sentiment.values()) or 1
        min_df = max(2, n_emails // 20)  # scale phrase threshold with corpus size
        common = [p for p, c in phrases.most_common() if c >= min_df][:10]

        return {
            "n_emails": n_emails,
            "average_message_length": round(avg("msg_len"), 1),
            "greeting_patterns": {
                g: round(c / n_emails, 3) for g, c in greet.most_common() if g != "none"
            },
            "punctuation_patterns": {k: round(v / n_emails, 3) for k, v in punct.items()},
            "capitalization_ratio": avg("cap_ratio"),
            "question_frequency": avg("question"),
            "vocabulary_richness": round(avg("vocab"), 2),
            "common_phrases": common,
            "reasoning_patterns": [m for m, _ in reasoning.most_common()],
            "sentiment_distribution": {
                k: round(sentiment.get(k, 0) / total_sent, 2) for k in ("pos", "neu", "neg")
            },
            "formality_level": round(avg("formality"), 2),
            "technical_terminology": round(avg("technical"), 2),
        }


if __name__ == "__main__":
    sample = "Hi team, quick update: the fix is in. Tests pass. Let me know if CI breaks!"
    print("features:", {k: round(v, 3) for k, v in StyleExtractor().extract(sample).items()})
