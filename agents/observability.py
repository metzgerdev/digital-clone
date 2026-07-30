"""Observability for the multi-agent system: config-change auditing and tracing.

Two logging entry points:

    log_config_change(key, old, new, reason=...)     # audit trail of settings changes
    with trace("draft", model=...) as span:          # timed, nestable execution spans
        ...
        span["outputs"]["tokens"] = 42

Both write a human-readable line to the console (stdlib logging) and a structured
JSON line to logs/ for later analysis — who changed what, and per-step latency /
error rate (p95, failure counts). Pure standard library, no heavy dependencies.
"""

from __future__ import annotations

import contextvars
import json
import logging
import time
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

LOG_DIR = Path(__file__).resolve().parent.parent / "logs"
CONFIG_LOG = LOG_DIR / "config_changes.jsonl"
CONFIG_MD = LOG_DIR / "config_changes.md"
TRACE_LOG = LOG_DIR / "traces.jsonl"

logger = logging.getLogger("digital_clone")


def configure_logging(level: int = logging.INFO) -> None:
    """Attach a console handler once. Safe to call repeatedly."""
    if logger.handlers:
        return
    logger.setLevel(level)
    handler = logging.StreamHandler()
    handler.setFormatter(
        logging.Formatter("%(asctime)s  %(levelname)-5s  %(message)s", datefmt="%H:%M:%S")
    )
    logger.addHandler(handler)


def _iso_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def _short_id() -> str:
    return uuid.uuid4().hex[:12]


def _append_jsonl(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as fh:
        fh.write(json.dumps(record, default=str) + "\n")


def _append_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as fh:
        fh.write(text)


# --------------------------------------------------------------------------- #
# 1. configuration-change logging                                             #
# --------------------------------------------------------------------------- #
def _is_num(x) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool)


def _fmt_num_delta(before: float, after: float) -> str:
    """Rate-like values (|v| <= 1) render as percentage points, else raw."""
    d = after - before
    if abs(before) <= 1 and abs(after) <= 1:
        return f"{d * 100:+.0f}%"
    return f"{d:+.3g}"


def _compute_delta(before, after):
    """Per-metric delta for the shared numeric keys of two metric dicts."""
    if not (isinstance(before, dict) and isinstance(after, dict)):
        return None
    out = {
        k: _fmt_num_delta(before[k], after[k])
        for k in before
        if k in after and _is_num(before[k]) and _is_num(after[k])
    }
    return out or None


def _pretty(key: str) -> str:
    return key.replace("_", " ").capitalize()


def _metric_cell(m) -> str:
    if m is None:
        return "—"
    if isinstance(m, dict):
        return ", ".join(f"{_pretty(k)} = {v}" for k, v in m.items())
    return str(m)


def _delta_cell(delta) -> str:
    if delta is None:
        return "—"
    if isinstance(delta, dict):
        return ", ".join(f"{_pretty(k)}: {v}" for k, v in delta.items())
    return str(delta)


def _keep_cell(keep, keep_reason: str) -> str:
    label = {True: "Yes", False: "No", None: "Pending"}[keep]
    return f"{label} — {keep_reason}" if keep_reason else label


def render_config_markdown(record: dict) -> str:
    """Render a config-change record as the standard Field / Value table."""
    rows = [
        ("Change", record["change"]),
        ("Reason", record["reason"]),
        ("Metric Before", _metric_cell(record["metric_before"])),
        ("Metric After", _metric_cell(record["metric_after"])),
        ("Delta", _delta_cell(record["delta"])),
        ("Keep?", _keep_cell(record["keep"], record["keep_reason"])),
    ]
    lines = [f"### {record['ts']} · {record['actor']}", "", "| Field | Value |", "| --- | --- |"]
    lines += [f"| {field} | {value} |" for field, value in rows]
    return "\n".join(lines) + "\n"


def log_config_change(
    change: str,
    reason: str,
    *,
    metric_before: dict | str | None = None,
    metric_after: dict | str | None = None,
    keep: bool | None = None,
    keep_reason: str = "",
    delta: str | dict | None = None,
    actor: str = "system",
) -> dict:
    """Record a configuration change as a tuning experiment.

    Captures the standard fields — Change, Reason, Metric Before / After, Delta,
    Keep? — so every threshold / prompt / model change is an auditable decision
    with before-and-after numbers.

        log_config_change(
            change="Adjusted fallback threshold from 0.75 to 0.70",
            reason="Fallback rate was 55%, above the 30-40% target",
            metric_before={"fallback_rate": 0.55, "final_score_mean": 0.68},
            metric_after={"fallback_rate": 0.38, "final_score_mean": 0.68},
            keep=True, keep_reason="now in target band, score unchanged")

    Delta is computed automatically from numeric metric dicts (rate-like values as
    percentage points) unless passed explicitly. Writes a JSON line to
    logs/config_changes.jsonl and a Field/Value table to logs/config_changes.md.
    Returns the record.
    """
    configure_logging()
    if delta is None:
        delta = _compute_delta(metric_before, metric_after)
    record = {
        "ts": _iso_now(),
        "event": "config_change",
        "change": change,
        "reason": reason,
        "metric_before": metric_before,
        "metric_after": metric_after,
        "delta": delta,
        "keep": keep,
        "keep_reason": keep_reason,
        "actor": actor,
    }
    _append_jsonl(CONFIG_LOG, record)
    _append_text(CONFIG_MD, render_config_markdown(record) + "\n")
    logger.info(
        "config  %s | delta %s | keep=%s", change, _delta_cell(delta), _keep_cell(keep, keep_reason)
    )
    return record


# --------------------------------------------------------------------------- #
# 2. execution tracing                                                        #
# --------------------------------------------------------------------------- #
_current_span: contextvars.ContextVar = contextvars.ContextVar("current_span", default=None)
_trace_id: contextvars.ContextVar = contextvars.ContextVar("trace_id", default=None)
_depth: contextvars.ContextVar = contextvars.ContextVar("trace_depth", default=0)


@contextmanager
def trace(name: str, **attrs):
    """Time a block as a span. Nestable — child spans inherit the parent's trace_id.

    Attach results inside the block via `span["outputs"][key] = value`. On exit it
    records duration, status (ok/error), and any error, then appends the span to
    logs/traces.jsonl and prints an indented console line.

        with trace("respond", question=q) as root:
            with trace("retrieve", k=5) as s:
                hits = kb.retrieve(q); s["outputs"]["hits"] = len(hits)
            with trace("draft", model="claude") as s:
                s["outputs"]["tokens"] = draft.tokens
            root["outputs"]["confidence"] = score

    Yields the span dict so the caller can enrich it.
    """
    configure_logging()
    parent = _current_span.get()
    trace_id = _trace_id.get() or _short_id()
    depth = _depth.get()

    span = {
        "trace_id": trace_id,
        "span_id": _short_id(),
        "parent_id": parent["span_id"] if parent else None,
        "name": name,
        "attrs": attrs,
        "outputs": {},
        "ts": _iso_now(),
    }
    tid_tok = _trace_id.set(trace_id)
    span_tok = _current_span.set(span)
    depth_tok = _depth.set(depth + 1)

    t0 = time.perf_counter()
    status, error = "ok", None
    try:
        yield span
    except Exception as exc:
        status, error = "error", repr(exc)
        raise
    finally:
        span["duration_ms"] = round((time.perf_counter() - t0) * 1000, 2)
        span["status"] = status
        if error:
            span["error"] = error
        _depth.reset(depth_tok)
        _current_span.reset(span_tok)
        _trace_id.reset(tid_tok)
        _append_jsonl(TRACE_LOG, span)
        tag = "OK " if status == "ok" else "ERR"
        extra = f"  {span['outputs']}" if span["outputs"] else ""
        logger.info(
            "trace  %s%s %s (%.1f ms)%s", "  " * depth, name, tag, span["duration_ms"], extra
        )


if __name__ == "__main__":
    configure_logging()

    # 1. configuration changes (tuning experiments)
    log_config_change(
        change="Adjusted fallback threshold from 0.75 to 0.70",
        reason="Fallback rate was 55%, above the 30-40% target",
        metric_before={"fallback_rate": 0.55, "final_score_mean": 0.68},
        metric_after={"fallback_rate": 0.38, "final_score_mean": 0.68},
        keep=True,
        keep_reason="fallback rate now in target band, score unchanged",
        actor="nam",
    )
    log_config_change(
        change="Switched draft model from flan-t5 to claude-via-openrouter",
        reason="local model too weak for style mimicry",
        keep=True,
        keep_reason="qualitatively much stronger drafts",
    )

    # 2. nested tracing (mimics the orchestrator loop)
    with trace("respond", question="what is bayesian cognition?") as root:
        with trace("retrieve", k=5) as s:
            time.sleep(0.05)
            s["outputs"]["hits"] = 5
        with trace("draft", model="claude") as s:
            time.sleep(0.10)
            s["outputs"]["tokens"] = 128
        root["outputs"]["confidence"] = 0.82

    print(f"\nlogs written under {LOG_DIR}/")
