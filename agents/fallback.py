"""Calendar fallback — the graceful exit when confidence is too low (Phase 4).

When the system cannot answer confidently (no supporting knowledge, or it never
cleared the bar after reflection), it declines to guess and offers a call instead,
in the author's voice. Booking is stubbed for v1.
"""

from __future__ import annotations

DEFAULT_BOOKING_URL = "https://cal.com/kay-mann/15min"


def calendar_fallback(
    question: str, name: str = "Kay", *, booking_url: str = DEFAULT_BOOKING_URL
) -> str:
    """A short, in-voice message that declines to guess and offers a call."""
    return (
        "Hi,\n\n"
        "Thanks for reaching out. I don't have enough in front of me to answer this "
        "confidently, and I'd rather not guess. Could we set up a quick call to talk it "
        f"through? Here's my calendar: {booking_url}\n\n"
        f"Thanks,\n{name}"
    )
