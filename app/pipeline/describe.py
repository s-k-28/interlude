"""Description drafting with a fit-constrained retry loop.

A description that overruns its silence window is worse than useless: it gets
cut off mid-sentence by the next line of dialogue. So drafting is not a single
model call. It is a loop:

    draft -> evaluate against the word budget -> if over, redraft with the
    measured overage fed back in -> repeat

This is the conditional, multi-attempt orchestration at the centre of the
pipeline. The evaluator is deterministic and pure, so the loop's control flow is
fully testable without touching a model.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass, field

from app.pipeline.gaps import Gap

# A trailing possessive or hyphenated token still counts as one word.
_WORD_RE = re.compile(r"[\w'-]+")


def count_words(text: str) -> int:
    return len(_WORD_RE.findall(text))


@dataclass(frozen=True, slots=True)
class FitVerdict:
    """The result of checking one draft against one gap."""

    fits: bool
    word_count: int
    budget: int

    @property
    def overage(self) -> int:
        """Words that must be removed. Zero when the draft fits."""
        return max(0, self.word_count - self.budget)


def evaluate_fit(text: str, gap: Gap) -> FitVerdict:
    """Check whether ``text`` can be spoken inside ``gap``."""
    words = count_words(text)
    return FitVerdict(fits=words <= gap.word_budget, word_count=words, budget=gap.word_budget)


@dataclass(slots=True)
class DraftAttempt:
    """One iteration of the drafting loop, retained for the manifest."""

    attempt: int
    text: str
    word_count: int
    budget: int
    accepted: bool


@dataclass(slots=True)
class DescriptionResult:
    """Final outcome of drafting a description for a single gap."""

    gap: Gap
    text: str
    accepted: bool
    attempts: list[DraftAttempt] = field(default_factory=list)

    @property
    def attempt_count(self) -> int:
        return len(self.attempts)


# A drafter takes (prompt, word_budget) and returns candidate text. The real
# implementation calls a vision model; tests substitute a deterministic stub.
Drafter = Callable[[str, int], str]


def build_prompt(
    scene_context: str,
    budget: int,
    *,
    previous_attempt: str = "",
    overage: int = 0,
) -> str:
    """Construct the drafting prompt.

    On a retry the prompt carries the previous draft and the exact number of
    words to cut. Telling the model *how far over* it went produces materially
    better second attempts than simply asking again for something shorter.
    """
    base = (
        "You are writing audio description for a blind listener.\n"
        "Describe only what is visually present. Do not interpret, editorialise, "
        "or describe audio.\n"
        f"Hard limit: {budget} words.\n\n"
        f"Scene: {scene_context}"
    )
    if not previous_attempt:
        return base
    return (
        f"{base}\n\n"
        f"Your previous attempt was {overage} word(s) over the limit:\n"
        f'"{previous_attempt}"\n\n'
        f"Rewrite it to fit within {budget} words. Cut detail, not grammar."
    )


def draft_description(
    gap: Gap,
    scene_context: str,
    drafter: Drafter,
    *,
    max_attempts: int = 3,
) -> DescriptionResult:
    """Draft a description that fits ``gap``, retrying on overrun.

    Args:
        gap: The silence window to fill.
        scene_context: What the vision model observed at this timestamp.
        drafter: Callable producing candidate text.
        max_attempts: Cap on model calls. After the last attempt the best
            candidate is truncated to the budget rather than discarded.

    Returns:
        A :class:`DescriptionResult`. ``accepted`` is False when every attempt
        overran and the text had to be truncated.
    """
    if max_attempts < 1:
        raise ValueError(f"max_attempts must be >= 1, got {max_attempts}")

    if gap.word_budget == 0:
        return DescriptionResult(gap=gap, text="", accepted=False, attempts=[])

    attempts: list[DraftAttempt] = []
    previous_text = ""
    overage = 0

    for attempt_number in range(1, max_attempts + 1):
        prompt = build_prompt(
            scene_context,
            gap.word_budget,
            previous_attempt=previous_text,
            overage=overage,
        )
        candidate = drafter(prompt, gap.word_budget).strip()
        verdict = evaluate_fit(candidate, gap)

        attempts.append(
            DraftAttempt(
                attempt=attempt_number,
                text=candidate,
                word_count=verdict.word_count,
                budget=verdict.budget,
                accepted=verdict.fits,
            )
        )

        if verdict.fits:
            return DescriptionResult(
                gap=gap, text=candidate, accepted=True, attempts=attempts
            )

        previous_text = candidate
        overage = verdict.overage

    # Every attempt overran. Truncate the last candidate on a word boundary so
    # the listener hears a clean partial description rather than a clipped one.
    truncated = " ".join(_WORD_RE.findall(previous_text)[: gap.word_budget])
    return DescriptionResult(gap=gap, text=truncated, accepted=False, attempts=attempts)
