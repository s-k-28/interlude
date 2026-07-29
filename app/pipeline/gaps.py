"""Silence-gap detection.

Audio description must be spoken *between* lines of dialogue. This module finds
those windows and computes how many words fit in each one.

Pure logic, no I/O, no network. Fully unit-testable without API keys.
"""

from __future__ import annotations

from dataclasses import dataclass

# Audio Description Coalition style guidance puts natural described-media
# narration at 150-180 wpm. 165 is the midpoint and what we budget against.
WORDS_PER_MINUTE = 165

# Below this a description is clipped and unintelligible to a listener.
MIN_GAP_SECONDS = 1.4


@dataclass(frozen=True, slots=True)
class Gap:
    """A silence window in which a description can be spoken."""

    start: float
    end: float

    def __post_init__(self) -> None:
        if self.end < self.start:
            raise ValueError(f"gap end {self.end} precedes start {self.start}")

    @property
    def duration(self) -> float:
        return self.end - self.start

    @property
    def word_budget(self) -> int:
        """Maximum words speakable in this window at WORDS_PER_MINUTE."""
        return max(0, int(self.duration * WORDS_PER_MINUTE / 60))


def find_gaps(
    words: list[tuple[float, float]],
    video_duration: float,
    min_gap: float = MIN_GAP_SECONDS,
) -> list[Gap]:
    """Return the silence windows in a video, in chronological order.

    Args:
        words: ``(start, end)`` timestamps in seconds for each spoken word, as
            returned by a transcription provider. May be unsorted and may
            contain overlaps (common with multi-speaker diarization).
        video_duration: Total video length in seconds.
        min_gap: Shortest window considered usable.

    Returns:
        Non-overlapping :class:`Gap` objects, each at least ``min_gap`` long.
    """
    if video_duration < 0:
        raise ValueError(f"video_duration must be non-negative, got {video_duration}")
    if min_gap <= 0:
        raise ValueError(f"min_gap must be positive, got {min_gap}")

    if not words:
        # Silent video: the whole thing is one window.
        return [Gap(0.0, video_duration)] if video_duration >= min_gap else []

    ordered = sorted(words)
    gaps: list[Gap] = []

    # Leading silence, before anyone speaks.
    first_word_start = ordered[0][0]
    if first_word_start >= min_gap:
        gaps.append(Gap(0.0, first_word_start))

    # Interior silences. `cursor` tracks the furthest point any word has
    # reached, so overlapping speech collapses into one spoken region instead
    # of producing phantom gaps.
    cursor = ordered[0][1]
    for start, end in ordered[1:]:
        if start - cursor >= min_gap:
            gaps.append(Gap(cursor, start))
        cursor = max(cursor, end)

    # Trailing silence, after the last word.
    if video_duration - cursor >= min_gap:
        gaps.append(Gap(cursor, video_duration))

    return gaps


def total_describable_seconds(gaps: list[Gap]) -> float:
    """Sum of all usable silence. Used for coverage reporting."""
    return sum(gap.duration for gap in gaps)
