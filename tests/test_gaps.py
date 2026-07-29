"""Tests for silence-gap detection."""

from __future__ import annotations

import pytest

from app.pipeline.gaps import (
    MIN_GAP_SECONDS,
    WORDS_PER_MINUTE,
    Gap,
    find_gaps,
    total_describable_seconds,
)


class TestGap:
    def test_duration(self) -> None:
        assert Gap(2.0, 5.5).duration == pytest.approx(3.5)

    def test_word_budget_matches_narration_rate(self) -> None:
        # 60s at 165 wpm == 165 words.
        assert Gap(0.0, 60.0).word_budget == WORDS_PER_MINUTE

    def test_word_budget_floors_rather_than_rounds(self) -> None:
        # 2.0s -> 5.5 words -> must floor to 5, never 6. Over-budget text
        # would be cut off mid-sentence by the following dialogue.
        assert Gap(0.0, 2.0).word_budget == 5

    def test_word_budget_never_negative(self) -> None:
        assert Gap(3.0, 3.0).word_budget == 0

    def test_rejects_inverted_bounds(self) -> None:
        with pytest.raises(ValueError, match="precedes start"):
            Gap(5.0, 2.0)

    def test_is_hashable(self) -> None:
        # frozen=True; gaps are used as dict keys when caching descriptions.
        assert len({Gap(0.0, 1.0), Gap(0.0, 1.0)}) == 1


class TestFindGaps:
    def test_silent_video_is_one_gap(self) -> None:
        assert find_gaps([], 10.0) == [Gap(0.0, 10.0)]

    def test_silent_video_shorter_than_min_gap_yields_nothing(self) -> None:
        assert find_gaps([], 0.5) == []

    def test_leading_silence_detected(self) -> None:
        gaps = find_gaps([(3.0, 4.0)], 4.0)
        assert gaps == [Gap(0.0, 3.0)]

    def test_leading_silence_below_threshold_skipped(self) -> None:
        assert find_gaps([(0.5, 4.0)], 4.0) == []

    def test_trailing_silence_detected(self) -> None:
        gaps = find_gaps([(0.0, 2.0)], 10.0)
        assert gaps == [Gap(2.0, 10.0)]

    def test_interior_gap_detected(self) -> None:
        gaps = find_gaps([(0.0, 2.0), (8.0, 9.0)], 9.0)
        assert gaps == [Gap(2.0, 8.0)]

    def test_interior_gap_below_threshold_skipped(self) -> None:
        # 1.0s < MIN_GAP_SECONDS
        assert find_gaps([(0.0, 2.0), (3.0, 5.0)], 5.0) == []

    def test_gap_exactly_at_threshold_is_included(self) -> None:
        gaps = find_gaps([(0.0, 2.0), (2.0 + MIN_GAP_SECONDS, 6.0)], 6.0)
        assert len(gaps) == 1
        assert gaps[0].duration == pytest.approx(MIN_GAP_SECONDS)

    def test_unsorted_input_is_handled(self) -> None:
        unsorted_words = [(8.0, 9.0), (0.0, 2.0)]
        assert find_gaps(unsorted_words, 9.0) == [Gap(2.0, 8.0)]

    def test_overlapping_words_do_not_create_phantom_gaps(self) -> None:
        # Speaker B starts before speaker A finishes. Naive iteration would
        # see (10.0 -> 3.0) and emit a negative-length gap.
        words = [(0.0, 10.0), (3.0, 4.0), (11.5, 12.0)]
        gaps = find_gaps(words, 12.0)
        assert gaps == [Gap(10.0, 11.5)]

    def test_fully_contained_word_is_absorbed(self) -> None:
        words = [(0.0, 10.0), (2.0, 3.0)]
        assert find_gaps(words, 10.0) == []

    def test_multiple_gaps_in_order(self) -> None:
        words = [(2.0, 3.0), (6.0, 7.0), (10.0, 11.0)]
        gaps = find_gaps(words, 15.0)
        assert gaps == [Gap(0.0, 2.0), Gap(3.0, 6.0), Gap(7.0, 10.0), Gap(11.0, 15.0)]
        # Gaps must never overlap each other.
        for earlier, later in zip(gaps, gaps[1:]):
            assert earlier.end <= later.start

    def test_custom_min_gap_is_respected(self) -> None:
        words = [(0.0, 2.0), (4.0, 5.0)]
        assert find_gaps(words, 5.0, min_gap=1.0) == [Gap(2.0, 4.0)]
        assert find_gaps(words, 5.0, min_gap=3.0) == []

    def test_rejects_negative_duration(self) -> None:
        with pytest.raises(ValueError, match="non-negative"):
            find_gaps([], -1.0)

    def test_rejects_nonpositive_min_gap(self) -> None:
        with pytest.raises(ValueError, match="must be positive"):
            find_gaps([], 10.0, min_gap=0.0)


class TestTotalDescribableSeconds:
    def test_sums_durations(self) -> None:
        assert total_describable_seconds([Gap(0.0, 2.0), Gap(5.0, 8.0)]) == pytest.approx(5.0)

    def test_empty_is_zero(self) -> None:
        assert total_describable_seconds([]) == 0.0
