"""Tests for the fit-constrained drafting loop.

The drafter is stubbed throughout, so these tests exercise real control flow
with no network access and no API keys.
"""

from __future__ import annotations

import pytest

from app.pipeline.describe import (
    DescriptionResult,
    build_prompt,
    count_words,
    draft_description,
    evaluate_fit,
)
from app.pipeline.gaps import Gap


def words(n: int) -> str:
    return " ".join(f"w{i}" for i in range(n))


class TestCountWords:
    def test_simple(self) -> None:
        assert count_words("a man walks in") == 4

    def test_empty(self) -> None:
        assert count_words("") == 0
        assert count_words("   ") == 0

    def test_punctuation_not_counted(self) -> None:
        assert count_words("A man, alone.") == 3

    def test_hyphenated_is_one_word(self) -> None:
        assert count_words("a well-lit room") == 3

    def test_possessive_is_one_word(self) -> None:
        assert count_words("the man's hat") == 3


class TestEvaluateFit:
    def test_fits_when_under_budget(self) -> None:
        gap = Gap(0.0, 60.0)  # 165-word budget
        verdict = evaluate_fit(words(10), gap)
        assert verdict.fits
        assert verdict.overage == 0

    def test_fits_exactly_at_budget(self) -> None:
        gap = Gap(0.0, 60.0)
        assert evaluate_fit(words(165), gap).fits

    def test_does_not_fit_one_word_over(self) -> None:
        gap = Gap(0.0, 60.0)
        verdict = evaluate_fit(words(166), gap)
        assert not verdict.fits
        assert verdict.overage == 1


class TestBuildPrompt:
    def test_first_attempt_has_no_retry_language(self) -> None:
        prompt = build_prompt("a lecture hall", 12)
        assert "12 words" in prompt
        assert "previous attempt" not in prompt

    def test_retry_includes_overage_and_prior_text(self) -> None:
        prompt = build_prompt("a hall", 5, previous_attempt="too many words here", overage=3)
        assert "3 word(s) over" in prompt
        assert "too many words here" in prompt

    def test_instructs_visual_only(self) -> None:
        assert "Do not interpret" in build_prompt("x", 5)


class TestDraftDescription:
    def test_accepts_first_fitting_draft(self) -> None:
        gap = Gap(0.0, 10.0)  # 27-word budget
        result = draft_description(gap, "a hall", lambda _p, _b: words(5))
        assert result.accepted
        assert result.attempt_count == 1

    def test_retries_when_over_budget(self) -> None:
        gap = Gap(0.0, 2.0)  # 5-word budget
        sizes = iter([20, 9, 4])
        result = draft_description(gap, "a hall", lambda _p, _b: words(next(sizes)))
        assert result.accepted
        assert result.attempt_count == 3
        assert [a.accepted for a in result.attempts] == [False, False, True]

    def test_retry_prompt_receives_measured_overage(self) -> None:
        gap = Gap(0.0, 2.0)  # 5-word budget
        seen: list[str] = []

        def drafter(prompt: str, _b: int) -> str:
            seen.append(prompt)
            return words(8) if len(seen) == 1 else words(3)

        draft_description(gap, "a hall", drafter)
        assert "previous attempt" not in seen[0]
        # 8 words against a 5-word budget == 3 over.
        assert "3 word(s) over" in seen[1]

    def test_truncates_after_exhausting_attempts(self) -> None:
        gap = Gap(0.0, 2.0)  # 5-word budget
        result = draft_description(
            gap, "a hall", lambda _p, _b: words(50), max_attempts=2
        )
        assert not result.accepted
        assert result.attempt_count == 2
        assert count_words(result.text) == 5

    def test_truncation_lands_on_word_boundary(self) -> None:
        gap = Gap(0.0, 2.0)
        result = draft_description(
            gap, "a hall", lambda _p, _b: "alpha beta gamma delta epsilon zeta eta", max_attempts=1
        )
        assert result.text == "alpha beta gamma delta epsilon"

    def test_zero_budget_gap_short_circuits(self) -> None:
        calls = 0

        def drafter(_p: str, _b: int) -> str:
            nonlocal calls
            calls += 1
            return "text"

        result = draft_description(Gap(0.0, 0.0), "a hall", drafter)
        assert result.text == ""
        assert not result.accepted
        assert calls == 0  # never wastes a model call on an unusable gap

    def test_respects_max_attempts_cap(self) -> None:
        calls = 0

        def drafter(_p: str, _b: int) -> str:
            nonlocal calls
            calls += 1
            return words(100)

        draft_description(Gap(0.0, 2.0), "a hall", drafter, max_attempts=4)
        assert calls == 4

    def test_rejects_invalid_max_attempts(self) -> None:
        with pytest.raises(ValueError, match="must be >= 1"):
            draft_description(Gap(0.0, 5.0), "x", lambda _p, _b: "y", max_attempts=0)

    def test_whitespace_is_stripped_from_model_output(self) -> None:
        result = draft_description(Gap(0.0, 10.0), "x", lambda _p, _b: "  a man walks  ")
        assert result.text == "a man walks"

    def test_attempt_history_is_retained_for_manifest(self) -> None:
        gap = Gap(0.0, 2.0)
        sizes = iter([20, 4])
        result = draft_description(gap, "a hall", lambda _p, _b: words(next(sizes)))
        # Provenance requires the rejected drafts, not just the winner.
        assert [a.word_count for a in result.attempts] == [20, 4]
        assert [a.budget for a in result.attempts] == [5, 5]
