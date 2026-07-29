"""Tests for token accounting."""

from __future__ import annotations

import pytest

from app.pipeline.tokens import CallRecord, TokenLedger, approximate_tokens


class TestApproximateTokens:
    def test_empty(self) -> None:
        assert approximate_tokens("") == 0

    def test_rounds_up(self) -> None:
        assert approximate_tokens("abcde") == 2  # 5 chars -> ceil(5/4)

    def test_monotonic(self) -> None:
        assert approximate_tokens("a" * 100) > approximate_tokens("a" * 50)


class TestCallRecord:
    def test_saved(self) -> None:
        assert CallRecord(uncompressed_tokens=100, compressed_tokens=30).saved == 70

    def test_negative_saving_clamped(self) -> None:
        assert CallRecord(uncompressed_tokens=30, compressed_tokens=100).saved == 0


class TestTokenLedger:
    def test_starts_empty(self) -> None:
        ledger = TokenLedger()
        assert ledger.call_count == 0
        assert ledger.reduction_ratio == 0.0

    def test_records_calls(self) -> None:
        ledger = TokenLedger()
        ledger.record(uncompressed_tokens=100, compressed_tokens=25)
        ledger.record(uncompressed_tokens=200, compressed_tokens=50)
        assert ledger.call_count == 2
        assert ledger.total_uncompressed == 300
        assert ledger.total_compressed == 75
        assert ledger.total_saved == 225

    def test_reduction_ratio(self) -> None:
        ledger = TokenLedger()
        ledger.record(uncompressed_tokens=1000, compressed_tokens=250)
        assert ledger.reduction_ratio == pytest.approx(0.75)

    def test_rejects_negative_counts(self) -> None:
        with pytest.raises(ValueError, match="non-negative"):
            TokenLedger().record(uncompressed_tokens=-1, compressed_tokens=0)

    def test_completion_tokens_tracked_separately(self) -> None:
        # Completion tokens are not compressible; mixing them into the prompt
        # ratio would inflate the reported saving.
        ledger = TokenLedger()
        ledger.record(uncompressed_tokens=100, compressed_tokens=20, completion_tokens=40)
        assert ledger.total_completion == 40
        assert ledger.reduction_ratio == pytest.approx(0.8)

    def test_summary_shape(self) -> None:
        ledger = TokenLedger()
        ledger.record(uncompressed_tokens=100, compressed_tokens=25)
        summary = ledger.summary()
        assert summary == {
            "calls": 1,
            "prompt_tokens_uncompressed": 100,
            "prompt_tokens_compressed": 25,
            "completion_tokens": 0,
            "tokens_saved": 75,
            "reduction_ratio": 0.75,
        }

    def test_summary_keeps_raw_counts_unrounded(self) -> None:
        ledger = TokenLedger()
        ledger.record(uncompressed_tokens=99991, compressed_tokens=33337)
        assert ledger.summary()["prompt_tokens_uncompressed"] == 99991
