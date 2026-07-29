"""Token reporting through a completed run.

Supersedes the token assertions in test_pipeline_integration.py, which access
``manifest.tokens`` by subscript. ``Manifest.tokens`` is a ``TokenUsage``
dataclass, not a mapping, so attribute access is correct.
"""

from __future__ import annotations

import pytest

from app.pipeline.manifest import TokenUsage
from app.pipeline.runner import JobState, build_manifest
from app.pipeline.tokens import TokenLedger


def test_ledger_totals_reach_the_manifest() -> None:
    state = JobState.new("source/k", "abc")
    state.ledger.record(uncompressed_tokens=1000, compressed_tokens=250)
    state.ledger.record(uncompressed_tokens=500, compressed_tokens=125)

    manifest = build_manifest(state)

    assert manifest.tokens.prompt_tokens_uncompressed == 1500
    assert manifest.tokens.prompt_tokens_compressed == 375
    assert manifest.tokens.calls == 2


def test_reduction_ratio_is_computed_not_stored() -> None:
    state = JobState.new("source/k", "abc")
    state.ledger.record(uncompressed_tokens=1000, compressed_tokens=250)

    assert build_manifest(state).tokens.reduction_ratio == pytest.approx(0.75)


def test_unmeasured_run_reports_zero_not_a_crash() -> None:
    # A run with Paritok disabled records nothing. Reporting must degrade to
    # zero rather than dividing by zero or claiming a saving that never happened.
    manifest = build_manifest(JobState.new("source/k", "abc"))

    assert manifest.tokens.calls == 0
    assert manifest.tokens.reduction_ratio == 0.0
    assert manifest.tokens.tokens_saved == 0


def test_token_counts_are_serialized_into_the_payload() -> None:
    state = JobState.new("source/k", "abc")
    state.ledger.record(uncompressed_tokens=800, compressed_tokens=200)

    payload = build_manifest(state).to_payload()

    assert payload["tokens"]["prompt_tokens_uncompressed"] == 800
    assert payload["tokens"]["tokens_saved"] == 600
    assert payload["tokens"]["reduction_ratio"] == pytest.approx(0.75)


def test_negative_saving_never_reported() -> None:
    # Compression that inflated the prompt must report zero, never a negative.
    usage = TokenUsage(prompt_tokens_uncompressed=100, prompt_tokens_compressed=140)

    assert usage.tokens_saved == 0
    assert usage.reduction_ratio == 0.0
