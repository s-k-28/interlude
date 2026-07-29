"""Tests for provenance manifests.

The central property under test: a manifest's hash must change if and only if
the recorded provenance changes. Everything else follows from that.
"""

from __future__ import annotations

import json

import pytest

from app.pipeline.manifest import (
    AssetRecord,
    Manifest,
    ManifestError,
    ManifestVerification,
    StepRecord,
    TokenUsage,
    canonical_json,
    parse_manifest,
    verify_hash,
)


def sample() -> Manifest:
    m = Manifest.new("run-1", "source/ab/cd/deadbeef.mp4", "deadbeef")
    m.add_step(
        StepRecord(
            index=0,
            name="transcribe",
            provider="assemblyai",
            model="best",
            output_keys=["transcript/aa/bb/cc"],
        )
    )
    m.add_asset(
        AssetRecord(
            key="transcript/aa/bb/cc",
            sha256="cc",
            size=120,
            namespace="transcript",
            content_type="application/json",
        )
    )
    return m


class TestCanonicalJson:
    def test_key_order_does_not_affect_output(self) -> None:
        assert canonical_json({"b": 1, "a": 2}) == canonical_json({"a": 2, "b": 1})

    def test_no_incidental_whitespace(self) -> None:
        assert canonical_json({"a": 1, "b": 2}) == '{"a":1,"b":2}'

    def test_rejects_nan(self) -> None:
        # NaN is not valid JSON; permitting it would produce a document other
        # parsers reject, making the hash unverifiable elsewhere.
        with pytest.raises(ManifestError, match="not canonically serializable"):
            canonical_json({"x": float("nan")})

    def test_rejects_infinity(self) -> None:
        with pytest.raises(ManifestError):
            canonical_json({"x": float("inf")})

    def test_unicode_preserved_not_escaped(self) -> None:
        assert canonical_json({"k": "café"}) == '{"k":"café"}'


class TestTokenUsage:
    def test_tokens_saved(self) -> None:
        assert TokenUsage(prompt_tokens_uncompressed=1000, prompt_tokens_compressed=250).tokens_saved == 750

    def test_reduction_ratio(self) -> None:
        usage = TokenUsage(prompt_tokens_uncompressed=1000, prompt_tokens_compressed=250)
        assert usage.reduction_ratio == pytest.approx(0.75)

    def test_no_division_by_zero(self) -> None:
        assert TokenUsage().reduction_ratio == 0.0

    def test_negative_saving_clamped(self) -> None:
        # Compression that made the prompt larger is reported as zero saving,
        # never as a negative. We do not flatter the number in either direction.
        usage = TokenUsage(prompt_tokens_uncompressed=100, prompt_tokens_compressed=140)
        assert usage.tokens_saved == 0
        assert usage.reduction_ratio == 0.0


class TestManifest:
    def test_requires_run_id(self) -> None:
        with pytest.raises(ManifestError, match="run_id must not be empty"):
            Manifest.new("", "k", "h")

    def test_hash_is_stable_across_calls(self) -> None:
        m = sample()
        assert m.canonical_hash() == m.canonical_hash()

    def test_identical_content_hashes_identically(self) -> None:
        a, b = sample(), sample()
        b.created_at = a.created_at  # only the timestamp differs
        assert a.canonical_hash() == b.canonical_hash()

    def test_hash_changes_when_a_step_is_added(self) -> None:
        m = sample()
        before = m.canonical_hash()
        m.add_step(StepRecord(index=1, name="describe", provider="google", model="gemini"))
        assert m.canonical_hash() != before

    def test_hash_changes_when_a_seed_changes(self) -> None:
        a, b = sample(), sample()
        b.created_at = a.created_at
        b.steps[0] = StepRecord(**{**vars_of(a.steps[0]), "seed": 42})
        assert a.canonical_hash() != b.canonical_hash()

    def test_hash_changes_when_token_counts_change(self) -> None:
        a, b = sample(), sample()
        b.created_at = a.created_at
        b.tokens = TokenUsage(prompt_tokens_uncompressed=10, prompt_tokens_compressed=5)
        assert a.canonical_hash() != b.canonical_hash()

    def test_to_json_embeds_its_own_hash(self) -> None:
        doc = json.loads(sample().to_json())
        assert len(doc["canonical_hash"]) == 64

    def test_rejected_attempts_are_recorded(self) -> None:
        m = sample()
        m.add_step(
            StepRecord(
                index=1, name="describe", provider="google", model="gemini",
                attempts=3, accepted=True,
            )
        )
        doc = json.loads(m.to_json())
        assert doc["steps"][1]["attempts"] == 3


class TestVerifyHash:
    def test_accepts_untampered_document(self) -> None:
        assert verify_hash(sample().to_json())

    def test_rejects_tampered_document(self) -> None:
        doc = json.loads(sample().to_json())
        doc["source"]["sha256"] = "forged"
        assert not verify_hash(json.dumps(doc))

    def test_rejects_tampered_asset_list(self) -> None:
        doc = json.loads(sample().to_json())
        doc["assets"][0]["size"] = 999999
        assert not verify_hash(json.dumps(doc))


class TestParseManifest:
    def test_rejects_invalid_json(self) -> None:
        with pytest.raises(ManifestError, match="not valid JSON"):
            parse_manifest("{nope")

    def test_rejects_non_object_root(self) -> None:
        with pytest.raises(ManifestError, match="root must be an object"):
            parse_manifest("[1,2,3]")

    def test_rejects_document_without_hash(self) -> None:
        with pytest.raises(ManifestError, match="missing canonical_hash"):
            parse_manifest('{"run_id":"x"}')


class TestManifestVerification:
    def test_ok_when_clean(self) -> None:
        assert ManifestVerification(hash_ok=True).ok

    def test_not_ok_when_hash_fails(self) -> None:
        assert not ManifestVerification(hash_ok=False).ok

    def test_not_ok_when_asset_missing(self) -> None:
        assert not ManifestVerification(hash_ok=True, missing_assets=["k"]).ok

    def test_not_ok_when_asset_corrupt(self) -> None:
        assert not ManifestVerification(hash_ok=True, corrupt_assets=["k"]).ok


def vars_of(record: StepRecord) -> dict:
    """StepRecord is slotted and frozen; rebuild its fields as a dict."""
    from dataclasses import asdict

    return asdict(record)
