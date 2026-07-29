"""Provenance manifests.

Every run emits a manifest: a canonical JSON document recording what was
generated, by which model, from which inputs, and with what parameters. The
document is hashed, and the hash is stored alongside it, so any later claim
about an asset's origin is verifiable rather than merely asserted.

Canonicalization matters. Two runs producing identical work must produce
byte-identical manifests, or the hash proves nothing. We therefore sort keys,
pin separators, and reject non-finite floats.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from typing import Any

SCHEMA_VERSION = "1.0"


class ManifestError(RuntimeError):
    """Raised when a manifest cannot be built or verified."""


def canonical_json(payload: dict[str, Any]) -> str:
    """Serialize deterministically.

    ``sort_keys`` fixes ordering, ``separators`` removes incidental whitespace,
    and ``allow_nan=False`` rejects NaN/Infinity, which are not valid JSON and
    would otherwise serialize into something no other parser accepts.
    """
    try:
        return json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
    except ValueError as exc:
        raise ManifestError(f"payload is not canonically serializable: {exc}") from exc


@dataclass(frozen=True, slots=True)
class AssetRecord:
    """One stored artifact and the claim we make about it."""

    key: str
    sha256: str
    size: int
    namespace: str
    content_type: str
    deduplicated: bool = False


@dataclass(frozen=True, slots=True)
class StepRecord:
    """One pipeline step, recorded for reproducibility.

    ``attempts`` is deliberately included: a description accepted on the third
    try is a materially different provenance claim from one accepted on the
    first, and an auditor needs to see the rejected drafts.
    """

    index: int
    name: str
    provider: str
    model: str
    model_version: str = ""
    seed: int | None = None
    params: dict[str, Any] = field(default_factory=dict)
    input_keys: list[str] = field(default_factory=list)
    output_keys: list[str] = field(default_factory=list)
    attempts: int = 1
    accepted: bool = True
    duration_ms: int = 0


@dataclass(frozen=True, slots=True)
class TokenUsage:
    """Measured token consumption, with and without compression.

    Reported as raw counts rather than a percentage so a reader can recompute
    the ratio and check our arithmetic.
    """

    prompt_tokens_uncompressed: int = 0
    prompt_tokens_compressed: int = 0
    completion_tokens: int = 0
    calls: int = 0

    @property
    def tokens_saved(self) -> int:
        return max(
            0, self.prompt_tokens_uncompressed - self.prompt_tokens_compressed
        )

    @property
    def reduction_ratio(self) -> float:
        """Fraction of prompt tokens eliminated. Zero when nothing was measured."""
        if self.prompt_tokens_uncompressed <= 0:
            return 0.0
        return self.tokens_saved / self.prompt_tokens_uncompressed


@dataclass(slots=True)
class Manifest:
    """A complete, hash-verifiable provenance record for one run."""

    run_id: str
    source_key: str
    source_sha256: str
    created_at: str
    steps: list[StepRecord] = field(default_factory=list)
    assets: list[AssetRecord] = field(default_factory=list)
    tokens: TokenUsage = field(default_factory=TokenUsage)
    schema_version: str = SCHEMA_VERSION
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def new(cls, run_id: str, source_key: str, source_sha256: str) -> Manifest:
        if not run_id:
            raise ManifestError("run_id must not be empty")
        return cls(
            run_id=run_id,
            source_key=source_key,
            source_sha256=source_sha256,
            created_at=datetime.now(UTC).isoformat(timespec="seconds"),
        )

    def add_step(self, step: StepRecord) -> None:
        self.steps.append(step)

    def add_asset(self, asset: AssetRecord) -> None:
        self.assets.append(asset)

    def to_payload(self) -> dict[str, Any]:
        """The hashable body, excluding the hash itself."""
        return {
            "schema_version": self.schema_version,
            "run_id": self.run_id,
            "created_at": self.created_at,
            "source": {"key": self.source_key, "sha256": self.source_sha256},
            "steps": [asdict(s) for s in self.steps],
            "assets": [asdict(a) for a in self.assets],
            "tokens": {
                **asdict(self.tokens),
                "tokens_saved": self.tokens.tokens_saved,
                "reduction_ratio": round(self.tokens.reduction_ratio, 6),
            },
            "metadata": self.metadata,
        }

    def canonical_hash(self) -> str:
        return hashlib.sha256(canonical_json(self.to_payload()).encode("utf-8")).hexdigest()

    def to_json(self) -> str:
        """The full document: body plus its own hash."""
        payload = self.to_payload()
        payload["canonical_hash"] = self.canonical_hash()
        return json.dumps(payload, sort_keys=True, indent=2, ensure_ascii=False)


@dataclass(frozen=True, slots=True)
class ManifestVerification:
    """Outcome of checking a manifest against reality."""

    hash_ok: bool
    missing_assets: list[str] = field(default_factory=list)
    corrupt_assets: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return self.hash_ok and not self.missing_assets and not self.corrupt_assets


def parse_manifest(document: str) -> dict[str, Any]:
    """Load a manifest document, raising on malformed input."""
    try:
        parsed = json.loads(document)
    except json.JSONDecodeError as exc:
        raise ManifestError(f"manifest is not valid JSON: {exc}") from exc
    if not isinstance(parsed, dict):
        raise ManifestError("manifest root must be an object")
    if "canonical_hash" not in parsed:
        raise ManifestError("manifest is missing canonical_hash")
    return parsed


def verify_hash(document: str) -> bool:
    """Recompute a manifest's hash and compare it to the recorded value."""
    parsed = parse_manifest(document)
    claimed = parsed.pop("canonical_hash")
    recomputed = hashlib.sha256(canonical_json(parsed).encode("utf-8")).hexdigest()
    return claimed == recomputed
