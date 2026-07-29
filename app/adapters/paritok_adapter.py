"""Sole entry point for the Paritok SDK.

Per the Unverified-API Protocol: no other module in this codebase imports
paritok. If the real API differs from what is assumed here, the blast radius is
this file alone.

VERIFIED (read from paritok 1.2.7 wheel source):
  * CompressionStats fields: original_tokens, compressed_tokens, items_compressed,
    items_skipped, cache_hits, tools_original, tools_kept, history_turns_compressed
  * CompressionStats.ratio == 1 - compressed/original  (fraction SAVED, higher better)
  * CompressionStats.saved_tokens, .tools_filtered
  * token_counter.count_tokens(text, model_or_encoding=...) -> int
  * Kill switch env var: PARITOK_DISABLE=1|true|yes
  * paritok.__version__ hard-codes "1.2.3" while METADATA says 1.2.7 (real bug)

ARCHITECTURAL FINDING — this changes the integration:
  ParitokClient wraps an *Anthropic-shaped* client only. It intercepts
  client.messages.create(**kwargs). There is NO generic httpx transport adapter
  and NO OpenAI-SDK wrapper in the library itself. OpenAI-shaped traffic is
  handled only by the standalone proxy (`paritok up`) via ANTHROPIC_BASE_URL /
  OPENAI_BASE_URL redirection.

  Interlude's description step targets Google Gemini, which is neither shape.
  Therefore we do NOT wrap the provider client. We use ParitokEngine directly on
  the prompt payload and measure with CompressionStats. That is both the honest
  integration and the one that yields a defensible measured number.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)

DISABLE_ENV = "PARITOK_DISABLE"


@dataclass(frozen=True, slots=True)
class CompressionOutcome:
    """Normalized result, decoupled from paritok's own types.

    The rest of the codebase depends on this shape, never on paritok's.
    """

    text: str
    original_tokens: int
    compressed_tokens: int
    applied: bool

    @property
    def saved_tokens(self) -> int:
        return max(0, self.original_tokens - self.compressed_tokens)

    @property
    def ratio(self) -> float:
        """Fraction of tokens eliminated. Matches paritok's own orientation."""
        if self.original_tokens <= 0:
            return 0.0
        return self.saved_tokens / self.original_tokens


def is_disabled() -> bool:
    return os.environ.get(DISABLE_ENV, "").strip().lower() in {"1", "true", "yes"}


def count_tokens(text: str) -> int:
    """Token count via paritok's counter, falling back to a labelled estimate."""
    try:
        from paritok.token_counter import count_tokens as _count

        return int(_count(text))
    except Exception:  # noqa: BLE001 - counter is advisory, never fatal
        return max(0, (len(text) + 3) // 4)


# ==== UNVERIFIED API — CONFIRM AGAINST REAL DOCS BEFORE RUNNING ====
# Assumed:    ParitokEngine() constructible with no required args, exposing a
#             compress(text: str) -> object with .compressed / .original_tokens
#             / .compressed_tokens (mirroring CompressionResult's verified fields).
# Basis:      CompressionResult field names ARE verified from source
#             (compressed, original_tokens, compressed_tokens, shadow_id, metadata).
#             The ENGINE ENTRY POINT that produces one is not yet confirmed.
# Confidence: medium
# Blast radius if wrong: this file only.
def compress_prefix(text: str) -> CompressionOutcome:
    """Compress a repeated prompt prefix, measuring both arms.

    Returns the original text unchanged, with applied=False, whenever paritok is
    disabled or unavailable. Compression is an optimization; it must never be a
    single point of failure for the pipeline.
    """
    original = count_tokens(text)

    if is_disabled():
        return CompressionOutcome(text, original, original, applied=False)

    try:
        # VERIFIED against the 1.2.7 wheel: the engine has no compress().
        # The real entry point is process_request(messages, tools, upstream_model),
        # which returns (messages, tools, CompressionStats, stubbed_tools).
        from paritok import ParitokEngine

        engine = ParitokEngine()
        messages = [{"role": "user", "content": text}]
        compressed, _tools, stats, _stubbed = engine.process_request(messages, None, "")

        out = text
        if compressed and isinstance(compressed[0].get("content"), str):
            out = compressed[0]["content"]

        return CompressionOutcome(
            text=out,
            original_tokens=int(getattr(stats, "original_tokens", original) or original),
            compressed_tokens=int(getattr(stats, "compressed_tokens", original) or original),
            applied=True,
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("paritok unavailable, running uncompressed: %s", exc)
        return CompressionOutcome(text, original, original, applied=False)


def sdk_version() -> str:
    """Installed version.

    Uses importlib.metadata because paritok.__version__ is stale — it reports
    1.2.3 while the distribution is 1.2.7. Logged as SDK feedback.
    """
    try:
        from importlib.metadata import version

        return version("paritok")
    except Exception:  # noqa: BLE001
        return "unknown"


if __name__ == "__main__":
    assert count_tokens("") == 0
    assert count_tokens("hello world") > 0

    out = CompressionOutcome("x", original_tokens=100, compressed_tokens=25, applied=True)
    assert out.saved_tokens == 75, out.saved_tokens
    assert abs(out.ratio - 0.75) < 1e-9, out.ratio

    zero = CompressionOutcome("x", 0, 0, applied=False)
    assert zero.ratio == 0.0

    grew = CompressionOutcome("x", original_tokens=50, compressed_tokens=80, applied=True)
    assert grew.saved_tokens == 0, "negative savings must clamp, never flatter"

    os.environ[DISABLE_ENV] = "1"
    disabled = compress_prefix("some prompt text")
    assert not disabled.applied
    assert disabled.text == "some prompt text"
    del os.environ[DISABLE_ENV]

    print(f"paritok_adapter self-check OK (sdk={sdk_version()})")
