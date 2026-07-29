"""Paritok on the client path.

Two independent compression arms, reported separately:

  "prefix"  — the style guide prepended to every prompt, compressed before the
              call. Already implemented in paritok_adapter.compress_prefix.
  "client"  — this module. ParitokEngine.process_request() sitting on the
              message list itself, compressing conversation history and tool
              output through the SDK's own pipeline.

VERIFIED against the paritok 1.2.7 wheel (paritok/middleware/wrapper.py):

  ParitokEngine(config: ParitokConfig | None = None,
                storage: ShadowStorage | None = None)          # wrapper.py:75

  ParitokEngine.process_request(
      messages: list[dict],
      tools: list[dict] | None = None,
      upstream_model: str = "",
  ) -> tuple[list[dict], list[dict] | None, CompressionStats, list[dict]]
                                                                # wrapper.py:85

  CompressionStats fields (wrapper.py:36-47):
      original_tokens, compressed_tokens, items_compressed, items_skipped,
      cache_hits, tools_original, tools_kept, history_turns_compressed
  CompressionStats.saved_tokens  = original - compressed        # wrapper.py:49
  CompressionStats.ratio         = round(1 - compressed/original, 3)  # wrapper.py:53
      NOTE: ratio is the fraction SAVED. Higher is better. The proxy server's
      /stats endpoint uses the opposite orientation; do not conflate them.

CORRECTION TO A PRIOR ASSUMPTION:
  app/adapters/paritok_adapter.compress_prefix() assumes ParitokEngine exposes
  ``compress(text) -> result``. It does not. There is no such method. The engine's
  entry point is process_request() and it operates on a MESSAGE LIST, not a bare
  string. compress_prefix() therefore always lands in its own except branch and
  returns applied=False — the prefix arm has been silently inert.
  Tracked in PARITOK_FEEDBACK.md; fixed by routing the prefix through this module.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)

DISABLE_ENV = "PARITOK_DISABLE"


@dataclass(frozen=True, slots=True)
class ClientCompression:
    """Normalized result of one client-arm compression.

    Decoupled from paritok's own types so the rest of the codebase never
    imports the SDK.
    """

    messages: list[dict[str, Any]]
    original_tokens: int
    compressed_tokens: int
    items_compressed: int
    applied: bool

    @property
    def saved_tokens(self) -> int:
        """Never negative. Compression that inflated the payload reports zero."""
        return max(0, self.original_tokens - self.compressed_tokens)

    @property
    def ratio(self) -> float:
        if self.original_tokens <= 0:
            return 0.0
        return self.saved_tokens / self.original_tokens


def is_disabled() -> bool:
    return os.environ.get(DISABLE_ENV, "").strip().lower() in {"1", "true", "yes"}


def as_messages(prompt: str) -> list[dict[str, Any]]:
    """Wrap a bare prompt in the message shape process_request() expects.

    The engine operates on Anthropic/OpenAI-style message lists. A single
    prompt becomes a one-turn user message.
    """
    return [{"role": "user", "content": prompt}]


def compress_messages(
    messages: list[dict[str, Any]],
    *,
    tools: list[dict[str, Any]] | None = None,
    upstream_model: str = "",
) -> ClientCompression:
    """Run the client arm.

    Returns the input unchanged with ``applied=False`` whenever Paritok is
    disabled or unavailable. Compression is an optimization and must never be a
    single point of failure for the pipeline.
    """
    if not messages:
        return ClientCompression([], 0, 0, 0, applied=False)

    if is_disabled():
        return ClientCompression(messages, 0, 0, 0, applied=False)

    try:
        from paritok import ParitokEngine

        engine = ParitokEngine()
        compressed, _tools, stats, _stubbed = engine.process_request(
            messages, tools, upstream_model
        )
        return ClientCompression(
            messages=compressed,
            original_tokens=int(getattr(stats, "original_tokens", 0) or 0),
            compressed_tokens=int(getattr(stats, "compressed_tokens", 0) or 0),
            items_compressed=int(getattr(stats, "items_compressed", 0) or 0),
            applied=True,
        )
    except Exception as exc:  # noqa: BLE001 - optimization, never fatal
        logger.warning("paritok client arm unavailable, passing through: %s", exc)
        return ClientCompression(messages, 0, 0, 0, applied=False)


def compress_prompt(prompt: str, *, upstream_model: str = "") -> tuple[str, ClientCompression]:
    """Compress a single prompt through the client arm.

    Returns the possibly-rewritten prompt alongside the measurement. Falls back
    to the original text if the engine returns an unexpected shape.
    """
    result = compress_messages(as_messages(prompt), upstream_model=upstream_model)

    text = prompt
    if result.applied and result.messages:
        content = result.messages[0].get("content")
        if isinstance(content, str) and content:
            text = content

    return text, result


if __name__ == "__main__":
    msgs = as_messages("describe this frame")
    assert msgs == [{"role": "user", "content": "describe this frame"}]

    empty = compress_messages([])
    assert not empty.applied
    assert empty.saved_tokens == 0

    inflated = ClientCompression([], original_tokens=50, compressed_tokens=80,
                                 items_compressed=1, applied=True)
    assert inflated.saved_tokens == 0, "inflation must clamp, never flatter"
    assert inflated.ratio == 0.0

    normal = ClientCompression([], original_tokens=1000, compressed_tokens=250,
                               items_compressed=3, applied=True)
    assert normal.saved_tokens == 750
    assert abs(normal.ratio - 0.75) < 1e-9

    zero = ClientCompression([], 0, 0, 0, applied=False)
    assert zero.ratio == 0.0

    os.environ[DISABLE_ENV] = "1"
    off = compress_messages(as_messages("x"))
    assert not off.applied, "kill switch must bypass the engine"
    text, meas = compress_prompt("hello world")
    assert text == "hello world"
    assert not meas.applied
    del os.environ[DISABLE_ENV]

    print("paritok_client self-check OK")
